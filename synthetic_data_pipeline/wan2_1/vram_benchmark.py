# vram_benchmark.py
#
# Sweep di profilazione memoria/tempo di Wan2.1-T2V-1.3B al variare di num_frames.
#
# Perche' esiste questa cartella accanto a wan2_2/: Wan2.2-TI2V-5B richiede >=24 GB
# di VRAM per model card e su questa scheda (16.3 GB) va in OOM vero prima di
# raggiungere il proprio punto nativo. Wan2.1-T2V-1.3B e' il Wan piu' piccolo
# disponibile (non esiste un Wan2.2 sotto i 5B) ed e' l'unico che su questo hardware
# possa competere. In piu' il suo nativo, 832x480, sta a +15% di pixel per frame da
# CogVideoX (480x720): e' il primo confronto a parita' di risoluzione, quindi il primo
# in cui la differenza misurata e' attribuibile ai modelli e non ai loro punti operativi.
#
# Struttura driver/worker identica a quella di cogvideox/ e wan2_2/, cosi' come i nomi
# delle colonne del CSV: i tre sweep vanno letti affiancati.
import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import psutil
import torch
from diffusers import AutoencoderKLWan, WanPipeline

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# Risoluzione nativa del modello: 480P, cioe' 832x480. Il 720p e' documentato come
# "meno stabile" su questa taglia, quindi si resta al nativo.
# Vincolo di forma: multipli di vae_scale_factor_spatial x patch_size del transformer.
# ATTENZIONE, qui il fattore e' diverso da Wan2.2: il VAE di Wan2.1 comprime 8x sullo
# spazio (non 16x), quindi il vincolo e' 8 x 2 = 16. 480 e 832 sono entrambi multipli.
HEIGHT, WIDTH = 480, 832
NUM_STEPS = 50    # step di riferimento (quelli di una generazione reale)
GUIDANCE = 5.0    # default della model card diffusers

# Frame rate nativo. La model card diffusers del 1.3B esporta a fps=15, ma il repo
# upstream Wan-Video/Wan2.1 usa 16: si segue l'upstream, perche' e' il frame rate di
# addestramento ed esportare piu' lento falsa il moto. La discrepanza vale un 6% sulle
# durate e non sposta nessuna conclusione, ma la scelta e' consapevole.
FPS = 16

# Lo sweep gira a pochi step invece dei 50 di riferimento: il picco di memoria si
# raggiunge entro il primo step (gli step successivi riusano gli stessi buffer),
# quindi 50 step moltiplicherebbero il wall time senza aggiungere informazione.
# Da SWEEP_STEPS si ricava il tempo per step e si estrapola linearmente il tempo che
# avrebbero richiesto NUM_STEPS step. Stesso valore usato per gli altri due modelli,
# dove l'estrapolazione e' stata validata contro misure reali a 50 step.
SWEEP_STEPS = 3

# Punti dello sweep: durata = num_frames / FPS, da ~1s al nativo (81 = 5.06s).
# Vincolo del VAE: num_frames deve essere 4k+1 (vae_scale_factor_temporal = 4),
# altrimenti pipeline_wan.py arrotonda al valore valido piu' vicino con un warning.
# La griglia si ferma al nativo: oltre, il modello va fuori distribuzione.
FRAME_POINTS = [17, 33, 49, 65, 81]

BUDGET_S = 300   # obiettivo: 5 minuti per video
TARGET_S = 2.0   # obiettivo: scenari da ~2s

# Seed fisso: senza, due run dello stesso punto partono da rumore iniziale diverso e
# tempi/picchi non sono rigorosamente confrontabili. Stesso valore dello smoke test e
# degli altri due modelli.
SEED = 0

# Tetto imposto all'allocatore PyTorch, come frazione della VRAM fisica. Stessa
# ragione documentata per wan2_2: su Windows nativo il driver rispetta la Sysmem
# Fallback Policy, quindi un tetto esplicito fa fallire l'allocazione PRIMA del
# cudaMalloc che innescherebbe lo swap silenzioso su RAM, e l'OOM diventa un
# OutOfMemoryError vero e intercettabile. Sotto WSL2 (dove gira CogVideoX) sarebbe
# inutile, perche' li' il driver la policy la ignora.
VRAM_FRACTION = 0.90

# Soglia di RAM host sotto la quale il driver non tenta nemmeno il punto. Piu' bassa
# dei 16.0 GB di wan2_2 perche' qui il caricamento della pipeline completa e' un
# transiente da ~14.5 GB (text encoder UMT5-XXL 10.58 + transformer ~2.6 + VAE) contro
# i ~22.5 GB del 5B. Su Windows non esiste l'OOM killer: il sistema spilla sul pagefile
# e il sintomo e' lentezza, quindi la soglia serve a non tentare punti che thrasherebbero.
HOST_RAM_MIN_GB = 12.0

# Tetto per singolo punto. Meta' di quello di wan2_2: il transformer e' 1.3B invece di
# 5B e i punti costano molto meno.
WORKER_TIMEOUT_S = 900

# Un file nuovo per ogni run invece di sovrascrivere.
OUTPUT_CSV = "outputs/vram_sweep_%s.csv" % datetime.now().strftime("%Y%m%d_%H%M%S")

# Stesso prompt positivo di cogvideox/ e wan2_2/, per confronto diretto.
prompt = (
    "POV shot from a person walking along a city sidewalk on a sunny afternoon, "
    "steady forward motion, parked cars lining the left side of the street, "
    "shop fronts and awnings on the right, pedestrians ahead in the distance, "
    "sharp focus, natural daylight, realistic urban environment, handheld camera feel,"
    "mass presence of trees, poles and suspended branches,"
)

# Negative prompt ufficiale del modello. La card del 1.3B lo riporta in inglese, quella
# di Wan2.2-TI2V-5B in cinese: sono la traduzione l'una dell'altra, stessa lista di
# concetti, quindi la differenza non introduce una variabile di confronto. Si usa quello
# della rispettiva card. CogVideoX non ha un negative prompt ufficiale e usa None.
negative_prompt = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly "
    "drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, many people in the background, walking backwards"
)

# --- strumentazione della memoria -------------------------------------------
# Cinque misure, tre di VRAM e due di RAM host, perche' raccontano cose diverse
# e la loro divergenza e' l'informazione utile:
#   alloc_peak    : picco dei tensori effettivamente in uso dall'allocatore PyTorch
#   reserved_peak : picco della VRAM che PyTorch ha prenotato al driver (>= alloc,
#                   include la cache che trattiene per riuso)
#   device_used   : VRAM occupata sulla scheda intera, vista dal driver (include altri
#                   processi, desktop Windows compreso); lettura ISTANTANEA, non un picco
#   host_rss      : RAM di sistema occupata da questo processo
#   host_avail    : RAM di sistema ancora libera
PROC = psutil.Process()

MEM_FIELDS = (
    "vram_alloc_peak_gb",
    "vram_reserved_peak_gb",
    "vram_device_used_gb",
    "host_rss_gb",
    "host_avail_gb",
)

CSV_HEADER = (
    ["num_frames", "duration_s", "elapsed_s", "s_per_step", "est_elapsed_%dsteps_s" % NUM_STEPS]
    + [c + "_denoise" for c in MEM_FIELDS]
    + [c + "_decode" for c in MEM_FIELDS]
    + ["sysmem_fallback_suspected", "oom", "fail_reason"]
)


def mem_snapshot():
    free, total = torch.cuda.mem_get_info()
    return {
        "vram_alloc_peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "vram_reserved_peak_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "vram_device_used_gb": (total - free) / 1024**3,
        "host_rss_gb": PROC.memory_info().rss / 1024**3,
        "host_avail_gb": psutil.virtual_memory().available / 1024**3,
    }


def make_step_probe(total_steps, out):
    """Campiona la memoria all'ultimo step di denoising, cioe' prima che parta il
    decode VAE (le due fasi hanno profili di memoria completamente diversi, e
    misurarle insieme e' l'errore che ha reso inutile il primo sweep di CogVideoX),
    e registra l'istante di ogni step per ricavare il tempo medio per step."""

    def probe(pipe, step_index, timestep, callback_kwargs):
        out["step_times"].append(time.time())
        if step_index == total_steps - 1:
            out["denoise"] = mem_snapshot()
        return callback_kwargs

    return probe


def estimate_full_run(elapsed, step_times, sweep_steps, target_steps):
    """Estrapola il tempo di una generazione a target_steps step da una misurata a
    sweep_steps. Il tempo si scompone in una parte proporzionale agli step (denoising)
    e una fissa (setup + decode VAE), che non va moltiplicata. Gli intervalli fra step
    consecutivi escludono gia' il warm-up che precede il primo step, quindi danno il
    costo marginale pulito di uno step. Attenzione in lettura: uno step di Wan
    comprende DUE forward del transformer, condizionato e non condizionato, eseguiti in
    sequenza (pipeline_wan.py li chiama separatamente invece di impilarli in un batch
    2N come fa CogVideoX)."""
    deltas = [b - a for a, b in zip(step_times, step_times[1:])]
    if not deltas or elapsed is None:
        return None, None
    s_per_step = sum(deltas) / len(deltas)
    overhead = elapsed - sweep_steps * s_per_step
    return s_per_step, overhead + target_steps * s_per_step


def sysmem_fallback_suspected(snap, vram_budget_gb):
    """Un picco dell'allocatore che eguaglia il budget disponibile puo' essere servito
    solo spillando su RAM di sistema. Con VRAM_FRACTION attiva il segnale atteso e'
    l'OOM vero (colonna oom) e non questo, perche' l'allocatore rifiuta prima di
    arrivarci; l'euristica resta come rete di sicurezza per le allocazioni che NON
    passano dall'allocatore PyTorch (workspace di cuDNN/cuBLAS), che il tetto non copre.
    Il confronto e' col budget effettivo e non con la capacita' fisica della scheda:
    col tetto attivo alloc_peak non puo' piu' raggiungere il 98% del totale fisico."""
    if snap is None:
        return False
    return snap["vram_alloc_peak_gb"] >= vram_budget_gb * 0.98


def failed_result(num_frames, fail_reason):
    """Riga per un punto che non ha prodotto misure: il worker e' morto, e' andato in
    timeout, o non e' stato nemmeno tentato."""
    return {
        "num_frames": num_frames,
        "duration_s": num_frames / FPS,
        "elapsed_s": None,
        "s_per_step": None,
        "est_elapsed_full_s": None,
        "denoise": None,
        "decode": None,
        "sysmem_fallback_suspected": False,
        "oom": False,
        "fail_reason": fail_reason,
    }


def result_to_row(res):
    def fmt(value, spec):
        return (spec % value) if value is not None else ""

    row = [
        res["num_frames"],
        "%.3f" % res["duration_s"],
        fmt(res["elapsed_s"], "%.1f"),
        fmt(res["s_per_step"], "%.3f"),
        fmt(res["est_elapsed_full_s"], "%.1f"),
    ]
    for phase in ("denoise", "decode"):
        snap = res[phase]
        row += [fmt(snap[c], "%.2f") if snap else "" for c in MEM_FIELDS]
    row += [res["sysmem_fallback_suspected"], res["oom"], res["fail_reason"]]
    return row


# --- worker: un solo punto dello sweep --------------------------------------
def run_worker(num_frames):
    """Esegue UN punto e restituisce il risultato come dizionario serializzabile.
    La diagnostica va su stderr, cosi' stdout resta parsabile dal driver."""
    # Va chiamata prima di qualunque allocazione, e nel worker e non nel driver:
    # inizializza il contesto CUDA, e il driver non deve toccarlo.
    torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION, device=0)
    vram_total_gb = torch.cuda.mem_get_info()[1] / 1024**3
    vram_budget_gb = vram_total_gb * VRAM_FRACTION

    # Il VAE resta in fp32 (in bf16 produce artefatti di decodifica); transformer e
    # text encoder in bf16.
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)

    # L'occupazione si somma parametro per parametro invece di moltiplicare per un
    # dtype unico: dentro il transformer convivono due dtype, perche' diffusers tiene
    # in fp32 i moduli elencati in _keep_in_fp32_modules anche quando la pipeline e'
    # caricata in bf16. Leggere il dtype dal primo parametro darebbe un'etichetta
    # sbagliata per l'intero modello.
    for nome in ("text_encoder", "transformer", "vae"):
        m = getattr(pipe, nome)
        n = sum(p.numel() for p in m.parameters())
        b = sum(p.numel() * p.element_size() for p in m.parameters())
        dtypes = sorted({str(p.dtype).replace("torch.", "") for p in m.parameters()})
        print("%s: %.2fB params, %.2f GB, dtype %s"
              % (nome, n / 1e9, b / 1024**3, "+".join(dtypes)),
              file=sys.stderr)

    # Il text encoder UMT5-XXL (10.58 GB di soli pesi) e' lo STESSO di Wan2.2 ed e' qui
    # il componente singolo piu' pesante, molto piu' del transformer da 1.3B. Serve una
    # volta sola e il suo risultato non dipende da num_frames: si calcolano gli
    # embedding subito e poi lo si elimina, cosi' non pesa piu' sulla misura.
    # Il transformer invece non si puo' eliminare allo stesso modo: viene invocato a
    # ogni step, due volte per step per via del CFG. Per lui la leva e'
    # enable_model_cpu_offload, che lo tiene in RAM host e lo porta su GPU solo mentre
    # serve, restituendo poi la VRAM al VAE per il decode.
    #
    # no_grad e' obbligatorio: solo pipe.__call__ e' decorato @torch.no_grad, mentre
    # encode_prompt chiamato direttamente costruisce il grafo di autograd, che
    # trattiene sia le attivazioni del text encoder sia i suoi pesi e rende inefficace
    # il rilascio sotto.
    pipe.text_encoder.to("cuda")
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=True,
            # WanPipeline.__call__ onora davvero num_videos_per_prompt (lo passa a
            # encode_prompt e a prepare_latents), a differenza di CogVideoX dove viene
            # riassegnato a 1 in silenzio: qui e li' sono equivalenti.
            num_videos_per_prompt=1,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )

    # Liberare davvero il text encoder richiede tre passaggi: togliere il riferimento
    # (= None), forzare il garbage collector a distruggere l'oggetto (gc.collect) e
    # restituire al driver la VRAM che l'allocatore PyTorch teneva in cache
    # (empty_cache). enable_model_cpu_offload va chiamato DOPO: installa hook sui
    # componenti registrati e li sposta su CPU, quindi deve trovare il text encoder
    # gia' sparito. Il suo loop salta i componenti che non sono piu' nn.Module.
    pipe.text_encoder = None
    gc.collect()
    torch.cuda.empty_cache()

    pipe.enable_model_cpu_offload()
    # tiling: spezza ogni frame in riquadri. La soglia e' 256 pixel / 8 di compressione
    # spaziale = 32 in latente, e a 832x480 il latente e' 104x60: attivo.
    # (In wan2_2 la stessa soglia dava 16, perche' li' la compressione e' 16x.)
    pipe.vae.enable_tiling()
    # slicing: spezza il decode lungo la dimensione BATCH, non lungo i frame. A
    # num_videos_per_prompt=1 il latente in ingresso al decode ha batch 1 e la riga e'
    # inerte, perche' il VAE la applica solo se z.shape[0] > 1; si tiene per parita'
    # con smoke_test.py e con gli altri due benchmark. Da non confondere con lo
    # spezzettamento temporale del decode, che nel VAE di Wan e' sempre attivo e piu'
    # fine: un frame latente per volta, con lo stato tenuto nel feat_cache.
    pipe.vae.enable_slicing()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    oom = False
    fail_reason = ""
    elapsed = None
    samples = {"denoise": None, "decode": None, "step_times": []}
    t0 = time.time()
    try:
        frames = pipe(
            # prompt e negative_prompt vanno a None: check_inputs solleva se si passano
            # insieme alla loro versione precalcolata.
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            height=HEIGHT,
            width=WIDTH,
            num_frames=num_frames,
            num_inference_steps=SWEEP_STEPS,
            guidance_scale=GUIDANCE,
            generator=torch.Generator(device="cuda").manual_seed(SEED),
            callback_on_step_end=make_step_probe(SWEEP_STEPS, samples),
        ).frames[0]
        elapsed = time.time() - t0
        del frames
    except torch.cuda.OutOfMemoryError:
        oom = True
        fail_reason = "cuda_oom"
    except RuntimeError as e:
        # in alcuni path di offload l'OOM risale come RuntimeError generico
        # invece del subtype dedicato
        if "out of memory" in str(e).lower():
            oom = True
            fail_reason = "cuda_oom"
        else:
            raise

    # Campione (b): la fase di decode e' gia' conclusa. Attenzione in lettura: qui
    # device_used arriva DOPO che maybe_free_model_hooks ha scaricato i moduli dalla
    # GPU, quindi sottostima il transiente del decode; per quella fase il dato
    # affidabile e' vram_alloc_peak_gb, che e' un picco vero.
    samples["decode"] = mem_snapshot()
    s_per_step, est_full = estimate_full_run(
        elapsed, samples["step_times"], SWEEP_STEPS, NUM_STEPS
    )

    return {
        "num_frames": num_frames,
        "duration_s": num_frames / FPS,
        "elapsed_s": elapsed,
        "s_per_step": s_per_step,
        "est_elapsed_full_s": est_full,
        "denoise": samples["denoise"],
        "decode": samples["decode"],
        "sysmem_fallback_suspected": bool(
            sysmem_fallback_suspected(samples["denoise"], vram_budget_gb)
            or sysmem_fallback_suspected(samples["decode"], vram_budget_gb)
        ),
        "oom": oom,
        "fail_reason": fail_reason,
    }


# --- driver: uno subprocess per punto ---------------------------------------
def run_point_isolated(num_frames):
    """Lancia il worker in un processo separato e traduce l'esito in un risultato.
    returncode 0 = ok; positivo = il worker ha sollevato un'eccezione Python.
    Il ramo negativo (terminato da segnale, -9 = SIGKILL dell'OOM killer) e' quello che
    salvava lo sweep di CogVideoX sotto Linux: su Windows non c'e' un OOM killer e resta
    di fatto irraggiungibile, si tiene per simmetria fra gli script."""
    avail_gb = psutil.virtual_memory().available / 1024**3
    if avail_gb < HOST_RAM_MIN_GB:
        print("  RAM host disponibile %.2f GB < %.2f GB: punto non tentato"
              % (avail_gb, HOST_RAM_MIN_GB), flush=True)
        return failed_result(num_frames, "host_ram_insufficient")

    cmd = [sys.executable, os.path.abspath(__file__), "--worker",
           "--num-frames", str(num_frames)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=WORKER_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print("  timeout dopo %ds" % WORKER_TIMEOUT_S, flush=True)
        return failed_result(num_frames, "timeout")

    if proc.returncode < 0:
        print("  ucciso dal sistema (segnale %d)" % -proc.returncode, flush=True)
        return failed_result(num_frames, "killed_by_os")
    if proc.returncode > 0:
        print("  worker uscito con codice %d:\n%s"
              % (proc.returncode, (proc.stderr or "").strip()[-2000:]), flush=True)
        return failed_result(num_frames, "worker_error")

    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        return failed_result(num_frames, "no_output")
    return json.loads(lines[-1])


def run_driver():
    os.makedirs("outputs", exist_ok=True)
    results = []

    csv_exists = os.path.exists(OUTPUT_CSV)
    # CSV scritto con flush dopo ogni riga: se il driver viene interrotto a mano,
    # i punti gia' misurati restano su disco invece di andare persi.
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(CSV_HEADER)
            f.flush()

        for num_frames in FRAME_POINTS:
            res = run_point_isolated(num_frames)

            den, dec = res["denoise"], res["decode"]
            print(
                "frames=%d durata=%.2fs elapsed=%s est_%dstep=%s | denoise: alloc=%s"
                " dev=%s host_avail=%s | decode: alloc=%s dev=%s host_avail=%s"
                " | fallback=%s fail=%s"
                % (
                    res["num_frames"],
                    res["duration_s"],
                    "%.1f" % res["elapsed_s"] if res["elapsed_s"] is not None else "FAIL",
                    NUM_STEPS,
                    "%.1f" % res["est_elapsed_full_s"] if res["est_elapsed_full_s"] is not None else "n/d",
                    "%.2f" % den["vram_alloc_peak_gb"] if den else "n/d",
                    "%.2f" % den["vram_device_used_gb"] if den else "n/d",
                    "%.2f" % den["host_avail_gb"] if den else "n/d",
                    "%.2f" % dec["vram_alloc_peak_gb"] if dec else "n/d",
                    "%.2f" % dec["vram_device_used_gb"] if dec else "n/d",
                    "%.2f" % dec["host_avail_gb"] if dec else "n/d",
                    res["sysmem_fallback_suspected"],
                    res["fail_reason"] or "-",
                ),
                flush=True,
            )

            writer.writerow(result_to_row(res))
            f.flush()
            results.append(res)

    first_oom = next((r for r in results if r["oom"]), None)
    first_failed = next((r for r in results if r["fail_reason"]), None)
    first_fallback = next((r for r in results if r["sysmem_fallback_suspected"]), None)
    first_over_budget = next(
        (r for r in results
         if r["est_elapsed_full_s"] is not None and r["est_elapsed_full_s"] > BUDGET_S),
        None,
    )

    viable = [
        r for r in results
        if not r["fail_reason"]
        and not r["sysmem_fallback_suspected"]
        and r["est_elapsed_full_s"] is not None
        and r["est_elapsed_full_s"] <= BUDGET_S
    ]
    recommended = min(viable, key=lambda r: abs(r["duration_s"] - TARGET_S)) if viable else None

    print("\n--- sintesi ---")
    print("CSV:", OUTPUT_CSV)
    print("primo OOM:", ("frames=%d" % first_oom["num_frames"]) if first_oom else "nessuno")
    print("primo fallito:",
          ("frames=%d (%s)" % (first_failed["num_frames"], first_failed["fail_reason"]))
          if first_failed else "nessuno")
    print("primo con sysmem fallback:",
          ("frames=%d" % first_fallback["num_frames"]) if first_fallback else "nessuno")
    print(
        "primo oltre %ds (stima a %d step):" % (BUDGET_S, NUM_STEPS),
        ("frames=%d (%.1fs)" % (first_over_budget["num_frames"],
                                first_over_budget["est_elapsed_full_s"]))
        if first_over_budget else "nessuno",
    )
    if recommended:
        print(
            "raccomandato: frames=%d durata=%.2fs tempo stimato a %d step=%.1fs"
            " picco_vram=%.2fGB"
            % (
                recommended["num_frames"],
                recommended["duration_s"],
                NUM_STEPS,
                recommended["est_elapsed_full_s"],
                recommended["denoise"]["vram_alloc_peak_gb"],
            )
        )
    else:
        print("raccomandato: nessun punto soddisfa i vincoli"
              " (fallito, fallback o oltre budget su tutta la griglia)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep VRAM/tempo di Wan2.1-T2V-1.3B al variare di num_frames.")
    parser.add_argument("--worker", action="store_true",
                        help="esegue un solo punto e stampa una riga JSON"
                             " (uso interno del driver)")
    parser.add_argument("--num-frames", type=int,
                        help="punto da misurare in modalita' worker")
    args = parser.parse_args()

    if args.worker:
        if args.num_frames is None:
            parser.error("--worker richiede --num-frames")
        print(json.dumps(run_worker(args.num_frames)))
    else:
        run_driver()
