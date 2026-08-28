# vram_benchmark.py
#
# Sweep di profilazione memoria/tempo di CogVideoX-2b al variare di num_frames.
#
# Struttura driver/worker (vedi run_driver / run_worker sotto): il driver lancia
# un subprocess per ogni punto dello sweep. Serve perche' l'esaurimento della RAM
# host fa intervenire l'OOM killer del kernel, che termina il processo con
# SIGKILL: un segnale non intercettabile da Python, quindi nessun try/except puo'
# salvare il punto ne' i punti successivi. Isolando ogni punto, la morte del
# figlio diventa un dato osservabile dal padre invece della fine dello sweep.
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
from diffusers import CogVideoXPipeline

MODEL_ID = "THUDM/CogVideoX-2b"

# Stessa risoluzione nativa e stessa configurazione VRAM-saving di smoke_test.py.
HEIGHT, WIDTH = 480, 720
NUM_STEPS = 50    # step di riferimento (quelli di una generazione reale)
GUIDANCE = 6.0
FPS = 8

# Lo sweep gira a pochi step invece dei 50 di riferimento: il picco di memoria si
# raggiunge entro il primo step (gli step successivi riusano gli stessi buffer),
# quindi 50 step moltiplicherebbero il wall time senza aggiungere informazione.
# Da SWEEP_STEPS si ricava il tempo per step e si estrapola linearmente il tempo
# che avrebbero richiesto NUM_STEPS step, mantenendo i risultati confrontabili.
# Estrapolazione validata contro misure reali a 50 step: errore fra 0% e 9%.
SWEEP_STEPS = 3

# Punti dello sweep: durata = num_frames / FPS, da ~1s al nativo (49 = 6.125s).
FRAME_POINTS = [9, 17, 25, 33, 41, 49]

BUDGET_S = 300   # obiettivo: 5 minuti per video
TARGET_S = 2.0   # obiettivo: scenari da ~2s

# Seed fisso: senza, due run dello stesso punto partono da rumore iniziale diverso
# e tempi/picchi non sono rigorosamente confrontabili.
SEED = 0

# Soglia di RAM host sotto la quale il driver non tenta nemmeno il punto: meglio
# registrarlo come non tentabile che farsi uccidere il processo dall'OOM killer.
HOST_RAM_MIN_GB = 4.0

# Tetto per singolo punto. Un punto sano costa ~10-30 s con SWEEP_STEPS=3 piu' il
# caricamento del modello; se ci mette dieci volte tanto sta quasi certamente
# girando su RAM di sistema.
WORKER_TIMEOUT_S = 600

# Un file nuovo per ogni run invece di sovrascrivere: il codice precedente apriva
# il CSV in "w", quindi ogni avvio cancellava i dati del run precedente.
OUTPUT_CSV = "outputs/vram_sweep_%s.csv" % datetime.now().strftime("%Y%m%d_%H%M%S")

# Il fallback sysmem resta una limitazione nota dell'ambiente: si esegue sotto
# WSL2 perche' su Windows nativo Smart App Control blocca le estensioni native
# del tokenizer (sentencepiece e tiktoken, entrambe verificate), ma sotto WSL2 il
# driver ignora "CUDA - Sysmem Fallback Policy" (microsoft/WSL#11050, chiusa senza
# fix), quindi un esaurimento di VRAM non produce un torch.cuda.OutOfMemoryError
# ma uno swap silenzioso su RAM via PCIe. Per questo il fallback viene rilevato
# dai numeri di memoria (sysmem_fallback_suspected) e non atteso come eccezione.
#
# torch.cuda.set_per_process_memory_fraction agirebbe solo sull'allocatore interno
# di PyTorch, non sul driver, quindi non impedirebbe comunque il fallback: resta
# commentato perche' limiterebbe la VRAM utilizzabile senza dare in cambio l'OOM
# vero.
#torch.cuda.set_per_process_memory_fraction(0.90, device=0)

prompt = (
    "POV shot from a person walking along a city sidewalk on a sunny afternoon, "
    "steady forward motion, parked cars lining the left side of the street, "
    "shop fronts and awnings on the right, pedestrians ahead in the distance, "
    "sharp focus, natural daylight, realistic urban environment, handheld camera feel,"
    "mass presence of trees, poles and suspended branches,"
)

negative_prompt = None

# --- strumentazione della memoria -------------------------------------------
# Cinque misure, tre di VRAM e due di RAM host, perche' raccontano cose diverse
# e la loro divergenza e' l'informazione utile:
#   alloc_peak    : picco dei tensori effettivamente in uso dall'allocatore PyTorch
#   reserved_peak : picco della VRAM che PyTorch ha prenotato al driver (>= alloc,
#                   include la cache che trattiene per riuso)
#   device_used   : VRAM occupata sulla scheda intera, vista dal driver (include
#                   altri processi); e' una lettura ISTANTANEA, non un picco
#   host_rss      : RAM di sistema occupata da questo processo
#   host_avail    : RAM di sistema ancora libera nella VM
# Le due misure host servono a distinguere un OOM di VRAM da un OOM di RAM: il
# kill a 25 frame del run precedente era il secondo, non il primo.
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
    misurarle insieme e' l'errore che ha reso inutile il run precedente), e
    registra l'istante di ogni step per ricavare il tempo medio per step."""

    def probe(pipe, step_index, timestep, callback_kwargs):
        out["step_times"].append(time.time())
        if step_index == total_steps - 1:
            out["denoise"] = mem_snapshot()
        return callback_kwargs

    return probe


def estimate_full_run(elapsed, step_times, sweep_steps, target_steps):
    """Estrapola il tempo di una generazione a target_steps step da una misurata a
    sweep_steps. Il tempo si scompone in una parte proporzionale agli step
    (denoising) e una fissa (setup + decode VAE), che non va moltiplicata.
    Gli intervalli fra step consecutivi escludono gia' il warm-up che precede il
    primo step, quindi danno il costo marginale pulito di uno step."""
    deltas = [b - a for a, b in zip(step_times, step_times[1:])]
    if not deltas or elapsed is None:
        return None, None
    s_per_step = sum(deltas) / len(deltas)
    overhead = elapsed - sweep_steps * s_per_step
    return s_per_step, overhead + target_steps * s_per_step


def sysmem_fallback_suspected(snap, vram_total_gb):
    """Un picco dell'allocatore che eguaglia o supera la capacita' fisica della
    scheda puo' essere servito solo spillando su RAM di sistema: e' il segnale
    diretto del sysmem fallback. Si usa questo criterio e non la differenza
    alloc_peak - device_used perche' quest'ultima diverge di ~2.5 GB anche in
    condizioni sane (alloc_peak e' un picco, device_used una lettura istantanea
    presa a modelli gia' scaricati dalla GPU), quindi darebbe falsi positivi."""
    if snap is None:
        return False
    return snap["vram_alloc_peak_gb"] >= vram_total_gb * 0.98


def failed_result(num_frames, fail_reason):
    """Riga per un punto che non ha prodotto misure: il worker e' morto, e' andato
    in timeout, o non e' stato nemmeno tentato."""
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
    vram_total_gb = torch.cuda.mem_get_info()[1] / 1024**3

    # Model card THUDM: CogVideoX-2b raccomanda fp16 (a differenza del 5b, in bf16).
    pipe = CogVideoXPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)

    # Peso dei tre componenti: numero di parametri e occupazione dei soli pesi in
    # fp16 (2 byte per parametro, escluse attivazioni e buffer temporanei). Serve
    # a verificare quale componente domina il picco: quello del run originale
    # (10.82 GB, identico a 9 e 17 frame) non dipendeva da num_frames perche'
    # apparteneva al text encoder T5, non alla parte che scala coi frame.
    for nome in ("text_encoder", "transformer", "vae"):
        m = getattr(pipe, nome)
        n = sum(p.numel() for p in m.parameters())
        print("%s: %.2fB params, %.2f GB in fp16" % (nome, n / 1e9, n * 2 / 1024**3),
              file=sys.stderr)

    # Il text encoder T5 (8.87 GB di soli pesi) serve una volta sola e il suo
    # risultato non dipende da num_frames: si calcolano gli embedding subito e poi
    # lo si elimina, cosi' non pesa piu' sulla misura.
    # encode_prompt con negative_prompt=None e do_classifier_free_guidance=True usa
    # internamente la stringa vuota (negative_prompt or ""), quindi il risultato e'
    # identico a quello che si otteneva passando negative_prompt=None a pipe().
    # no_grad e' obbligatorio: solo pipe.__call__ e' decorato @torch.no_grad, mentre
    # encode_prompt chiamato direttamente costruisce il grafo di autograd. Il grafo
    # trattiene sia le attivazioni di T5 (~3.6 GB) sia i suoi pesi, rendendo
    # inefficace il rilascio sotto: misurati 14.37 GB ancora occupati dopo il free,
    # col picco dello sweep a 18.21 GB su una scheda da 16.3 GB (quindi sysmem
    # fallback e 98 s/step invece di 1.7 s/step).
    pipe.text_encoder.to("cuda")
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            device=torch.device("cuda"),
            dtype=torch.float16,
        )

    # Liberare davvero il text encoder richiede tre passaggi: togliere il riferimento
    # (= None), forzare il garbage collector a distruggere l'oggetto (gc.collect) e
    # restituire al driver la VRAM che l'allocatore PyTorch teneva in cache
    # (empty_cache). enable_model_cpu_offload va chiamato DOPO: installa hook sui
    # componenti registrati e li sposta su CPU, quindi deve trovare il text encoder
    # gia' sparito. Non serve toccare pipe.model_cpu_offload_seq: il loop di
    # enable_model_cpu_offload salta i componenti che non sono piu' nn.Module.
    pipe.text_encoder = None
    gc.collect()
    torch.cuda.empty_cache()

    pipe.enable_model_cpu_offload()
    # tiling: spezza ogni frame in riquadri; slicing: decodifica un frame per volta.
    # Sono complementari, entrambi riducono il picco della fase di decode VAE.
    pipe.vae.enable_tiling()
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

    # Campione (b): la fase di decode e' gia' conclusa. Attenzione in lettura:
    # qui device_used arriva DOPO che maybe_free_model_hooks ha scaricato i
    # moduli dalla GPU, quindi sottostima il transiente del decode; per quella
    # fase il dato affidabile e' vram_alloc_peak_gb, che e' un picco vero.
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
            sysmem_fallback_suspected(samples["denoise"], vram_total_gb)
            or sysmem_fallback_suspected(samples["decode"], vram_total_gb)
        ),
        "oom": oom,
        "fail_reason": fail_reason,
    }


# --- driver: uno subprocess per punto ---------------------------------------
def run_point_isolated(num_frames):
    """Lancia il worker in un processo separato e traduce l'esito in un risultato.
    returncode 0 = ok; negativo = terminato da segnale (per convenzione -N dove N
    e' il numero del segnale, quindi -9 e' SIGKILL, cioe' l'OOM killer); positivo
    = il worker ha sollevato un'eccezione Python."""
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
              % (proc.returncode, proc.stderr.strip()[-2000:]), flush=True)
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
        description="Sweep VRAM/tempo di CogVideoX-2b al variare di num_frames.")
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
