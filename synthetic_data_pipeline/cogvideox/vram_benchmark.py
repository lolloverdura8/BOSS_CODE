# vram_benchmark.py
import csv
import os
import time

import torch
from diffusers import CogVideoXPipeline

MODEL_ID = "THUDM/CogVideoX-2b"

# Stessa risoluzione nativa e stessa configurazione VRAM-saving di smoke_test.py.
HEIGHT, WIDTH = 480, 720
NUM_STEPS = 50    # fisso, per coerenza col benchmark Wan2.2
GUIDANCE = 6.0
FPS = 8

# Punti dello sweep: durata = num_frames / FPS, da ~1s al nativo (49 = 6.125s).
FRAME_POINTS = [9, 17, 25, 33, 41, 49]

BUDGET_S = 300   # obiettivo: 5 minuti per video
TARGET_S = 2.0   # obiettivo: scenari da ~2s

OUTPUT_CSV = "outputs/vram_sweep.csv"

# Rilevamento indiretto del fallback VRAM->RAM: su questa macchina Smart App
# Control blocca il caricamento nativo del tokenizer (sentencepiece e tiktoken
# hanno entrambi l'estensione compilata bloccata, verificato) quindi si esegue
# sotto WSL2, dove pero' "CUDA - Sysmem Fallback Policy" viene ignorata dal
# driver e il fallback e' sempre attivo (microsoft/WSL#11050): non si puo'
# quindi contare su un vero torch.cuda.OutOfMemoryError. Il watchdog sotto
# misura il tempo per singolo step di denoising (via callback_on_step_end)
# invece del tempo totale del video, perche' il tempo totale cresce comunque
# con num_frames anche in assenza di fallback (piu' frame = sequenza piu'
# lunga = attention piu' costosa). Un vero swap su RAM di sistema via PCIe
# produce invece un salto di un ordine di grandezza nel tempo per step, non
# una crescita graduale.
STEP_TIMEOUT_MULT = 6.0   # step_dt > 6x la baseline del run -> sospetto fallback
STEP_TIMEOUT_ABS_S = 2.0  # ...ma solo se e' anche piu' lento di 2s in assoluto
                           # (evita falsi positivi quando i tempi sono gia' minuscoli)


class RamFallbackDetected(Exception):
    """Sollevata dal callback di step quando il tempo per step suggerisce che
    l'allocazione e' stata dirottata su RAM di sistema invece di fallire con
    un vero OOM (vedi commento sopra su STEP_TIMEOUT_MULT)."""


def make_step_watchdog():
    # Stato per-run: il primo step (index 0) include overhead one-off (selezione
    # algoritmo cuDNN, compilazione kernel) quindi non e' una baseline valida;
    # si usa il secondo step (index 1) come riferimento per quel run.
    state: dict[str, float | None] = {"t_prev": None, "baseline": None}

    def watchdog(pipe, step_index, timestep, callback_kwargs):
        now = time.time()
        if state["t_prev"] is not None:
            step_dt = now - state["t_prev"]
            if step_index == 1:
                state["baseline"] = step_dt
            elif state["baseline"] is not None:
                if step_dt > STEP_TIMEOUT_ABS_S and step_dt > STEP_TIMEOUT_MULT * state["baseline"]:
                    raise RamFallbackDetected(
                        "step %d: %.2fs vs baseline %.2fs" % (step_index, step_dt, state["baseline"])
                    )
        state["t_prev"] = now
        return callback_kwargs

    return watchdog

# Su Windows + GeForce il driver NVIDIA di default sostituisce un cudaMalloc
# fallito con un fallback silenzioso su RAM di sistema (System Memory
# Fallback Policy, dal driver 536.40): la VRAM risulta piena ma non si ottiene
# mai un vero torch.cuda.OutOfMemoryError, solo un rallentamento enorme (swap
# via PCIe). Il tetto sotto NON basta a evitarlo: agisce solo sull'allocatore
# interno di PyTorch, non sul driver, quindi non impedisce il fallback (verificato
# su questa macchina: il run e' comunque andato in RAM di sistema). Il bypass
# reale va fatto FUORI dal codice, una volta sola: NVIDIA Control Panel (o, se
# mancante perche' ritirato dal bundle driver, installarlo dal Microsoft Store)
# -> Gestisci impostazioni 3D -> Impostazioni Programma -> aggiungi
# .venv\Scripts\python.exe di questo ambiente -> "CUDA - Sysmem Fallback
# Policy" = "Prefer No Sysmem Fallback". Il tetto qui sotto resta solo come
# margine di sicurezza (lascia VRAM al contesto CUDA e ad altri processi),
# non come meccanismo anti-fallback.
#torch.cuda.set_per_process_memory_fraction(0.90, device=0)

# Model card THUDM: CogVideoX-2b raccomanda fp16 (a differenza del 5b, in bf16).
pipe = CogVideoXPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

prompt = (
    "POV shot from a person walking along a city sidewalk on a sunny afternoon, "
    "steady forward motion, parked cars lining the left side of the street, "
    "shop fronts and awnings on the right, pedestrians ahead in the distance, "
    "sharp focus, natural daylight, realistic urban environment, handheld camera feel,"
    "mass presence of trees, poles and suspended branches,"
)

negative_prompt = None

os.makedirs("outputs", exist_ok=True)

results = []

# CSV scritto con flush dopo ogni riga: se una run va in crash reale (non
# intercettato dal try/except sotto) o il processo viene interrotto a mano, i
# punti gia' misurati restano su disco invece di andare persi.
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["num_frames", "duration_s", "elapsed_s", "peak_vram_gb", "oom", "fail_reason"])
    f.flush()

    for num_frames in FRAME_POINTS:
        duration_s = num_frames / FPS
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        oom = False
        fail_reason = ""
        elapsed = None
        t0 = time.time()
        try:
            frames = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=HEIGHT,
                width=WIDTH,
                num_frames=num_frames,
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE,
                callback_on_step_end=make_step_watchdog(),
            ).frames[0]
            elapsed = time.time() - t0
            del frames
        except torch.cuda.OutOfMemoryError:
            oom = True
            fail_reason = "cuda_oom"
        except RamFallbackDetected:
            oom = True
            fail_reason = "ram_fallback_watchdog"
        except RuntimeError as e:
            # in alcuni path di offload l'OOM risale come RuntimeError generico
            # invece del subtype dedicato
            if "out of memory" in str(e).lower():
                oom = True
                fail_reason = "cuda_oom"
            else:
                raise

        peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
        torch.cuda.empty_cache()

        elapsed_str = "%.1f" % elapsed if elapsed is not None else ""
        print(
            "frames=%d durata=%.2fs elapsed=%s picco_vram=%.2fGB oom=%s fail_reason=%s"
            % (num_frames, duration_s, elapsed_str or "OOM", peak_vram_gb, oom, fail_reason or "-")
        )

        writer.writerow([num_frames, "%.3f" % duration_s, elapsed_str, "%.2f" % peak_vram_gb, oom, fail_reason])
        f.flush()

        results.append(
            {
                "num_frames": num_frames,
                "duration_s": duration_s,
                "elapsed_s": elapsed,
                "peak_vram_gb": peak_vram_gb,
                "oom": oom,
                "fail_reason": fail_reason,
            }
        )

first_oom = next((r for r in results if r["oom"]), None)
first_over_budget = next((r for r in results if not r["oom"] and r["elapsed_s"] > BUDGET_S), None)

viable = [r for r in results if not r["oom"] and r["elapsed_s"] <= BUDGET_S]
recommended = min(viable, key=lambda r: abs(r["duration_s"] - TARGET_S)) if viable else None

print("\n--- sintesi ---")
print("primo OOM:", ("frames=%d" % first_oom["num_frames"]) if first_oom else "nessuno")
print(
    "primo oltre %ds:" % BUDGET_S,
    ("frames=%d (%.1fs)" % (first_over_budget["num_frames"], first_over_budget["elapsed_s"]))
    if first_over_budget
    else "nessuno",
)
if recommended:
    print(
        "raccomandato: frames=%d durata=%.2fs tempo=%.1fs picco_vram=%.2fGB"
        % (
            recommended["num_frames"],
            recommended["duration_s"],
            recommended["elapsed_s"],
            recommended["peak_vram_gb"],
        )
    )
else:
    print("raccomandato: nessun punto soddisfa entrambi i vincoli (OOM o oltre budget su tutta la griglia)")
