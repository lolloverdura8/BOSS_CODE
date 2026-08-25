# vram_benchmark.py
import csv
import time

import torch
from diffusers import AutoencoderKLWan, WanPipeline

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Stessa risoluzione nativa e stessa configurazione VRAM-saving di smoke_test.py.
HEIGHT, WIDTH = 704, 1280
NUM_STEPS = 50   # fisso, non sweepato: sotto i 40 il denoising resta incompleto
GUIDANCE = 5.0
FPS = 24

# Punti dello sweep: durata = num_frames / FPS, da ~1s al nativo (121 = 5.04s).
FRAME_POINTS = [25, 49, 73, 97, 121]

BUDGET_S = 300   # obiettivo: 5 minuti per video
TARGET_S = 2.0   # obiettivo: scenari da ~2s

OUTPUT_CSV = "outputs/vram_sweep.csv"

# Su Windows + GeForce il driver NVIDIA di default sostituisce un cudaMalloc
# fallito con un fallback silenzioso su RAM di sistema (System Memory
# Fallback Policy): la VRAM risulta piena ma non si ottiene mai un vero
# torch.cuda.OutOfMemoryError, solo un rallentamento enorme (swap via PCIe).
# Imponendo un tetto esplicito all'allocatore di PyTorch, l'allocazione fallisce
# PRIMA del cudaMalloc che innescherebbe il fallback, quindi l'OOM diventa
# reale e intercettabile dal try/except sotto. 0.90 lascia margine per il
# contesto CUDA e altri processi che gia' occupano VRAM a riposo.
torch.cuda.set_per_process_memory_fraction(0.90, device=0)

# Il VAE resta in fp32 (in bf16 produce artefatti di decodifica); transformer e
# text encoder in bf16. Model cpu offload perche' i tre componenti insieme non
# stanno nei 16 GB della 5070 Ti; vae tiling per la decodifica a 704x1280.
vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

prompt = (
    "POV shot from a person walking along a city sidewalk on a sunny afternoon, "
    "steady forward motion, parked cars lining the left side of the street, "
    "shop fronts and awnings on the right, pedestrians ahead in the distance, "
    "sharp focus, natural daylight, realistic urban environment, handheld camera feel,"
    "mass presence of trees, poles and suspended branches,"
)

negative_prompt = (  # official Chinese negative prompt from the model card
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

results = []

# CSV scritto con flush dopo ogni riga: se una run va in crash reale (non
# intercettato dal try/except sotto) o il processo viene interrotto a mano, i
# punti gia' misurati restano su disco invece di andare persi.
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["num_frames", "duration_s", "elapsed_s", "peak_vram_gb", "oom"])
    f.flush()

    for num_frames in FRAME_POINTS:
        duration_s = num_frames / FPS
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        oom = False
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
            ).frames[0]
            elapsed = time.time() - t0
            del frames
        except torch.cuda.OutOfMemoryError:
            oom = True
        except RuntimeError as e:
            # in alcuni path di offload l'OOM risale come RuntimeError generico
            # invece del subtype dedicato
            if "out of memory" in str(e).lower():
                oom = True
            else:
                raise

        peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
        torch.cuda.empty_cache()

        elapsed_str = "%.1f" % elapsed if elapsed is not None else ""
        print(
            "frames=%d durata=%.2fs elapsed=%s picco_vram=%.2fGB oom=%s"
            % (num_frames, duration_s, elapsed_str or "OOM", peak_vram_gb, oom)
        )

        writer.writerow([num_frames, "%.3f" % duration_s, elapsed_str, "%.2f" % peak_vram_gb, oom])
        f.flush()

        results.append(
            {
                "num_frames": num_frames,
                "duration_s": duration_s,
                "elapsed_s": elapsed,
                "peak_vram_gb": peak_vram_gb,
                "oom": oom,
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
