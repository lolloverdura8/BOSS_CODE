# smoke_test.py
import os
import time

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

MODEL_ID = "THUDM/CogVideoX-5b"

# Risoluzione nativa del modello (sample_height=60, sample_width=90,
# vae_scale_factor_spatial=8 -> 480x720): forzare la risoluzione nativa di
# Wan2.2 (704x1280) porterebbe il modello fuori distribuzione, stesso
# principio gia' rispettato in wan2_2/smoke_test.py.
HEIGHT, WIDTH = 480, 720
# Durata = NUM_FRAMES / FPS: 49 frame = 6.125 s, il nativo del modello
# (vincolo num_frames = 4k+1 per allinearsi alla compressione temporale del VAE).
NUM_FRAMES = 49
NUM_STEPS = 50    # default del modello
GUIDANCE = 6.0    # default nativo CogVideoX (Wan usa 5.0: modelli diversi, non si riusa il valore di Wan)
FPS = 8           # fps nativo

# A differenza di Wan, CogVideoX non ha un vincolo documentato di VAE in fp32:
# l'intera pipeline viene caricata in bf16. Se in fase di verifica emergono
# artefatti di decodifica, passare a VAE separato in fp32 come per Wan.
pipe = CogVideoXPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

# Stesso prompt positivo di wan2_2/smoke_test.py, per confronto diretto.
prompt = (
    "POV shot from a person walking along a city sidewalk on a sunny afternoon, "
    "steady forward motion, parked cars lining the left side of the street, "
    "shop fronts and awnings on the right, pedestrians ahead in the distance, "
    "sharp focus, natural daylight, realistic urban environment, handheld camera feel,"
    "mass presence of trees, poles and suspended branches,"
)

# CogVideoX non ha un negative prompt ufficiale (a differenza del cinese da
# model card di Wan): si lascia None.
negative_prompt = None

os.makedirs("outputs", exist_ok=True)

t0 = time.time()
frames = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=HEIGHT,
    width=WIDTH,
    num_frames=NUM_FRAMES,
    num_inference_steps=NUM_STEPS,
    guidance_scale=GUIDANCE,
).frames[0]
elapsed = time.time() - t0

export_to_video(frames, "outputs/smoke_test.mp4", fps=FPS)

print("frames:", len(frames), frames[0].shape, "durata: %.2f s" % (len(frames) / FPS))
print("tempo generazione: %.1f s" % elapsed)
print("picco VRAM: %.2f GB" % (torch.cuda.max_memory_allocated() / 1024**3))
