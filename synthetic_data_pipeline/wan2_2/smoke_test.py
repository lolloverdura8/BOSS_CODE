# smoke_test.py
import time
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Risoluzione nativa del modello (720p, compressione VAE 16x16x4): generare
# fuori da 704x1280 porta il modello fuori distribuzione e degrada la resa.
HEIGHT, WIDTH = 704, 1280
# Durata = NUM_FRAMES / FPS: 121 frame = 5.04 s, il nativo del modello.
# Il picco VRAM e' dominato dai ~9.3 GB di pesi del transformer residenti, non
# dalle attivazioni: misurati 11.19 GB a 480x832 su 25 frame e 11.54 GB a
# 704x1280 su 49 (token 4.2x, memoria +0.35 GB). Se 121 dovesse dare OOM,
# scendere a 81 o 49.
NUM_FRAMES = 121
NUM_STEPS = 50           # default del modello; sotto i 40 il denoising resta incompleto
GUIDANCE = 5.0
FPS = 24                 # frame rate nativo: esportare piu' lento falsa il moto

# Il VAE resta in fp32 (in bf16 produce artefatti di decodifica); transformer e
# text encoder in bf16. I tre componenti insieme non stanno nei 16 GB della
# 5070 Ti, quindi cpu offload sequenziale a livello di modello.
vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

# A 704x1280 la decodifica VAE a piena risoluzione e' il punto in cui tipicamente
# arriva l'OOM: la si fa a tasselli.
pipe.vae.enable_tiling()

prompt = (
    "POV shot from a person walking along a city sidewalk on a sunny afternoon, "
    "steady forward motion, parked cars lining the left side of the street, "
    "shop fronts and awnings on the right, pedestrians ahead in the distance, "
    "sharp focus, natural daylight, realistic urban environment, handheld camera feel,"
    "mass presence of trees, poles and suspended branches,"
)

# Negative prompt ufficiale del modello (in cinese nella model card): scoraggia
# sovraesposizione, staticita', volti e arti deformi, sfondo affollato.
negative_prompt = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

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
