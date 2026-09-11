# generate_test_clips.py
#
# Passo 0.2 esteso del PIANO_ATTACCO_OR4.1: 5 clip per classe con Wan2.2-TI2V-5B,
# da passare poi a SAM 3 (sam3_1/eval_on_generated.py) e DA3
# (depth_anything3/eval_on_generated.py). Parametri nativi e ottimizzazioni di
# memoria sono quelli di smoke_test.py, che resta la referenza commentata: qui si
# ripetono solo le ragioni delle cose che cambiano.
#
# Uscite in outputs/test_clips/:
#   {clip_id}.mp4                 il video, per guardarlo
#   frames/{clip_id}/NNN.png      un frame ogni FRAME_STRIDE, PNG lossless presi
#                                 dall'array della pipeline e NON estratti dall'mp4:
#                                 gli annotatori devono vedere i pixel generati, non
#                                 quelli ricompressi in H.264
#   manifest.jsonl                una riga per clip generata, con esito e memoria
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import psutil
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video
from PIL import Image

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

HEIGHT, WIDTH = 704, 1280
# 73 frame (3.04 s): il punto piu' lungo che la 5070 Ti regge, vedi smoke_test.py.
NUM_FRAMES = 73
NUM_STEPS = 50
GUIDANCE = 5.0
FPS = 24
VRAM_FRACTION = 0.90

# Un frame ogni 4: e' il coverage_frame_gap del reference set CARLA (B.1). Frame
# consecutivi sono quasi identici e per un detector valgono uno solo. 0..72 passo 4
# = 19 frame per clip.
FRAME_STRIDE = 4

# --- controlli pre-volo -------------------------------------------------------
# VRAM libera sulla scheda, in GiB come tutte le misure di questa cartella. Nello
# sweep il punto a 73 frame e' riuscito con 14.02 GiB di picco prenotato e 15.72
# GiB occupati sulla scheda (vram_sweep_20260828_135314.csv): il resto, 1.70 GiB,
# era desktop + contesto CUDA. Su 15.92 GiB totali restavano quindi 14.22 GiB per
# PyTorch, ed e' quello il minimo dimostrato. Sotto, il run rischia un OOM a meta'
# clip dopo minuti di denoising: meglio fermarsi prima di caricare.
VRAM_FREE_MIN_GB = 14.2
# Stessa soglia di vram_benchmark.py: con enable_model_cpu_offload transformer e
# VAE restano parcheggiati in RAM host per tutto il run.
HOST_RAM_MIN_GB = 16.0
# Dopo il rilascio del text encoder sulla GPU non deve restare quasi nulla, perche'
# il resto della pipeline e' ancora su CPU e gli embedding sono parcheggiati su CPU.
# Sopra questa soglia il rilascio non ha funzionato: il caso tipico e' encode_prompt
# chiamato fuori da no_grad, che trattiene i pesi nel grafo di autograd.
RELEASE_MAX_GB = 0.5

OUT_DIR = os.path.join("outputs", "test_clips")

# 5 prompt per classe, una scena per prompt. Vincoli comuni: POV da pedone su
# marciapiede, UN solo oggetto bersaglio davanti a pochi metri, ben visibile. Per i
# sospesi il SUPPORTO deve stare in campo (tronco, palo), perche' e' il riferimento
# del controllo di coerenza di DA3, e l'oggetto si sviluppa di traverso al percorso
# e non verso la camera, cosi' oggetto e supporto stanno alla stessa profondita'.
PROMPTS = {
    "monopattino": [
        "POV shot from a person walking on a city sidewalk, steady forward motion, an electric "
        "kick scooter parked upright on its kickstand in the middle of the sidewalk about three "
        "meters ahead, clearly visible, sunny afternoon, sharp focus, realistic urban environment",
        "POV shot from a person walking on a city sidewalk, steady forward motion, an electric "
        "kick scooter lying on its side across the sidewalk about four meters ahead, overcast "
        "daylight, sharp focus, realistic urban environment",
        "POV shot from a person walking on a city sidewalk, steady forward motion, a rental "
        "electric kick scooter parked against a building wall on the right about three meters "
        "ahead, soft early morning light, sharp focus, realistic urban environment",
        "POV shot from a person walking on a city sidewalk, steady forward motion, a person riding "
        "an electric kick scooter towards the camera on the sidewalk about five meters ahead, "
        "daylight, sharp focus, realistic urban environment",
        "POV shot from a person walking on a city sidewalk, steady forward motion, an electric "
        "kick scooter parked at the edge of the sidewalk next to a pedestrian crossing about three "
        "meters ahead, wet pavement after rain, cloudy sky, sharp focus, realistic urban environment",
    ],
    "ramo_sporgente": [
        "POV shot from a person walking on a tree-lined sidewalk, steady forward motion, a low "
        "tree branch sticking out sideways from a tree trunk across the sidewalk at head height "
        "about three meters ahead, the trunk clearly visible on the right, sunny day, sharp focus, "
        "realistic urban environment",
        "POV shot from a person walking on a park path, steady forward motion, a thick tree branch "
        "growing sideways from a trunk on the left and crossing the path at head height about four "
        "meters ahead, overcast daylight, sharp focus, realistic environment",
        "POV shot from a person walking on a residential sidewalk, steady forward motion, a leafy "
        "branch extending sideways from a tree trunk next to a garden wall and crossing the "
        "sidewalk at head height about three meters ahead, late afternoon light, sharp focus, "
        "realistic urban environment",
        "POV shot from a person walking on a city sidewalk in winter, steady forward motion, a "
        "bare leafless branch extending sideways from a tree trunk on the right across the "
        "sidewalk at head height about three meters ahead, cloudy sky, sharp focus, realistic "
        "urban environment",
        "POV shot from a person walking on a shaded sidewalk, steady forward motion, a low "
        "branch with green leaves extending sideways from a tree trunk on the left across the "
        "sidewalk at head height about four meters ahead, dappled sunlight, sharp focus, "
        "realistic urban environment",
    ],
    "insegna_cartello_basso": [
        "POV shot from a person walking on a city sidewalk, steady forward motion, a rectangular "
        "shop sign hanging from a metal pole at the edge of the sidewalk, the sign panel at head "
        "height about three meters ahead, sunny afternoon, sharp focus, realistic urban environment",
        "POV shot from a person walking on a city sidewalk, steady forward motion, a traffic sign "
        "mounted low on a metal pole on the sidewalk, its panel at head height about three meters "
        "ahead, overcast daylight, sharp focus, realistic urban environment",
        "POV shot from a person walking on a narrow old town sidewalk, steady forward motion, a "
        "small cafe sign hanging from a short pole bracket at head height about three meters "
        "ahead, soft morning light, sharp focus, realistic urban environment",
        "POV shot from a person walking on a city sidewalk, steady forward motion, a temporary "
        "road works sign fixed to a metal pole at head height on the sidewalk about four meters "
        "ahead, daylight, sharp focus, realistic urban environment",
        "POV shot from a person walking on a city sidewalk, steady forward motion, a street name "
        "sign mounted low on a pole at head height about three meters ahead, warm evening light, "
        "sharp focus, realistic urban environment",
    ],
    "ostacolo_sospeso_generico": [
        "POV shot from a person walking on a city sidewalk, steady forward motion, a horizontal "
        "metal bar fixed between two poles across the sidewalk at head height about three meters "
        "ahead, sunny day, sharp focus, realistic urban environment",
        "POV shot from a person walking on a city sidewalk, steady forward motion, a horizontal "
        "metal pipe supported by two vertical posts crossing the sidewalk at head height about "
        "four meters ahead, overcast daylight, sharp focus, realistic urban environment",
        "POV shot from a person walking on a city sidewalk, steady forward motion, the low "
        "horizontal metal beam of an awning frame supported by two poles crossing the sidewalk at "
        "head height about three meters ahead, afternoon light, sharp focus, realistic urban "
        "environment",
        "POV shot from a person walking on a pedestrian passage, steady forward motion, a "
        "horizontal wooden beam resting on two wooden posts across the passage at head height "
        "about three meters ahead, soft daylight, sharp focus, realistic environment",
        "POV shot from a person walking on a city sidewalk, steady forward motion, a horizontal "
        "barrier bar mounted between two poles across the sidewalk at head height about four "
        "meters ahead, cloudy sky, sharp focus, realistic urban environment",
    ],
}

# Negative prompt ufficiale del modello, identico a smoke_test.py.
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

PROC = psutil.Process()


def mem_snapshot():
    """Le cinque metriche di vram_benchmark.py, stessi nomi e stesse unita' (GiB)."""
    free, total = torch.cuda.mem_get_info()
    return {
        "vram_alloc_peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "vram_reserved_peak_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "vram_device_used_gb": round((total - free) / 1024**3, 3),
        "host_rss_gb": round(PROC.memory_info().rss / 1024**3, 3),
        "host_avail_gb": round(psutil.virtual_memory().available / 1024**3, 3),
    }


def preflight():
    avail_gb = psutil.virtual_memory().available / 1024**3
    if avail_gb < HOST_RAM_MIN_GB:
        sys.exit("RAM host disponibile %.2f GB < %.1f GB: chiudere applicazioni prima di"
                 " lanciare." % (avail_gb, HOST_RAM_MIN_GB))
    # mem_get_info inizializza il contesto CUDA, quindi il libero letto qui e' gia'
    # al netto del contesto: e' lo spazio che l'allocatore avra' davvero.
    free, total = torch.cuda.mem_get_info()
    free_gb, used_gb = free / 1024**3, (total - free) / 1024**3
    print("pre-volo: VRAM libera %.2f GB, gia' occupata %.2f GB su %.2f; RAM host libera"
          " %.2f GB" % (free_gb, used_gb, total / 1024**3, avail_gb))
    if free_gb < VRAM_FREE_MIN_GB:
        sys.exit("VRAM libera %.2f GB < %.1f GB richiesti a %d frame: sulla scheda sono gia'"
                 " occupati %.2f GB. Chiudere browser e applicazioni con accelerazione GPU"
                 " (nvidia-smi mostra chi occupa memoria) e rilanciare."
                 % (free_gb, VRAM_FREE_MIN_GB, NUM_FRAMES, used_gb))


def build_jobs():
    jobs = []
    for cls, prompts in PROMPTS.items():
        for n, prompt in enumerate(prompts):
            # seed = indice progressivo: riproducibile e diverso per ogni clip
            jobs.append({"clip_id": "%s_%02d" % (cls, n), "classe": cls,
                         "prompt": prompt, "seed": len(jobs)})
    return jobs


def encode_all(pipe, prompts):
    """Embedding di tutti i prompt, poi il text encoder esce dalla VRAM e dal processo.

    Il negativo si codifica una volta sola invece che una per clip: con
    do_classifier_free_guidance=False encode_prompt restituisce solo l'embedding
    del testo passato, e il negativo e' un testo come un altro.
    """
    pipe.text_encoder.to("cuda")
    embeds = {}
    with torch.no_grad():
        for text in list(prompts) + [NEGATIVE_PROMPT]:
            e, _ = pipe.encode_prompt(prompt=text, do_classifier_free_guidance=False,
                                      num_videos_per_prompt=1,
                                      device=torch.device("cuda"), dtype=torch.bfloat16)
            # parcheggiati su CPU: sulla GPU salgono solo quelli della clip in corso
            embeds[text] = e.to("cpu")
    pipe.text_encoder = None
    gc.collect()
    torch.cuda.empty_cache()

    left_gb = torch.cuda.memory_allocated() / 1024**3
    if left_gb >= RELEASE_MAX_GB:
        sys.exit("dopo il rilascio del text encoder restano %.2f GB allocati (soglia %.1f):"
                 " il rilascio non ha funzionato." % (left_gb, RELEASE_MAX_GB))
    print("text encoder rilasciato: %.3f GB ancora allocati, %d embedding su CPU"
          % (left_gb, len(embeds)))
    return embeds


def save_frames(frames, frame_dir):
    os.makedirs(frame_dir, exist_ok=True)
    for k in range(0, len(frames), FRAME_STRIDE):
        img = (np.clip(frames[k], 0.0, 1.0) * 255.0).round().astype(np.uint8)
        Image.fromarray(img).save(os.path.join(frame_dir, "%03d.png" % k))


def main():
    parser = argparse.ArgumentParser(
        description="Genera le clip di test del passo 0.2 esteso con Wan2.2-TI2V-5B.")
    parser.add_argument("--limit", type=int, default=None,
                        help="solo le prime N clip della lista, per una prova")
    args = parser.parse_args()

    jobs = build_jobs()
    if args.limit:
        jobs = jobs[:args.limit]
    todo = [j for j in jobs if not os.path.exists(os.path.join(OUT_DIR, j["clip_id"] + ".mp4"))]
    print("clip richieste: %d, gia' presenti: %d, da generare: %d"
          % (len(jobs), len(jobs) - len(todo), len(todo)))
    if not todo:
        return

    preflight()
    torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION, device=0)

    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)

    embeds = encode_all(pipe, [j["prompt"] for j in todo])

    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, "manifest.jsonl")
    vram_budget_gb = torch.cuda.mem_get_info()[1] / 1024**3 * VRAM_FRACTION
    neg = embeds[NEGATIVE_PROMPT]
    esiti = {"ok": 0, "oom": 0, "sospetto_fallback": 0}
    peak_max = 0.0

    for n, job in enumerate(todo):
        print("\n[%d/%d] %s (seed %d)" % (n + 1, len(todo), job["clip_id"], job["seed"]),
              flush=True)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        frames, esito = None, "ok"
        try:
            frames = pipe(
                prompt=None,
                negative_prompt=None,
                prompt_embeds=embeds[job["prompt"]].to("cuda"),
                negative_prompt_embeds=neg.to("cuda"),
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE,
                generator=torch.Generator(device="cuda").manual_seed(job["seed"]),
            ).frames[0]
        except torch.cuda.OutOfMemoryError:
            esito = "oom"
            # un'eccezione a meta' pipe() salta maybe_free_model_hooks: senza questa
            # chiamata il modulo che era in GPU ci resta anche per la clip successiva
            pipe.maybe_free_model_hooks()
        elapsed = time.time() - t0
        snap = mem_snapshot()

        if esito == "ok" and snap["vram_alloc_peak_gb"] >= vram_budget_gb * 0.98:
            esito = "sospetto_fallback"
        if frames is not None:
            save_frames(frames, os.path.join(OUT_DIR, "frames", job["clip_id"]))
            # prima un file temporaneo, poi il nome definitivo: l'mp4 e' il marcatore di
            # clip completata, e un file troncato da un'interruzione non deve sembrarlo
            final = os.path.join(OUT_DIR, job["clip_id"] + ".mp4")
            tmp = os.path.join(OUT_DIR, job["clip_id"] + ".tmp.mp4")
            export_to_video(frames, tmp, fps=FPS)
            os.replace(tmp, final)

        record = dict(job, esito=esito, tempo_s=round(elapsed, 1), num_frames=NUM_FRAMES,
                      height=HEIGHT, width=WIDTH, steps=NUM_STEPS, guidance=GUIDANCE,
                      fps=FPS, frame_stride=FRAME_STRIDE, **snap)
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        esiti[esito] += 1
        peak_max = max(peak_max, snap["vram_alloc_peak_gb"])
        print("  esito %s, %.1f s, picco allocato %.2f GB, prenotato %.2f GB, occupata sulla"
              " scheda %.2f GB, RAM host libera %.2f GB"
              % (esito, elapsed, snap["vram_alloc_peak_gb"], snap["vram_reserved_peak_gb"],
                 snap["vram_device_used_gb"], snap["host_avail_gb"]), flush=True)

        del frames
        gc.collect()
        torch.cuda.empty_cache()

    print("\n--- sintesi ---")
    print("clip generate:     ok %d, oom %d, sospetto fallback %d"
          % (esiti["ok"], esiti["oom"], esiti["sospetto_fallback"]))
    print("picco allocato massimo fra le clip: %.2f GB su un budget di %.2f"
          % (peak_max, vram_budget_gb))
    print("manifest:          %s" % manifest_path)


if __name__ == "__main__":
    main()
