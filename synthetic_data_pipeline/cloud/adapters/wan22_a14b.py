# wan22_a14b.py — Wan2.2-T2V-A14B
#
# 27B parametri in mixture-of-experts, 14B attivi. Il README Wan2.2 dichiara
# "at least 80GB VRAM" per l'implementazione di riferimento; con diffusers e
# enable_model_cpu_offload la VRAM scende molto, ma serve RAM host abbondante.
# E' uno dei due modelli che impongono la scheda da 80 GB al bake-off.
#
# VIVE IN UN VENV A PARTE (envs/requirements-a14b.txt). La model card dichiara:
# "requires features that are currently available only in the main branch of
# diffusers". La 0.40.0 degli altri generatori non basta.
#
# DUE DENOISER, DUE GUIDANCE. Questo modello ha transformer (rumore alto) e
# transformer_2 (rumore basso), e il passaggio fra i due avviene a
# boundary_ratio * num_train_timesteps. boundary_ratio e transformer_2 arrivano
# dalla configurazione del repo con from_pretrained, quindi non si passano;
# guidance_scale e guidance_scale_2 SI', e sono diversi (4,0 e 3,0). Passare un
# solo guidance vorrebbe dire far girare la fase a rumore basso con il valore
# sbagliato. Firma verificata il 15/09/2026 sulla documentazione diffusers di
# WanPipeline.__call__.
import gc

import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanPipeline

MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

SPEC = {
    "model_id": MODEL_ID,
    # 720P nativo orizzontale, come dichiarato dal README Wan2.2.
    "height": 720,
    "width": 1280,
    "num_frames": 81,
    # 40 e non 50: e' il valore dell'esempio della model card per questo modello.
    "steps": 40,
    "guidance": 4.0,
    "guidance_2": 3.0,
    "fps": 16,
    "dtype": "bfloat16 (VAE fp32)",
    # Stesso negative prompt ufficiale della famiglia Wan.
    "negative": (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
        "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
        "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
        "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    ),
}


def load(prompts):
    """Il text encoder resta residente, a differenza di wan22_5b.

    Li' il precompute con rilascio serviva a far entrare il modello in 16 GB.
    Qui la scheda e' da 80 GB e enable_model_cpu_offload parcheggia comunque su
    CPU quello che non serve: aggiungere il precompute significherebbe
    complicare l'adapter piu' rischioso del lotto per una memoria che non manca.
    Gli embedding sono gli stessi, cambia solo dove stanno: l'uscita non cambia.
    """
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    # Se il repo non portasse il secondo denoiser, le due guidance sarebbero una
    # bugia silenziosa: guidance_scale_2 verrebbe ignorata e il manifest direbbe
    # il contrario. Meglio saperlo al caricamento.
    if getattr(pipe, "transformer_2", None) is None:
        print("ATTENZIONE: transformer_2 assente, guidance_2 verra' ignorata dalla pipeline.")
    print("boundary_ratio della pipeline: %s" % getattr(pipe, "boundary_ratio", None))
    return {"pipe": pipe}


def generate(handle, prompt, seed):
    out = handle["pipe"](
        prompt=prompt,
        negative_prompt=SPEC["negative"],
        height=SPEC["height"],
        width=SPEC["width"],
        num_frames=SPEC["num_frames"],
        num_inference_steps=SPEC["steps"],
        guidance_scale=SPEC["guidance"],
        guidance_scale_2=SPEC["guidance_2"],
        # ESPLICITO, non per pignoleria: in diffusers 0.40.0 WanPipeline ha
        # output_type="np" di default ma CogVideoXPipeline ha "pil". Lasciare
        # il default darebbe frame uint8 in 0..255 dove il contratto vuole
        # float in 0..1, e save_frames li scriverebbe tutti bianchi senza
        # sollevare niente.
        output_type="np",
        generator=torch.Generator(device="cuda").manual_seed(seed),
    )
    return np.asarray(out.frames[0])


def on_oom(handle):
    handle["pipe"].maybe_free_model_hooks()


def unload(handle):
    handle["pipe"] = None
    gc.collect()
    torch.cuda.empty_cache()
