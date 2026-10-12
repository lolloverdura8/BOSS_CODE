# wan22_fun5b_ctrl.py — Wan2.2-Fun-5B-Control (Alibaba PAI), ramo Sim2Real
#
# Video-to-video controllato: un solo control_video per clip (depth nella
# configurazione A, edge nella B, vedi sim2real.CONFIGS), letto dal pacchetto di
# CARLA/prepare_controls.py gia' a 1280x704, 121 frame, 24 fps.
#
# Derivato da examples/wan2.2_fun/predict_v2v_control_5b.py di aigc-apps/VideoX-Fun
# (commit 4b7b6402, letto il 12/10/2026). Lo script e' scritto a costanti di modulo
# ed esegue tutto all'import: qui la stessa costruzione della pipeline e' divisa in
# load() e generate(), e gira DENTRO il processo del runner, nel venv fun. Cosi' i
# frame escono come tensore, senza passare per un mp4.
#
# TRE SCOSTAMENTI DALL'ESEMPIO, VOLUTI:
#   - sample_size [704, 1280]: e' [altezza, larghezza], e l'esempio [1280, 704] e'
#     verticale. Copiato tale e quale darebbe clip in verticale e GT sfasata;
#   - TeaCache spento (nell'esempio e' acceso): approssima i passi di denoising e
#     cambia la clip, e qui si misura la fedelta' ai controlli;
#   - GPU_memory_mode "model_full_load" invece di "sequential_cpu_offload": sull'A100
#     da 80 GB il modello intero ci sta, e l'offload per strati e' molto piu' lento.
#
# SETUP SUL POD: setup_pod.sh fun (clone di VideoX-Fun, venv fun, pesi in
# /workspace/models/Wan2.2-Fun-5B-Control).
import gc
import os
import sys

import numpy as np
import torch

import prompts
import sim2real

VIDEOX_DIR = os.environ.get("VIDEOX_FUN_DIR", "/workspace/VideoX-Fun")
MODEL_DIR = os.environ.get("WAN_FUN_MODEL_DIR", "/workspace/models/Wan2.2-Fun-5B-Control")
CONFIG_PATH = "config/wan2.2/wan_civitai_5b.yaml"

SPEC = {
    "model_id": "alibaba-pai/Wan2.2-Fun-5B-Control",
    "repo": "aigc-apps/VideoX-Fun",
    "commit": "4b7b6402a1e0f0406bd6801fb66c0a00bd922621",
    "config": CONFIG_PATH,
    "sample_size": [704, 1280],           # [altezza, larghezza]
    "height": 704,
    "width": 1280,
    "num_frames": 121,
    "fps": 24,
    # Valori dell'esempio ufficiale.
    "steps": 40,
    "guidance": 6.0,
    "sampler": "Flow",
    "shift": 5,
    "teacache": False,
    "cfg_skip_ratio": 0,
    "gpu_memory_mode": "model_full_load",
    "dtype": "bfloat16",
    "configs": sim2real.CONFIGS["wan22_fun5b_ctrl"],
    "negative": prompts.NEGATIVE_PROMPT,
    "pixel_source": "tensore della pipeline (senza compressione)",
}

# Pesi bf16 di transformer, text encoder UMT5 e VAE: circa 23 GB. Il margine copre
# le attivazioni a 121 frame in 704x1280; e' una soglia di prudenza, da rivedere col
# picco misurato sulla prima clip.
VRAM_FREE_MIN_GB = 40.0
HOST_RAM_MIN_GB = 32.0


def _preflight():
    import gpu
    avail = gpu.host_ram()[1]
    free, total = torch.cuda.mem_get_info()
    print(f"pre-volo wan22_fun5b_ctrl: VRAM libera {free / 1024**3:.2f} GB su"
          f" {total / 1024**3:.2f}, RAM host libera {avail:.2f} GB")
    if avail < HOST_RAM_MIN_GB:
        raise SystemExit(f"RAM host disponibile {avail:.2f} GB < {HOST_RAM_MIN_GB:.1f} GB")
    if free / 1024**3 < VRAM_FREE_MIN_GB:
        raise SystemExit(f"VRAM libera {free / 1024**3:.2f} GB < {VRAM_FREE_MIN_GB:.1f} GB")
    for path in (VIDEOX_DIR, MODEL_DIR):
        if not os.path.isdir(path):
            raise SystemExit(f"non trovato: {path} (setup_pod.sh fun)")


def _models(config, dtype):
    """Transformer, VAE, tokenizer e text encoder, come nello script ufficiale."""
    from omegaconf import OmegaConf
    from videox_fun.models import (
        AutoencoderKLWan,
        AutoencoderKLWan3_8,
        AutoTokenizer,
        Wan2_2Transformer3DModel,
        WanT5EncoderModel,
    )
    tkw, vkw, ekw = (config["transformer_additional_kwargs"], config["vae_kwargs"],
                     config["text_encoder_kwargs"])
    if tkw.get("transformer_combination_type", "single") != "single":
        raise SystemExit(f"config MoE inattesa per il 5B: {CONFIG_PATH}")
    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(MODEL_DIR, tkw.get("transformer_low_noise_model_subpath", "transformer")),
        transformer_additional_kwargs=OmegaConf.to_container(tkw),
        low_cpu_mem_usage=True, torch_dtype=dtype)
    vae_cls = {"AutoencoderKLWan": AutoencoderKLWan,
               "AutoencoderKLWan3_8": AutoencoderKLWan3_8}[vkw.get("vae_type", "AutoencoderKLWan")]
    vae = vae_cls.from_pretrained(os.path.join(MODEL_DIR, vkw.get("vae_subpath", "vae")),
                                  additional_kwargs=OmegaConf.to_container(vkw)).to(dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(MODEL_DIR, ekw.get("tokenizer_subpath", "tokenizer")))
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(MODEL_DIR, ekw.get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(ekw),
        low_cpu_mem_usage=True, torch_dtype=dtype).eval()
    return transformer, vae, tokenizer, text_encoder


def load(prompts_list):
    """Costruisce la pipeline una volta sola; i prompt si codificano a ogni clip."""
    _preflight()
    if VIDEOX_DIR not in sys.path:
        sys.path.insert(0, VIDEOX_DIR)
    from diffusers import FlowMatchEulerDiscreteScheduler
    from omegaconf import OmegaConf
    from videox_fun.dist import set_multi_gpus_devices
    from videox_fun.pipeline import Wan2_2FunControlPipeline
    from videox_fun.utils import apply_gpu_memory_mode, filter_kwargs

    dtype = torch.bfloat16
    device = set_multi_gpus_devices(1, 1)
    config = OmegaConf.load(os.path.join(VIDEOX_DIR, CONFIG_PATH))
    transformer, vae, tokenizer, text_encoder = _models(config, dtype)
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler,
                        OmegaConf.to_container(config["scheduler_kwargs"])))
    pipe = Wan2_2FunControlPipeline(transformer=transformer, transformer_2=None, vae=vae,
                                    tokenizer=tokenizer, text_encoder=text_encoder,
                                    scheduler=scheduler)
    apply_gpu_memory_mode(pipe, SPEC["gpu_memory_mode"], device, dtype)
    # TeaCache e cfg_skip non si attivano: nessuna chiamata a enable_teacache /
    # enable_cfg_skip, che nello script sono le uniche vie per accenderli.
    boundary = config["transformer_additional_kwargs"].get("boundary", 0.875)
    ratio = vae.config.temporal_compression_ratio
    frames = int((SPEC["num_frames"] - 1) // ratio * ratio) + 1
    if frames != SPEC["num_frames"]:
        raise SystemExit(f"121 frame non compatibili con la compressione temporale {ratio}")
    print(f"Wan2.2-Fun-5B-Control caricato: {SPEC['gpu_memory_mode']}, TeaCache spento")
    return {"pipe": pipe, "device": device, "boundary": boundary}


def pipe_kwargs(generator, inpaint_video, inpaint_mask, control_video, boundary):
    """Gli argomenti della pipeline oltre al prompt, separati per poterli provare
    senza GPU: altezza e larghezza scambiate darebbero una clip in verticale."""
    size = SPEC["sample_size"]
    return {
        "num_frames": SPEC["num_frames"], "negative_prompt": SPEC["negative"],
        "height": size[0], "width": size[1], "generator": generator,
        "guidance_scale": SPEC["guidance"], "num_inference_steps": SPEC["steps"],
        "video": inpaint_video, "mask_video": inpaint_mask, "control_video": control_video,
        "control_camera_video": None, "ref_image": None,
        "boundary": boundary, "shift": SPEC["shift"],
    }


def to_frames(sample):
    """(1, C, T, H, W) in 0..1 -> (T, H, W, C) float32, il contratto degli adapter."""
    return sample[0].permute(1, 2, 3, 0).float().cpu().numpy().astype(np.float32)


def generate(handle, prompt, seed, control):
    from videox_fun.utils import get_image_to_video_latent, get_video_to_video_latent
    size, n = SPEC["sample_size"], SPEC["num_frames"]
    # Nessuna immagine iniziale o finale: solo il controllo guida la clip.
    inpaint_video, inpaint_mask, _ = get_image_to_video_latent(None, None, video_length=n,
                                                               sample_size=size)
    control_video, _, _, _ = get_video_to_video_latent(
        sim2real.wan_control_video(control), video_length=n, sample_size=size,
        fps=SPEC["fps"], ref_image=None)
    generator = torch.Generator(device=handle["device"]).manual_seed(seed)
    with torch.no_grad():
        sample = handle["pipe"](prompt, **pipe_kwargs(generator, inpaint_video, inpaint_mask,
                                                      control_video, handle["boundary"])).videos
    return to_frames(sample)


def on_oom(handle):
    gc.collect()
    torch.cuda.empty_cache()


def unload(handle):
    handle["pipe"] = None
    gc.collect()
    torch.cuda.empty_cache()
