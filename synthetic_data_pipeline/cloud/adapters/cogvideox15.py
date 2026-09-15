# cogvideox15.py — CogVideoX1.5-5B
#
# L'ultima versione della famiglia: CogVideoX1.5 e' dell'8 novembre 2024, e non
# ho trovato un CogVideoX2 cercando su zai-org/CogVideo il 15/09/2026
# (l'organizzazione THUDM e' stata rinominata zai-org, il vecchio nome rimanda).
#
# In locale la pipeline usata era CogVideoX-2b a 480x720 e 49 frame, in fp16.
# Qui si sale alla 1.5-5B, che e' la taglia buona: 1360x768, 81 frame, bf16.
# Sono due modelli diversi, non lo stesso a risoluzione maggiore.
#
# UNICO SENZA NEGATIVE PROMPT. La model card non ne pubblica uno, e inventarne
# uno sarebbe dargli un aiuto che gli altri non hanno: nel bake-off il negative
# e' una variabile libera per modello, e il manifest registra che qui e' assente.
import gc

import numpy as np
import torch
from diffusers import CogVideoXPipeline

MODEL_ID = "zai-org/CogVideoX1.5-5B"

SPEC = {
    "model_id": MODEL_ID,
    # Risoluzione raccomandata dalla model card.
    "height": 768,
    "width": 1360,
    # Il vincolo del modello e' num_frames = 16N + 1 con N <= 10. 81 e' il
    # default della model card (N=5, cioe' 5 s a 16 fps di generazione).
    "num_frames": 81,
    "steps": 50,
    "guidance": 6.0,
    # 8 e non 16: e' l'fps che la model card passa a export_to_video. Cambia solo
    # la velocita' di riproduzione dell'mp4, non i pixel che vedono SAM e DA3.
    "fps": 8,
    "dtype": "bfloat16",
    "negative": None,
}


def load(prompts):
    """Text encoder residente: vale la stessa ragione di wan22_a14b.

    La model card raccomanda enable_sequential_cpu_offload per chi ha poca
    memoria, ma e' molto piu' lento perche' sposta un modulo alla volta a ogni
    step. Su una scheda da 80 GB si usa enable_model_cpu_offload, che sposta una
    volta per modulo: il bake-off misura anche i secondi per clip, e rallentare
    un modello con una scelta di memoria che non serve falserebbe M7.
    """
    pipe = CogVideoXPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    return {"pipe": pipe}


def generate(handle, prompt, seed):
    out = handle["pipe"](
        prompt=prompt,
        height=SPEC["height"],
        width=SPEC["width"],
        num_frames=SPEC["num_frames"],
        num_inference_steps=SPEC["steps"],
        guidance_scale=SPEC["guidance"],
        num_videos_per_prompt=1,
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
