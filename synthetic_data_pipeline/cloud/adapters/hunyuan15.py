# hunyuan15.py — HunyuanVideo-1.5 (Tencent)
#
# Non e' nel §2.4 di OR4.1, che nomina Wan2.2, CogVideoX e Open-Sora: e' uscito
# dopo (20 novembre 2025). Sta nel bake-off perche' con 8,3B parametri dichiara
# 14 GB con offload, cioe' e' il candidato che potrebbe girare anche in locale.
#
# L'ID DEL REPO NON E' QUELLO SCRITTO NELLA DOCUMENTAZIONE DIFFUSERS. Il docstring
# di HunyuanVideo15Pipeline.__call__ cita
# "hunyuanvideo-community/HunyuanVideo-1.5-480p_t2v", che non esiste. Elencando
# l'organizzazione via API HuggingFace il 15/09/2026, i repo veri hanno
# "Diffusers-" in mezzo: HunyuanVideo-1.5-Diffusers-720p_t2v. Senza questa
# correzione il download fallirebbe sul pod, a GPU accesa.
#
# NON PRENDE guidance_scale A RUNTIME. La guidance sta in un oggetto pipe.guider
# (ClassifierFreeGuidance, scale 6.0 di default) e passarla come argomento
# solleverebbe. Si cambia con pipe.guider = pipe.guider.new(guidance_scale=...).
import gc

import numpy as np
import torch
from diffusers import HunyuanVideo15Pipeline

# Variante 720p e non 480p, e non distillata: il bake-off chiede a ogni modello
# il suo meglio, e le distillate barattano qualita' per velocita'.
MODEL_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v"

SPEC = {
    "model_id": MODEL_ID,
    "height": 720,
    "width": 1280,
    # Default del modello: 121 frame, circa 5 s a 24 fps.
    "num_frames": 121,
    "steps": 50,
    # Non e' un argomento di __call__: e' la configurazione del guider. Sta in
    # SPEC perche' il manifest deve registrare con che guidance si e' generato.
    "guidance": 6.0,
    "fps": 24,
    "dtype": "bfloat16",
    # La model card non pubblica un negative prompt per il t2v.
    "negative": None,
}


def load(prompts):
    pipe = HunyuanVideo15Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    # Se il default del guider cambiasse fra una versione e l'altra di diffusers,
    # SPEC["guidance"] direbbe una cosa e il modello ne farebbe un'altra. Meglio
    # allinearlo esplicitamente e stamparlo.
    try:
        pipe.guider = pipe.guider.new(guidance_scale=SPEC["guidance"])
        print("guider: %s" % pipe.guider)
    except Exception as e:
        print("ATTENZIONE: guider non riconfigurato (%s); SPEC['guidance'] potrebbe non"
              " corrispondere a quella usata davvero." % e)

    # BACKEND DI ATTENZIONE: serve, non e' un'ottimizzazione. A 121 frame in 720p
    # col backend di default il picco misurato e' 65,94 GB e va in OOM anche su
    # un'A100 da 80, nonostante offload e tiling.
    #
    # VA CHIESTA LA VARIANTE *VARLEN*. HunyuanVideo passa una attn_mask (e' il
    # padding del prompt nella cross-attention), e le varianti dense di flash la
    # rifiutano entrambe. Provate e fallite sull'A100 il 17/09/2026:
    #   _flash_3_hub -> "`attn_mask` is not supported for flash-attn 3"
    #   flash_hub    -> "`attn_mask` is not supported for flash-attn 2"
    # In tutti e due i casi set_attention_backend() era RIUSCITO: si limita a
    # scaricare il kernel, e il guasto arriva al primo step, dopo il caricamento.
    #
    # flash_varlen_hub invece la maschera la gestisce: la normalizza, la converte
    # in cu_seqlens e impacchetta K e V (attention_dispatch.py di diffusers
    # 0.40.0). Il limite dichiarato li' riguarda solo il caso ring_degree > 1,
    # cioe' context parallel su piu' GPU, che qui non c'e'.
    # _native_efficient e' il kernel memory-efficient di PyTorch: accetta la
    # maschera, non chiede il pacchetto kernels, ed e' il ripiego.
    for backend in ("flash_varlen_hub", "_native_efficient"):
        try:
            pipe.transformer.set_attention_backend(backend)
            print("attention backend: %s" % backend)
            break
        except Exception as e:
            print("attention backend %s non disponibile (%s)" % (backend, e))
    else:
        print("attention backend: quello di default. A 121 frame in 720p e'"
              " probabile un OOM.")

    return {"pipe": pipe}


def generate(handle, prompt, seed):
    out = handle["pipe"](
        prompt=prompt,
        height=SPEC["height"],
        width=SPEC["width"],
        num_frames=SPEC["num_frames"],
        num_inference_steps=SPEC["steps"],
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
