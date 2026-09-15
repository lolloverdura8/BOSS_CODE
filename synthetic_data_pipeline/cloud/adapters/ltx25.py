# ltx25.py — LTX-2.5 (Lightricks)
#
# 22B parametri, gennaio 2026, supera LTX-2.3 e LTX-2. Come HunyuanVideo-1.5 non
# e' nel §2.4 di OR4.1: e' uscito dopo.
#
# ERA IL CANDIDATO PIU' A RISCHIO DEL BAKE-OFF, E NON LO E' PIU'. Il piano lo
# dava come da installare col pacchetto proprio ltx-pipelines via uv, fuori da
# diffusers, e da scartare se non fosse partito entro un'ora a S2. Controllando
# il venv il 15/09/2026, diffusers 0.40.0 espone gia' LTX2Pipeline e tutta la
# famiglia LTX2*: nessun pacchetto in piu', nessun uv, stesso ambiente degli
# altri generatori.
#
# DUE COSE DA SAPERE PRIMA DI S2:
#
# 1. Lightricks/LTX-2.5-Diffusers e' GATED (gated: auto sull'API HuggingFace,
#    verificato il 15/09/2026). L'approvazione e' automatica ma le condizioni
#    vanno accettate una volta sulla pagina del modello, con l'account
#    lolloverdura8, ALTRIMENTI il download fallisce sul pod a GPU accesa.
# 2. La licenza e' una community license con tetto di fatturato ($10M/anno), non
#    una Apache 2.0 come Wan. Se LTX vince il bake-off, la licenza va letta prima
#    di generare il dataset di addestramento di un prodotto. Non e' una decisione
#    tecnica.
import gc

import numpy as np
import torch
from diffusers import LTX2Pipeline

MODEL_ID = "Lightricks/LTX-2.5-Diffusers"

SPEC = {
    "model_id": MODEL_ID,
    # Il vincolo del modello e' che altezza e larghezza siano multipli di 32.
    # 704 = 22x32 e 1280 = 40x32, che e' anche la risoluzione di Wan2.2-5B:
    # un confondimento in meno fra i due.
    "height": 704,
    "width": 1280,
    # Il vincolo e' num_frames % 8 == 1. 121 = 15x8 + 1.
    "num_frames": 121,
    # I default della pipeline sono 30 step e guidance 3.0. Restano quelli: sono
    # cio' che il modello dichiara come punto di lavoro, e alzarli per pareggiare
    # i 50 step degli altri vorrebbe dire scegliere io il suo punto operativo
    # invece di usare il suo.
    "steps": 30,
    "guidance": 3.0,
    "fps": 24,
    "dtype": "bfloat16",
    "negative": None,
}


def load(prompts):
    pipe = LTX2Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    try:
        pipe.vae.enable_tiling()
    except AttributeError:
        # Il VAE di LTX-2 e' un'altra classe (AutoencoderKLLTX2Video): se non
        # espone il tiling non e' un errore, e' solo un'ottimizzazione in meno.
        print("VAE senza enable_tiling: si prosegue senza.")
    return {"pipe": pipe}


def generate(handle, prompt, seed):
    out = handle["pipe"](
        prompt=prompt,
        height=SPEC["height"],
        width=SPEC["width"],
        num_frames=SPEC["num_frames"],
        frame_rate=float(SPEC["fps"]),
        num_inference_steps=SPEC["steps"],
        guidance_scale=SPEC["guidance"],
        # LTX2Pipeline ha output_type="pil" di default: senza questo, i frame
        # tornerebbero uint8 in 0..255 invece che float in 0..1.
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
