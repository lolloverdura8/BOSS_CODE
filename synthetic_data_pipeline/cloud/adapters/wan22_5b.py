# wan22_5b.py — Wan2.2-TI2V-5B
#
# Derivato da wan2_2/generate_test_clips.py, che resta la referenza commentata.
# I parametri e le ottimizzazioni sono gli stessi, perche' sono quelli con cui e'
# stata generata monopattino_00 (seed 0, 569,1 s, picco allocato 12,538 GB): e'
# la clip contro cui si verifica che lo spostamento non abbia cambiato nulla.
#
# UNA COSA NON E' STATA PORTATA: torch.cuda.set_per_process_memory_fraction.
# Esiste solo perche' su Windows nativo il driver rispetta la Sysmem Fallback
# Policy, e serviva a trasformare uno sforamento in un OutOfMemoryError vero
# invece che in uno swap silenzioso su RAM. Su Linux quel fallback non c'e' e il
# tetto limiterebbe soltanto: su un'H100 da 80 GB taglierebbe via 8 GB per
# niente.
import gc

import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanPipeline

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

SPEC = {
    "model_id": MODEL_ID,
    "height": 704,
    "width": 1280,
    # 73 frame (3,04 s): il punto piu' lungo che la 5070 Ti regge. Il nativo del
    # modello e' 121, e su una scheda da 80 GB ci starebbe: resta 73 per il
    # bake-off, perche' e' il punto gia' misurato e cambiarlo introdurrebbe una
    # variabile in piu' proprio nel modello che fa da riferimento.
    "num_frames": 73,
    "steps": 50,
    "guidance": 5.0,
    "fps": 24,
    "dtype": "bfloat16 (VAE fp32)",
    # Negative prompt ufficiale del modello.
    "negative": (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
        "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
        "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
        "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    ),
}

# VRAM libera minima prima di caricare. Nello sweep del 28/08 il punto a 73 frame
# e' riuscito con 14,02 GiB di picco prenotato: sotto questa soglia il run
# rischia un OOM a meta' clip dopo minuti di denoising. E' una proprieta' del
# modello, non della scheda, quindi resta qui e non in gpu.py.
VRAM_FREE_MIN_GB = 14.2

# Il caricamento tiene in RAM host text encoder (11,36 GB) + transformer (~10 GB)
# + VAE fp32 (2,82 GB) prima che enable_model_cpu_offload li ridimensioni. Su una
# macchina da 31,8 GB il margine era di 1-4 GB, ed e' cio' che ha fermato il run
# notturno del 14/09. Sui pod da 80 GB di VRAM la RAM host e' abbondante, ma il
# controllo resta: e' il vincolo che ha spostato tutto questo lavoro sul cloud.
HOST_RAM_MIN_GB = 16.0

# Dopo il rilascio del text encoder sulla GPU non deve restare quasi nulla.
RELEASE_MAX_GB = 0.5


def _preflight():
    # gpu.host_ram() e non psutil: sul pod psutil riporta la RAM della macchina
    # e questa guardia non scatterebbe mai, proprio dove serve. Vedi gpu.py.
    import gpu
    avail = gpu.host_ram()[1]
    if avail < HOST_RAM_MIN_GB:
        raise SystemExit("RAM host disponibile %.2f GB < %.1f GB richiesti dal caricamento"
                         " di Wan2.2-5B." % (avail, HOST_RAM_MIN_GB))
    free, total = torch.cuda.mem_get_info()
    free_gb = free / 1024**3
    print("pre-volo wan22_5b: VRAM libera %.2f GB su %.2f, RAM host libera %.2f GB"
          % (free_gb, total / 1024**3, avail))
    if free_gb < VRAM_FREE_MIN_GB:
        raise SystemExit("VRAM libera %.2f GB < %.1f GB richiesti a %d frame."
                         % (free_gb, VRAM_FREE_MIN_GB, SPEC["num_frames"]))


def load(prompts):
    """Carica, codifica TUTTI i prompt, e butta fuori il text encoder.

    UMT5-XXL pesa 11,36 GB, quasi meta' del modello, e serve una volta sola. Il
    torch.no_grad() e' obbligatorio, non prudenza: encode_prompt chiamato
    direttamente non e' sotto no_grad (lo e' solo pipe.__call__), e senza, il
    grafo di autograd trattiene i pesi, il rilascio non funziona e il picco sale
    oltre la scheda. E' la trappola n.1 scoperta su CogVideoX.
    """
    _preflight()

    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)

    pipe.text_encoder.to("cuda")
    embeds = {}
    with torch.no_grad():
        # Il negativo si codifica una volta sola invece che una per clip: con
        # do_classifier_free_guidance=False encode_prompt restituisce solo
        # l'embedding del testo passato, e il negativo e' un testo come un altro.
        for text in list(dict.fromkeys(list(prompts) + [SPEC["negative"]])):
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
        raise SystemExit("dopo il rilascio del text encoder restano %.2f GB allocati"
                         " (soglia %.1f): il rilascio non ha funzionato."
                         % (left_gb, RELEASE_MAX_GB))
    print("text encoder rilasciato: %.3f GB ancora allocati, %d embedding su CPU"
          % (left_gb, len(embeds)))

    # DOPO il rilascio, perche' enable_model_cpu_offload deve trovarlo gia' sparito.
    pipe.enable_model_cpu_offload()
    # Attivo a 704x1280, dove il decode e' il punto tipico di OOM. enable_slicing
    # resta per parita' con lo smoke test, anche se a batch 1 e' inerte: il VAE lo
    # applica solo se z.shape[0] > 1, cioe' sulla dimensione batch e non sui frame.
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    return {"pipe": pipe, "embeds": embeds}


def generate(handle, prompt, seed):
    pipe, embeds = handle["pipe"], handle["embeds"]
    if prompt not in embeds:
        raise KeyError("prompt non codificato in load(): il runner deve passare a load()"
                       " tutti i prompt che poi chiede a generate().")
    out = pipe(
        prompt=None,
        negative_prompt=None,
        prompt_embeds=embeds[prompt].to("cuda"),
        negative_prompt_embeds=embeds[SPEC["negative"]].to("cuda"),
        height=SPEC["height"],
        width=SPEC["width"],
        num_frames=SPEC["num_frames"],
        num_inference_steps=SPEC["steps"],
        guidance_scale=SPEC["guidance"],
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
    # Un'eccezione a meta' pipe() salta maybe_free_model_hooks: senza questa
    # chiamata il modulo che era in GPU ci resta anche per la clip successiva.
    handle["pipe"].maybe_free_model_hooks()


def unload(handle):
    handle["pipe"] = None
    handle["embeds"] = None
    gc.collect()
    torch.cuda.empty_cache()
