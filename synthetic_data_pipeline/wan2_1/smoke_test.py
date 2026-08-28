# smoke_test.py
#
# Generazione singola di verifica: produce un video guardabile per confermare che
# l'ambiente funzioni end-to-end. Non e' uno strumento di misura (per quello c'e'
# vram_benchmark.py), ma condivide con esso seed e parametri nativi, cosi' il video
# a 81 frame e' la controparte visiva della riga frames=81 del CSV del benchmark.
#
# E' anche il video che rende finalmente confrontabile a occhio Wan contro CogVideoX:
# 832x480 contro 480x720 sono +15% di pixel per frame, mentre Wan2.2 a 704x1280 ne
# aveva 2.6x e la differenza visiva era dominata dalla risoluzione, non dal modello.
import gc
import os
import time

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# Risoluzione nativa del modello: 480P, cioe' 832x480 (il 720p e' documentato come
# "meno stabile" su questa taglia). Vincolo di forma: multipli di
# vae_scale_factor_spatial (8, non 16 come in Wan2.2) x patch_size (2) = 16.
HEIGHT, WIDTH = 480, 832
# Durata = NUM_FRAMES / FPS: 81 frame = 5.06 s, il nativo del modello.
# Vincolo del VAE: num_frames = 4k+1.
NUM_FRAMES = 81
NUM_STEPS = 50           # default del modello
GUIDANCE = 5.0           # default della model card diffusers
# Frame rate nativo. La card diffusers esporta a fps=15, il repo upstream
# Wan-Video/Wan2.1 usa 16: si segue l'upstream, perche' e' il frame rate di
# addestramento ed esportare piu' lento falsa il moto.
FPS = 16

# Seed fisso, stesso valore usato da vram_benchmark.py: senza generator il rumore
# latente iniziale cambia a ogni run e due esecuzioni danno video diversi. Fissandolo
# lo smoke test diventa riproducibile (un run andato storto si puo' rifare identico) e
# confrontabile con il punto a 81 frame del benchmark, che parte dallo stesso rumore.
SEED = 0

# Stesso tetto all'allocatore imposto dal benchmark, per la stessa ragione: su Windows
# nativo il driver rispetta la Sysmem Fallback Policy, quindi un tetto esplicito fa
# fallire l'allocazione PRIMA del cudaMalloc che innescherebbe lo swap silenzioso su
# RAM, e si ottiene un OutOfMemoryError vero invece di un run centinaia di volte piu'
# lento. Tenerlo identico nei due script e' anche cio' che rende confrontabile il picco
# stampato qui con la riga del CSV.
VRAM_FRACTION = 0.90
torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION, device=0)

# Il VAE resta in fp32 (in bf16 produce artefatti di decodifica); transformer e
# text encoder in bf16.
vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)

# Stesso prompt positivo di vram_benchmark.py, per confronto diretto.
prompt = (
    "POV shot from a person walking along a city sidewalk on a sunny afternoon, "
    "steady forward motion, parked cars lining the left side of the street, "
    "shop fronts and awnings on the right, pedestrians ahead in the distance, "
    "sharp focus, natural daylight, realistic urban environment, handheld camera feel,"
    "mass presence of trees, poles and suspended branches,"
)

# Negative prompt ufficiale del modello. La card del 1.3B lo riporta in inglese, quella
# di Wan2.2-TI2V-5B in cinese: sono la traduzione l'una dell'altra.
negative_prompt = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG "
    "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly "
    "drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, many people in the background, walking backwards"
)

# Il text encoder UMT5-XXL (10.58 GB di soli pesi) e' lo STESSO di Wan2.2 ed e' qui il
# componente singolo piu' pesante, quattro volte il transformer da 1.3B. Serve una
# volta sola, all'inizio: si calcolano gli embedding subito e poi lo si elimina. Non e'
# un dettaglio di misura ma di risorse: enable_model_cpu_offload() non lo elimina, lo
# parcheggia nella RAM di sistema e lo risveglia sulla GPU quando serve, quindi senza
# questo blocco quei 10.58 GB restano occupati in RAM host per tutti i 50 step. In piu'
# il picco VRAM stampato a fine run resterebbe dominato dal text encoder invece di
# misurare la generazione, e non sarebbe confrontabile col CSV.
# Il transformer non si puo' eliminare allo stesso modo: viene invocato a ogni step,
# due volte per step per via del CFG. Per lui la leva e' l'offload, non il rilascio.
# no_grad e' obbligatorio: solo pipe.__call__ e' decorato @torch.no_grad, mentre
# encode_prompt chiamato direttamente costruisce il grafo di autograd, che trattiene
# sia le attivazioni del text encoder sia i suoi pesi e rende inefficace il rilascio.
pipe.text_encoder.to("cuda")
with torch.no_grad():
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        # PER GENERARE PIU' VIDEO DALLO STESSO PROMPT si alza questo valore.
        # WanPipeline.__call__ onora davvero l'omonimo parametro (a differenza di
        # CogVideoX, dove viene riassegnato a 1 in silenzio): i due punti sono
        # equivalenti, basta alzarlo in uno. Due conseguenze da sistemare quando si
        # alzera' N:
        #   - pipe(...).frames conterrebbe N video, mentre qui sotto si esporta
        #     .frames[0]: servirebbe un ciclo sull'export con nomi di file distinti;
        #   - enable_slicing() smette di essere inerte (vedi commento piu' sotto).
        num_videos_per_prompt=1,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )

# Liberare davvero il text encoder richiede tre passaggi: togliere il riferimento
# (= None), forzare il garbage collector a distruggere l'oggetto (gc.collect) e
# restituire al driver la VRAM che l'allocatore PyTorch teneva in cache (empty_cache).
pipe.text_encoder = None
gc.collect()
torch.cuda.empty_cache()

# enable_model_cpu_offload va chiamato DOPO il blocco sopra: installa hook sui componenti
# registrati e li sposta su CPU, quindi deve trovare il text encoder gia' sparito.
pipe.enable_model_cpu_offload()
# tiling: spezza ogni frame in riquadri. La soglia e' 256 pixel / 8 di compressione
# spaziale = 32 in latente, e a 832x480 il latente e' 104x60: realmente attivo.
pipe.vae.enable_tiling()
# slicing: spezza il decode lungo la dimensione BATCH. A num_videos_per_prompt=1 il
# latente in ingresso al decode ha batch 1 e la riga e' inerte, perche' il VAE la applica
# solo se z.shape[0] > 1; si tiene per parita' con vram_benchmark.py e perche' diventera'
# attiva alzando num_videos_per_prompt (vedi commento sopra). Da non confondere con lo
# spezzettamento del decode lungo l'asse temporale, che nel VAE di Wan e' sempre attivo
# e piu' fine: un frame latente per volta, con lo stato tenuto nel feat_cache.
pipe.vae.enable_slicing()

os.makedirs("outputs", exist_ok=True)

# Azzerare i contatori qui e' cio' che rende interpretabile il picco stampato a fine run:
# senza, includerebbe il caricamento del modello e la fase di encoding del text encoder,
# cioe' proprio quello che il blocco sopra serve a togliere di mezzo.
torch.cuda.reset_peak_memory_stats()

t0 = time.time()
frames = pipe(
    # prompt e negative_prompt vanno a None: check_inputs solleva se si passano
    # insieme alla loro versione precalcolata.
    prompt=None,
    negative_prompt=None,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    height=HEIGHT,
    width=WIDTH,
    num_frames=NUM_FRAMES,
    num_inference_steps=NUM_STEPS,
    guidance_scale=GUIDANCE,
    generator=torch.Generator(device="cuda").manual_seed(SEED),
).frames[0]
elapsed = time.time() - t0

# export_to_video sceglie il backend a runtime: con imageio + imageio-ffmpeg installati
# scrive in H.264, altrimenti ricade sul ramo OpenCV (deprecato in diffusers) che scrive
# in mp4v. 480 e 832 sono entrambi divisibili per 16, quindi il macro_block_size di
# default non fa riscalare l'immagine.
export_to_video(frames, "outputs/smoke_test.mp4", fps=FPS)

# Tre metriche di VRAM, che raccontano cose diverse: alloc e' il picco dei tensori
# effettivamente in uso, reserved il picco di quanto PyTorch ha prenotato al driver
# (>= alloc, include la cache trattenuta per riuso: il divario fra i due misura la
# frammentazione), device_used quanto risulta occupato sulla scheda intera (desktop
# Windows compreso, ~1.4 GB a riposo).
# Attenzione: mem_get_info viene letto DOPO il ritorno di pipe(), quindi dopo che
# maybe_free_model_hooks ha gia' scaricato i moduli dalla GPU: device_used sottostima il
# transiente del decode. Il dato affidabile e' il picco allocato, che e' un picco vero e
# non una lettura istantanea.
vram_free, vram_total = torch.cuda.mem_get_info()
vram_total_gb = vram_total / 1024**3
vram_budget_gb = vram_total_gb * VRAM_FRACTION
alloc_peak_gb = torch.cuda.max_memory_allocated() / 1024**3
reserved_peak_gb = torch.cuda.max_memory_reserved() / 1024**3
device_used_gb = (vram_total - vram_free) / 1024**3

# Wan restituisce output_type="np" di default, quindi frames e' un array numpy
# (T, H, W, 3) e si stampa .shape (CogVideoX restituisce PIL e usa .size).
print("frames:", len(frames), frames[0].shape, "durata: %.2f s" % (len(frames) / FPS))
print("seed:", SEED)
print("tempo generazione: %.1f s" % elapsed)
print("picco VRAM allocata: %.2f GB" % alloc_peak_gb)
print("picco VRAM prenotata: %.2f GB" % reserved_peak_gb)
print("VRAM occupata sulla scheda: %.2f GB su %.2f GB" % (device_used_gb, vram_total_gb))

# Col tetto dell'allocatore attivo il segnale atteso di esaurimento VRAM e' un
# OutOfMemoryError vero, non questo avviso: l'euristica resta come rete di sicurezza
# per le allocazioni che NON passano dall'allocatore PyTorch (workspace di cuDNN e
# cuBLAS), che il tetto non copre. Il confronto e' col budget effettivo e non con la
# capacita' fisica della scheda, altrimenti col tetto attivo non scatterebbe mai.
if alloc_peak_gb >= vram_budget_gb * 0.98:
    print("ATTENZIONE: sospetto sysmem fallback (picco allocato >= 98%% del budget"
          " di %.2f GB). Il tempo di generazione sopra non e' rappresentativo."
          % vram_budget_gb)
