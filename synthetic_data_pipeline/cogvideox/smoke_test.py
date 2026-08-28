# smoke_test.py
#
# Generazione singola di verifica: produce un video guardabile per confermare che
# l'ambiente funzioni end-to-end. Non e' uno strumento di misura (per quello c'e'
# vram_benchmark.py), ma condivide con esso seed e parametri nativi, cosi' il video
# a 49 frame e' la controparte visiva della riga frames=49 del CSV del benchmark.
import gc
import os
import time

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

MODEL_ID = "THUDM/CogVideoX-2b"

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

# Seed fisso, stesso valore usato da vram_benchmark.py: senza generator il rumore
# latente iniziale cambia a ogni run e due esecuzioni danno video diversi. Fissandolo
# lo smoke test diventa riproducibile (un run andato storto si puo' rifare identico) e
# confrontabile con il punto a 49 frame del benchmark, che parte dallo stesso rumore.
SEED = 0

# A differenza di Wan, CogVideoX non ha un vincolo documentato di VAE in fp32:
# l'intera pipeline viene caricata in fp16 (model card THUDM: CogVideoX-2b
# raccomanda fp16, a differenza del 5b che raccomanda bf16). Se in fase di
# verifica emergono artefatti di decodifica, passare a VAE separato in fp32
# come per Wan.
pipe = CogVideoXPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)

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

# Il text encoder T5 (8.87 GB di soli pesi, il 71% del modello) serve una volta sola,
# all'inizio: si calcolano gli embedding subito e poi lo si elimina. Non e' un dettaglio
# di misura ma di risorse: enable_model_cpu_offload() non elimina T5, lo parcheggia nella
# RAM di sistema e lo risveglia sulla GPU quando serve, quindi senza questo blocco quei
# 8.87 GB restano occupati in RAM host per tutti i 50 step, e la RAM host e' il vincolo
# stretto di questa VM (15 GB totali). In piu' il picco VRAM stampato a fine run
# resterebbe dominato da T5 (~10.8 GB) invece di misurare la generazione.
# encode_prompt con negative_prompt=None e do_classifier_free_guidance=True usa
# internamente la stringa vuota (negative_prompt or ""), quindi il risultato e' identico
# a quello che si otteneva passando negative_prompt=None a pipe().
# no_grad e' obbligatorio: solo pipe.__call__ e' decorato @torch.no_grad, mentre
# encode_prompt chiamato direttamente costruisce il grafo di autograd. Il grafo trattiene
# sia le attivazioni di T5 (~3.6 GB) sia i suoi pesi, rendendo inefficace il rilascio
# sotto: misurati 14.37 GB ancora occupati dopo il free, con picco a 18.21 GB su una
# scheda da 16.3 GB (quindi sysmem fallback e 98 s/step invece di 1.7 s/step).
pipe.text_encoder.to("cuda")
with torch.no_grad():
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        # PER GENERARE PIU' VIDEO DALLO STESSO PROMPT SI ALZA QUESTO VALORE, NON
        # l'omonimo parametro di pipe(): in diffusers 0.40.0 __call__ dichiara
        # num_videos_per_prompt nella firma ma lo riassegna a 1 incondizionatamente
        # prima di usarlo, quindi passarlo li' non ha alcun effetto e non solleva
        # nessun errore. Qui invece funziona: _get_t5_prompt_embeds fa repeat + view,
        # gli embedding escono con shape [N, seq_len, dim] e __call__ ricava
        # batch_size = prompt_embeds.shape[0] (i nostri embedding gli passano
        # attraverso intatti, i suoi rami interni sono protetti da "if ... is None").
        # Tre conseguenze da sistemare quando si alzera' N:
        #   - pipe(...).frames conterrebbe N video, mentre qui sotto si esporta
        #     .frames[0]: servirebbe un ciclo sull'export con nomi di file distinti;
        #   - il costo del denoising scala ~xN (i latenti sono N, 2N con CFG);
        #   - enable_slicing() smette di essere inerte (vedi commento piu' sotto).
        num_videos_per_prompt=1,
        device=torch.device("cuda"),
        dtype=torch.float16,
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
# tiling: spezza ogni frame in riquadri, ed e' realmente attivo qui (la soglia e' meta'
# della risoluzione nativa, latente 30x45, e il nostro latente e' 60x90).
pipe.vae.enable_tiling()
# slicing: spezza il decode lungo la dimensione BATCH. A num_videos_per_prompt=1 il
# latente in ingresso al decode ha batch 1 e la riga e' inerte, perche' il VAE la applica
# solo se z.shape[0] > 1; si tiene per parita' con vram_benchmark.py e perche' diventera'
# attiva alzando num_videos_per_prompt (vedi commento sopra). Da non confondere con lo
# spezzettamento del decode lungo l'asse temporale (num_latent_frames_batch_size = 2),
# che e' sempre attivo e non dipende da questa chiamata.
pipe.vae.enable_slicing()

os.makedirs("outputs", exist_ok=True)

# Azzerare i contatori qui e' cio' che rende interpretabile il picco stampato a fine run:
# senza, includerebbe il caricamento del modello e la fase di encoding di T5, cioe'
# proprio quello che il blocco sopra serve a togliere di mezzo.
torch.cuda.reset_peak_memory_stats()

t0 = time.time()
frames = pipe(
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
# in mp4v. 480 e 720 sono entrambi divisibili per 16, quindi il macro_block_size di
# default non fa riscalare l'immagine.
export_to_video(frames, "outputs/smoke_test.mp4", fps=FPS)

# Tre metriche di VRAM, che raccontano cose diverse: alloc e' il picco dei tensori
# effettivamente in uso, reserved il picco di quanto PyTorch ha prenotato al driver
# (>= alloc, include la cache trattenuta per riuso: il divario fra i due misura la
# frammentazione), device_used quanto risulta occupato sulla scheda intera.
# Attenzione: mem_get_info viene letto DOPO il ritorno di pipe(), quindi dopo che
# maybe_free_model_hooks ha gia' scaricato i moduli dalla GPU: device_used sottostima il
# transiente del decode. Il dato affidabile e' il picco allocato, che e' un picco vero e
# non una lettura istantanea.
vram_free, vram_total = torch.cuda.mem_get_info()
vram_total_gb = vram_total / 1024**3
alloc_peak_gb = torch.cuda.max_memory_allocated() / 1024**3
reserved_peak_gb = torch.cuda.max_memory_reserved() / 1024**3
device_used_gb = (vram_total - vram_free) / 1024**3

print("frames:", len(frames), frames[0].size, "durata: %.2f s" % (len(frames) / FPS))
print("seed:", SEED)
print("tempo generazione: %.1f s" % elapsed)
print("picco VRAM allocata: %.2f GB" % alloc_peak_gb)
print("picco VRAM prenotata: %.2f GB" % reserved_peak_gb)
print("VRAM occupata sulla scheda: %.2f GB su %.2f GB" % (device_used_gb, vram_total_gb))

# Sotto WSL2 il driver ignora "CUDA - Sysmem Fallback Policy" (microsoft/WSL#11050,
# chiusa senza fix): un esaurimento di VRAM non produce un OutOfMemoryError ma uno swap
# silenzioso su RAM di sistema via PCIe, e lo script continua a funzionare centinaia di
# volte piu' lentamente. Un picco dell'allocatore che eguaglia la capacita' fisica della
# scheda puo' essere servito solo spillando su RAM: e' il segnale diretto del fallback,
# ed e' lo stesso criterio usato da vram_benchmark.py.
if alloc_peak_gb >= vram_total_gb * 0.98:
    print("ATTENZIONE: sospetto sysmem fallback (picco allocato >= 98%% della VRAM"
          " fisica). Il tempo di generazione sopra non e' rappresentativo.")
