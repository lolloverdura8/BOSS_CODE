# smoke_test.py
import sys, os

# La cartella di questo script contiene il clone del repo in sam3\ (senza
# __init__.py): come sys.path[0] farebbe ombra al package installato sam3,
# risolvendolo come namespace package con __file__ = None e rompendo il
# lookup delle risorse (assets/bpe_*). La si rimuove dal path.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
from sam3.model.sam3_image_processor import Sam3Processor

# SAM 3, non SAM 3.1, e la ragione e' vincolante: in questa release SAM 3.1
# esiste solo come modello VIDEO multiplex. Verificato in tre punti del sorgente:
#
#   - build_sam3_predictor(version="sam3.1") instrada a
#     build_sam3_multiplex_video_predictor (model_builder.py, riga 1244);
#   - il collo del modello immagine ha quattro livelli FPN
#     (scale_factors=[4.0, 2.0, 1.0, 0.5], riga 114), quello multiplex ne ha tre
#     (riga 932);
#   - sam3.1_multiplex.pt contiene infatti solo convs.0/1/2 nel ramo detector.
#
# Caricare il checkpoint 3.1 nel modello immagine lascia quindi il quarto livello
# FPN con pesi CASUALI - misurato: 1130 chiavi coperte su 1134 - e quel livello
# alimenta l'encoder. Il modello girerebbe lo stesso, producendo maschere
# plausibili e degradate, senza sollevare nulla: _load_checkpoint carica con
# strict=False e si limita a stampare. Per questo il controllo di copertura qui
# sotto e' un'uscita con errore e non una riga di log.
VERSION = "sam3"

IMAGE = "test_images/frame_000150.jpg"
PROMPT = "car"

# fp16 e non bf16. I due formati occupano gli stessi byte - 2 per valore - quindi
# la scelta non cambia la VRAM di un solo byte: cambia la distribuzione dei bit.
# fp16 ha piu' mantissa e meno esponente, quindi puo' andare in overflow, e
# quando succede restituisce NaN senza sollevare eccezioni. Per questo sotto c'e'
# un controllo esplicito sui valori non finiti.
DTYPE = torch.float16


def report_checkpoint_coverage(model, checkpoint_path):
    """Quante chiavi del modello il checkpoint copre davvero.

    Serve perche' _load_checkpoint() del builder carica con strict=False e si
    limita a STAMPARE le chiavi mancanti: un checkpoint incompatibile - per
    esempio la variante multiplex di SAM 3.1 su un modello immagine - lascerebbe
    parte dei pesi all'inizializzazione casuale e il modello girerebbe lo stesso,
    producendo maschere plausibili e prive di senso. Qui il dato diventa un
    numero da guardare invece di una riga di log da scorrere.
    """
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]
    detector = dict((k.replace("detector.", ""), v) for k, v in raw.items() if "detector" in k)

    sd = model.state_dict()
    missing = [k for k in sd if k not in detector]
    unexpected = [k for k in detector if k not in sd]
    mismatched = [k for k in detector if k in sd and tuple(detector[k].shape) != tuple(sd[k].shape)]
    covered = len(sd) - len(missing)

    print("checkpoint:        %s" % os.path.basename(checkpoint_path))
    print("chiavi coperte:    %d su %d (%.1f%%)" % (covered, len(sd), 100.0 * covered / len(sd)))
    if missing:
        print("chiavi MANCANTI:   %d, prime: %s" % (len(missing), missing[:5]))
    if mismatched:
        print("shape DISCORDI:    %d, prime: %s" % (len(mismatched), mismatched[:5]))
    if unexpected:
        print("chiavi in piu':    %d (attese sulla variante multiplex)" % len(unexpected))
    del raw, detector
    return len(missing), len(mismatched)


checkpoint_path = download_ckpt_from_hf(version=VERSION)
model = build_sam3_image_model(checkpoint_path=checkpoint_path)
n_missing, n_mismatched = report_checkpoint_coverage(model, checkpoint_path)

# model.half() dimezza i pesi residenti; torch.autocast da solo NON lo fa, perche'
# lascia i pesi in fp32 e converte solo le attivazioni. Su alcuni modelli pero'
# la combinazione rompe: gli operatori che autocast tiene in fp32 - layer norm in
# testa - si troverebbero pesi fp16 e ingressi fp32. Si prova la via economica e
# si ripiega su autocast da solo se non regge, dichiarando quale delle due ha
# funzionato.
precision_mode = "half+autocast"
try:
    model = model.half()
except Exception as exc:
    print("model.half() non applicabile (%s): si resta su autocast" % type(exc).__name__)
    precision_mode = "autocast"

processor = Sam3Processor(model)
image = Image.open(IMAGE)

torch.cuda.reset_peak_memory_stats()


def run():
    with torch.autocast("cuda", dtype=DTYPE):
        state = processor.set_image(image)
        return processor.set_text_prompt(state=state, prompt=PROMPT)


try:
    output = run()
except RuntimeError as exc:
    if precision_mode != "half+autocast":
        raise
    print("half+autocast fallito (%s), si riprova con il solo autocast" % exc)
    model = model.float()
    processor = Sam3Processor(model)
    precision_mode = "autocast"
    torch.cuda.reset_peak_memory_stats()
    output = run()

masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

# fp16 fallisce producendo NaN, non eccezioni: se succede il numero di detection
# resta plausibile e le maschere sono spazzatura. bf16 costa gli stessi byte.
non_finite = not torch.isfinite(scores).all() or not torch.isfinite(boxes).all()
if non_finite:
    print("\nATTENZIONE: output non finito in %s. Ripiego su bfloat16, che occupa"
          " gli stessi byte." % DTYPE)
    model = model.float()
    processor = Sam3Processor(model)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=PROMPT)
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    precision_mode = "autocast-bf16"

print("\n--- sintesi ---")
print("precisione:        %s" % precision_mode)
print("immagine:          %s (%dx%d)" % (IMAGE, image.width, image.height))
print("prompt:            %r" % PROMPT)
print("masks:             %s" % (tuple(masks.shape),))
print("boxes:             %s" % (tuple(boxes.shape),))
print("scores:            %s" % (tuple(scores.shape),))
print("num detections:    %d" % len(scores))
print("picco VRAM:        %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))

if n_missing or n_mismatched:
    sys.exit("checkpoint incompleto per questo modello: %d chiavi mancanti, %d shape"
             " discordi. I pesi non coperti sono rimasti casuali." % (n_missing, n_mismatched))
