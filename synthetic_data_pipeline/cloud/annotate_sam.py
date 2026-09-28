# annotate_sam.py
#
# SAM 3 sui frame generati da cloud/runner.py.
#
# Qui non c'e' ground truth: il generatore inventa la scena e nessuno dice dove
# sta l'oggetto. Non si calcola quindi uno IoU ma il TASSO DI INDIVIDUAZIONE per
# frame in modalita' testo, cioe' quanto spesso SAM trova da solo l'oggetto che il
# prompt di generazione ha chiesto. E' la misura che decide il bivio della Fase 1
# (PIANO_ATTACCO_OR4.1.md, passo 0.2), da leggere contro il 54,5% della bicicletta
# CARLA.
#
# Per gli ostacoli sospesi si cerca anche il SUPPORTO (tronco, palo): la sua
# maschera serve ad annotate_da3.py per il controllo di
# coerenza della profondita'. Maschere (.npz), overlay per la revisione a occhio e
# CSV finiscono in --out-dir.
#
# Dal 28/09/2026, per i sospesi, oggetto e supporto non sono piu' le istanze di
# score massimo prese ciascuna per conto suo: si raccolgono tutti i candidati e la
# scelta la fa sam_selezione.py, per posizione e contatto. I candidati restano in
# candidati/<clip>/NNN.npz, e --riseleziona rifa' la scelta senza caricare SAM.
import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

# gpu.py e sam_selezione.py stanno in questa stessa cartella, quindi vanno
# importati PRIMA della riga sotto.
import gpu
import sam_selezione

# Stessa ragione di smoke_test.py: un clone in sam3\ farebbe ombra al package.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

import numpy as np
import torch
from PIL import Image, ImageDraw

from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

# SAM 3 e non SAM 3.1: il perche' e' in smoke_test.py.
VERSION = "sam3"
DTYPE = torch.float16

TELEMETRY_EVERY = 10

# Soglie di memoria e temperatura: riempite in main() dalla CLI, con i default di
# gpu.py ricavati dalla scheda presente. Prima erano cablate a 13,0 e 6,0 GB,
# tarate sui 16 GB della 5070 Ti: su un'altra scheda fermavano il run sbagliando.
VRAM_BUDGET_GB = None
VRAM_FREE_MIN_GB = None
TEMP_LIMIT_C = None

# classe BOSS -> (prompt oggetto, prompt supporto o None). Il supporto esiste solo
# per i sospesi, ed e' la cosa a cui l'oggetto e' fissato nei prompt di generazione.
TEXT_PROMPTS = {
    "monopattino": ("electric scooter", None),
    # Prove del 23 e 28/09 sulle due clip del ramo di wan22_5b. "tree branch" e "low
    # hanging branch over the sidewalk" trovano 38/38 ma sulla parte legnosa o su un
    # ramo caduto a terra; "leafy" sposta la maschera sulle foglie che sporgono, cioe'
    # sull'ostacolo vero (26/38). Aggiungere "over the sidewalk" riporta il ramo a terra.
    "ramo_sporgente": ("low hanging leafy branch", "tree trunk"),
    "insegna_cartello_basso": ("sign", "pole"),
    "ostacolo_sospeso_generico": ("horizontal bar", "pole"),
}

# Secondo prompt dell'oggetto, i cui candidati si uniscono a quelli del primo quando
# si toccano (sam_selezione.py). Scheda del 28/09: "leafy" prende le foglie ma perde
# il tratto legnoso fra tronco e foglie, "tree branch" prende proprio quello.
EXTRA_OBJECT_PROMPTS = {
    "ramo_sporgente": "tree branch",
}

# Colonne diagnostiche della selezione, in coda al CSV: annotate_da3.py legge solo
# clip_id, classe e frame, quindi aggiungerle non lo tocca.
DIAG_KEYS = ("prompt_extra", "esito_selezione", "n_cand_oggetto", "n_cand_extra",
             "n_cand_supporto", "n_doppioni_supporto", "seme_prompt", "seme_y1", "seme_y2",
             "area_seme", "area_finale", "n_uniti", "n_scartati_unione",
             "px_sottratti_supporto", "n_supporti_contatto", "criterio_supporto",
             "dist_supporto_px", "terminale_degenere", "n_scartati_non_finiti",
             "n_maschere_vuote", "n_supporti_scavalcati")

# Gruppi di candidati salvati per frame in candidati/<clip>/NNN.npz.
GRUPPI_CANDIDATI = ("oggetto", "extra", "supporto")

# Tasso di individuazione in modalita' testo della bicicletta CARLA:
# sam_eval_20260908_114604.csv, 18 istanze trovate su 33.
CARLA_TEXT_RATE_BICICLETTA = 54.5

OVERLAY_ALPHA = 0.45
COLOR_OBJ = (255, 0, 0)
COLOR_SUP = (0, 90, 255)

# Riempita in main() da --out-dir.
OUT_DIR = None


def report_checkpoint_coverage(model, checkpoint_path):
    """Copiata da smoke_test.py: quante chiavi del modello il checkpoint copre davvero."""
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]
    detector = dict((k.replace("detector.", ""), v) for k, v in raw.items() if "detector" in k)

    sd = model.state_dict()
    missing = [k for k in sd if k not in detector]
    mismatched = [k for k in detector if k in sd and tuple(detector[k].shape) != tuple(sd[k].shape)]
    covered = len(sd) - len(missing)
    print("checkpoint:        %s" % os.path.basename(checkpoint_path))
    print("chiavi coperte:    %d su %d (%.1f%%)" % (covered, len(sd), 100.0 * covered / len(sd)))
    del raw, detector
    return len(missing), len(mismatched)


def to_numpy_masks(output):
    """Le maschere di SAM come array booleano (N, H, W)."""
    m = output["masks"]
    if m is None or len(m) == 0:
        return np.zeros((0, 0, 0), dtype=bool)
    m = m.detach().to(torch.bool).cpu().numpy()
    if m.ndim == 4:
        m = m[:, 0]
    return m


def build_model():
    """Da eval_on_carla.py: model.half() su questo modello non regge, e il probe
    in evaluate() lo scopre al primo frame e ripiega su autocast da solo."""
    checkpoint_path = download_ckpt_from_hf(version=VERSION)
    model = build_sam3_image_model(checkpoint_path=checkpoint_path)
    mode = "half+autocast"
    try:
        model = model.half()
    except Exception as exc:
        print("model.half() non applicabile (%s): si resta su autocast"
              % type(exc).__name__, file=sys.stderr)
        mode = "autocast"
    return model, checkpoint_path, mode


def best_detection(output):
    """La detection con score massimo, estratta PRIMA di reset_all_prompts, che
    cancella masks/boxes/scores dallo stato."""
    masks = to_numpy_masks(output)
    if len(masks) == 0:
        return None, 0, True
    scores = output["scores"].detach().float().cpu().numpy()
    finite = bool(np.isfinite(scores).all())
    k = int(np.argmax(scores))
    box = output["boxes"][k].detach().float().cpu().numpy()
    return {"mask": masks[k], "score": float(scores[k]), "box": box}, len(masks), finite


def all_detections(output) -> tuple[list[dict], int, int]:
    """(candidati, non finiti, maschere vuote): tutte le istanze sopra soglia,
    estratte PRIMA di reset_all_prompts. Upstream la soglia scarta gia' gli score
    NaN, quindi il controllo utile e' su box e maschere: un candidato non finito si
    scarta e si conta, invece di arrivare fino a annotate_da3.py. Le maschere vuote
    si contano a parte: non sono un guasto fp16."""
    masks = to_numpy_masks(output)
    if len(masks) == 0:
        return [], 0, 0
    scores = output["scores"].detach().float().cpu().numpy()
    boxes = output["boxes"].detach().float().cpu().numpy()
    cands, non_finiti, vuote = [], 0, 0
    for k in range(len(masks)):
        if not (np.isfinite(scores[k]) and np.isfinite(boxes[k]).all()):
            non_finiti += 1
        elif not masks[k].any():
            vuote += 1
        else:
            cands.append({"mask": masks[k], "score": float(scores[k]), "box": boxes[k]})
    return cands, non_finiti, vuote


def current_prompts(cls: str) -> dict:
    obj_prompt, sup_prompt = TEXT_PROMPTS[cls]
    return {"oggetto": obj_prompt, "extra": EXTRA_OBJECT_PROMPTS.get(cls),
            "supporto": sup_prompt}


def save_candidates(path: str, gruppi: dict, shape: tuple[int, int], cls: str,
                    clips_dir: str) -> None:
    """Maschere impacchettate a bit: si puo' rifare la selezione senza SAM. Con i
    prompt e la cartella delle clip, che --riseleziona confronta con quelli di
    adesso: candidati di prompt vecchi con regole nuove darebbero numeri misti."""
    H, W = shape
    prompts = current_prompts(cls)
    arrays = {"shape": np.array([H, W]), "clips_dir": np.array(os.path.abspath(clips_dir))}
    for nome in GRUPPI_CANDIDATI:
        arrays["prompt_" + nome] = np.array(prompts[nome] or "")
        cands = gruppi[nome]
        masks = np.stack([c["mask"] for c in cands]) if cands else np.zeros((0, H, W), bool)
        arrays[nome + "_maschere"] = np.packbits(masks, axis=-1)
        arrays[nome + "_score"] = np.array([c["score"] for c in cands], np.float32)
        arrays[nome + "_box"] = np.array([c["box"] for c in cands], np.float32).reshape(-1, 4)
    np.savez_compressed(path, **arrays)


def load_candidates(path: str, cls: str, clips_dir: str) -> dict:
    z = np.load(path)
    for nome, atteso in current_prompts(cls).items():
        salvato = str(z["prompt_" + nome])
        if salvato != (atteso or ""):
            sys.exit("%s: candidati generati con il prompt %s %r, oggi %r. Va rifatto"
                     " il run con SAM." % (path, nome, salvato, atteso))
    if str(z["clips_dir"]) != os.path.abspath(clips_dir):
        sys.exit("%s: candidati da %s, --clips-dir e' %s" % (path, z["clips_dir"], clips_dir))
    W = int(z["shape"][1])
    gruppi = {}
    for nome in GRUPPI_CANDIDATI:
        masks = np.unpackbits(z[nome + "_maschere"], axis=-1, count=W).astype(bool)
        gruppi[nome] = [{"mask": m, "score": float(s), "box": b} for m, s, b in
                        zip(masks, z[nome + "_score"], z[nome + "_box"], strict=True)]
    return gruppi


def load_clips(clips_dir):
    path = os.path.join(clips_dir, "manifest.jsonl")
    if not os.path.exists(path):
        sys.exit("manifest non trovato: %s. Prima cloud/runner.py." % path)
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                # l'ultima riga vince: un rilancio dopo un OOM riscrive l'esito
                records[rec["clip_id"]] = rec
    clips = []
    for rec in records.values():
        pngs = sorted(glob.glob(os.path.join(clips_dir, "frames", rec["clip_id"], "*.png")))
        if rec["esito"] != "oom" and pngs:
            if rec["classe"] not in TEXT_PROMPTS:
                sys.exit("classe %r senza prompt in TEXT_PROMPTS" % rec["classe"])
            clips.append((rec, pngs))
    return clips


def save_overlay(image, obj, sup, label, path):
    rgb = np.asarray(image).astype(np.float32)
    for det, color in ((sup, COLOR_SUP), (obj, COLOR_OBJ)):
        if det is not None:
            m = det["mask"]
            rgb[m] = rgb[m] * (1.0 - OVERLAY_ALPHA) + np.array(color, np.float32) * OVERLAY_ALPHA
    img = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for det, color, name in ((sup, COLOR_SUP, "supporto"), (obj, COLOR_OBJ, label)):
        if det is not None:
            x0, y0, x1, y1 = [float(v) for v in det["box"]]
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            draw.text((x0 + 4, max(0.0, y0 - 14)), "%s %.2f" % (name, det["score"]), fill=color)
    img.save(path, quality=90)


def detect_groups(processor, image: Image.Image, cls: str) -> tuple[dict, dict]:
    """(gruppi, scarti) dei tre prompt di una classe sospesa, con una sola
    set_image: reset_all_prompts non tocca le feature dell'immagine. Il prompt
    dell'oggetto va per primo, sullo state appena creato, come prima del 28/09."""
    prompts = current_prompts(cls)
    gruppi, primo = {}, True
    scarti = {"n_scartati_non_finiti": 0, "n_maschere_vuote": 0}
    with torch.autocast("cuda", dtype=DTYPE):
        state = processor.set_image(image)
        for nome in GRUPPI_CANDIDATI:
            if not prompts[nome]:
                gruppi[nome] = []
                continue
            if not primo:
                processor.reset_all_prompts(state)
            primo = False
            gruppi[nome], non_finiti, vuote = all_detections(
                processor.set_text_prompt(state=state, prompt=prompts[nome]))
            scarti["n_scartati_non_finiti"] += non_finiti
            scarti["n_maschere_vuote"] += vuote
    return gruppi, scarti


def select_frame(gruppi: dict, scarti: dict, cls: str,
                 shape: tuple[int, int]) -> tuple[dict | None, dict | None, dict]:
    sel = sam_selezione.seleziona(gruppi["oggetto"], gruppi["extra"], gruppi["supporto"], shape)
    diag = dict(sel.diag, prompt_extra=EXTRA_OBJECT_PROMPTS.get(cls, ""), **scarti)
    return sel.oggetto, sel.supporto, diag


def write_frame(clip_id: str, cls: str, frame: int, image: Image.Image,
                obj: dict | None, sup: dict | None) -> None:
    """Il .npz letto da annotate_da3.py: chiavi, tipi e percorso non cambiano."""
    mask_dir = os.path.join(OUT_DIR, "masks", clip_id)
    ovl_dir = os.path.join(OUT_DIR, "overlay", clip_id)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(ovl_dir, exist_ok=True)
    empty = np.zeros((image.height, image.width), dtype=bool)
    np.savez_compressed(
        os.path.join(mask_dir, "%03d.npz" % frame),
        oggetto=obj["mask"] if obj else empty,
        supporto=sup["mask"] if sup else empty,
        trovato=obj is not None,
        supporto_trovato=sup is not None,
        box_oggetto=obj["box"] if obj else np.zeros(4, np.float32))
    save_overlay(image, obj, sup, cls, os.path.join(ovl_dir, "%03d.jpg" % frame))


def make_row(clip_id: str, cls: str, frame: int, obj: dict | None, sup: dict | None,
             n_obj: int, n_sup: int, diag: dict) -> dict:
    obj_prompt, sup_prompt = TEXT_PROMPTS[cls]
    row = {
        "clip_id": clip_id, "classe": cls, "frame": frame,
        "prompt_oggetto": obj_prompt, "trovato": obj is not None,
        "score": round(obj["score"], 4) if obj else "",
        "n_detections": n_obj,
        "area_px": int(obj["mask"].sum()) if obj else 0,
        "box_xyxy": " ".join("%.1f" % v for v in obj["box"]) if obj else "",
        "prompt_supporto": sup_prompt or "",
        "supporto_trovato": (sup is not None) if sup_prompt else "",
        "score_supporto": round(sup["score"], 4) if sup else "",
        "n_detections_supporto": n_sup if sup_prompt else "",
    }
    row.update({k: diag.get(k, "") for k in DIAG_KEYS})
    return row


def annotate_frame(processor, image: Image.Image, cls: str, cand_path: str,
                   clips_dir: str) -> tuple:
    """(obj, sup, n_obj, n_sup, diag, finito). Il monopattino resta sulla strada di
    prima del 28/09: stesso prompt sullo state appena creato, stessa best_detection."""
    if TEXT_PROMPTS[cls][1]:
        shape = (image.height, image.width)
        gruppi, scarti = detect_groups(processor, image, cls)
        save_candidates(cand_path, gruppi, shape, cls, clips_dir)
        obj, sup, diag = select_frame(gruppi, scarti, cls, shape)
        return (obj, sup, len(gruppi["oggetto"]), len(gruppi["supporto"]), diag,
                scarti["n_scartati_non_finiti"] == 0)
    with torch.autocast("cuda", dtype=DTYPE):
        state = processor.set_image(image)
        obj, n_obj, finite = best_detection(
            processor.set_text_prompt(state=state, prompt=TEXT_PROMPTS[cls][0]))
    # una box NaN arriverebbe a ground_band in annotate_da3.py e lo farebbe fallire:
    # il frame diventa "non trovato". Con box finite non cambia nulla, bit per bit.
    if obj is not None and not np.isfinite(obj["box"]).all():
        obj, finite = None, False
    return obj, None, n_obj, 0, {}, finite


def evaluate(clips_dir, limit):
    clips = load_clips(clips_dir)[:limit] if limit else load_clips(clips_dir)
    if not clips:
        sys.exit("nessuna clip con frame in %s" % clips_dir)

    gpu.check_vram_free(VRAM_FREE_MIN_GB)

    model, checkpoint_path, precision_mode = build_model()
    n_missing, n_mismatched = report_checkpoint_coverage(model, checkpoint_path)
    if n_missing or n_mismatched:
        sys.exit("checkpoint incompleto per questo modello: %d chiavi mancanti, %d shape"
                 " discordi. I pesi non coperti sono rimasti casuali." % (n_missing, n_mismatched))
    processor = Sam3Processor(model)

    # Sondaggio sul primo frame, come in eval_on_carla.py: il ripiego su autocast
    # si scopre qui invece che a meta' run. set_image e set_text_prompt sono gia'
    # decorati @torch.inference_mode() upstream, non serve avvolgerli di nuovo.
    if precision_mode == "half+autocast":
        rec0, pngs0 = clips[0]
        try:
            with torch.autocast("cuda", dtype=DTYPE):
                st = processor.set_image(Image.open(pngs0[0]).convert("RGB"))
                processor.set_text_prompt(state=st, prompt=TEXT_PROMPTS[rec0["classe"]][0])
        except RuntimeError as exc:
            print("half+autocast non regge su questo modello (%s): si prosegue con il"
                  " solo autocast" % exc, file=sys.stderr)
            model = model.float()
            processor = Sam3Processor(model)
            precision_mode = "autocast"
    print("precisione:        %s / %s" % (precision_mode, DTYPE))
    print("clip:              %d, frame: %d" % (len(clips), sum(len(p) for _, p in clips)))

    torch.cuda.reset_peak_memory_stats()
    rows = []
    contaminated = 0
    non_finite = 0
    started = time.time()
    n = 0

    for rec, pngs in clips:
        cls = rec["classe"]
        cand_dir = os.path.join(OUT_DIR, "candidati", rec["clip_id"])
        if TEXT_PROMPTS[cls][1]:
            os.makedirs(cand_dir, exist_ok=True)

        for png in pngs:
            frame = int(os.path.splitext(os.path.basename(png))[0])
            image = Image.open(png).convert("RGB")
            obj, sup, n_obj, n_sup, diag, finite = annotate_frame(
                processor, image, cls, os.path.join(cand_dir, "%03d.npz" % frame), clips_dir)
            # fp16 fallisce producendo NaN, non eccezioni: il frame va contato
            if not finite:
                non_finite += 1
            write_frame(rec["clip_id"], cls, frame, image, obj, sup)
            rows.append(make_row(rec["clip_id"], cls, frame, obj, sup, n_obj, n_sup, diag))

            peak = torch.cuda.max_memory_allocated() / 1e9
            if peak > VRAM_BUDGET_GB:
                sys.exit("picco VRAM %.2f GB oltre il budget di %.1f GB al frame %d:"
                         " il run si ferma invece di proseguire in sysmem fallback."
                         % (peak, VRAM_BUDGET_GB, n))
            if n % TELEMETRY_EVERY == 0:
                tel = gpu.gpu_telemetry()
                if gpu.is_throttling(tel, TEMP_LIMIT_C):
                    contaminated += 1
                    print("  frame %d: throttling (temp=%s, mask=%s)"
                          % (n, tel["gpu_temp_c"], tel["throttle_mask"]), file=sys.stderr)
            n += 1
            if n % 20 == 0:
                print("  %d frame" % n, flush=True)

    elapsed = time.time() - started
    peak = torch.cuda.max_memory_allocated() / 1e9
    return rows, elapsed, peak, contaminated, non_finite, precision_mode


def from_csv(r: dict) -> dict:
    """Una riga del CSV riletta con i tipi che summarize() si aspetta: "False"
    come stringa sarebbe vera."""
    out = dict(r)
    out["trovato"] = r["trovato"] == "True"
    out["score"] = float(r["score"]) if r["score"] else ""
    if r["supporto_trovato"] != "":
        out["supporto_trovato"] = r["supporto_trovato"] == "True"
    for k in DIAG_KEYS:
        out.setdefault(k, "")
    return out


def reselect(clips_dir: str) -> list[dict]:
    """Rifa' la selezione dei sospesi dai candidati salvati, senza caricare SAM.
    Le righe del monopattino e i suoi .npz non si toccano. Il CSV riletto e' il
    piu' recente di --out-dir: prompt e cartella delle clip si verificano sui
    candidati in load_candidates()."""
    found = sorted(glob.glob(os.path.join(OUT_DIR, "sam_generated_*.csv")))
    if not found:
        sys.exit("nessun sam_generated_*.csv in %s: prima un run con SAM" % OUT_DIR)
    with open(found[-1], encoding="utf-8") as f:
        old = list(csv.DictReader(f))
    print("riselezione da:    %s" % found[-1])
    rows = []
    for r in old:
        cls, frame = r["classe"], int(r["frame"])
        if cls not in TEXT_PROMPTS:
            sys.exit("%s: classe %r senza prompt in TEXT_PROMPTS" % (found[-1], cls))
        if not TEXT_PROMPTS[cls][1]:
            rows.append(from_csv(r))
            continue
        path = os.path.join(OUT_DIR, "candidati", r["clip_id"], "%03d.npz" % frame)
        if not os.path.exists(path):
            sys.exit("candidati mancanti: %s. Il run e' precedente al salvataggio dei"
                     " candidati: va rifatto con SAM." % path)
        gruppi = load_candidates(path, cls, clips_dir)
        png = os.path.join(clips_dir, "frames", r["clip_id"], "%03d.png" % frame)
        image = Image.open(png).convert("RGB")
        scarti = {k: int(r.get(k) or 0) for k in ("n_scartati_non_finiti", "n_maschere_vuote")}
        obj, sup, diag = select_frame(gruppi, scarti, cls, (image.height, image.width))
        write_frame(r["clip_id"], cls, frame, image, obj, sup)
        rows.append(make_row(r["clip_id"], cls, frame, obj, sup, len(gruppi["oggetto"]),
                             len(gruppi["supporto"]), diag))
    return rows


def write_csv(rows: list[dict]) -> str:
    out_path = os.path.join(OUT_DIR, "sam_generated_%s.csv"
                            % datetime.now().strftime("%Y%m%d_%H%M%S"))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return out_path


def summarize(rows):
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["classe"]].append(r)
    print("\ntasso di individuazione per frame, prompt testuale, soglia di confidenza 0,5:")
    # "supp contatto" e "supp vicino" separati: solo il primo e' un supporto che
    # tocca l'oggetto a un estremo, il secondo sta a meno di 40 px e basta.
    print("  %-28s %6s %8s %9s %10s %14s %12s"
          % ("classe", "frame", "trovato", "tasso", "score med", "supp contatto",
             "supp vicino"))
    for cls in TEXT_PROMPTS:
        rc = by_class.get(cls)
        if not rc:
            continue
        found = [r for r in rc if r["trovato"]]
        score = float(np.median([r["score"] for r in found])) if found else float("nan")
        contatto = vicino = "-"
        if TEXT_PROMPTS[cls][1]:
            for crit in ("contatto_estremo", "vicino"):
                quota = "%.1f%%" % (100.0 * sum(1 for r in rc if r["supporto_trovato"]
                                                 and r["criterio_supporto"] == crit) / len(rc))
                if crit == "vicino":
                    vicino = quota
                else:
                    contatto = quota
        print("  %-28s %6d %8d %8.1f%% %10.3f %14s %12s"
              % (cls, len(rc), len(found), 100.0 * len(found) / len(rc), score,
                 contatto, vicino))
    print("  riferimento: bicicletta CARLA in modalita' testo %.1f%%" % CARLA_TEXT_RATE_BICICLETTA)


def main():
    parser = argparse.ArgumentParser(
        description="SAM 3 sui frame generati da Wan (passo 0.2 esteso).")
    parser.add_argument("--clips-dir", default=os.path.join("..", "wan2_2", "outputs", "test_clips"),
                        help="cartella con manifest.jsonl e frames/; sul pod /workspace/out/<modello>")
    parser.add_argument("--out-dir", default=os.path.join("outputs", "generated_eval"),
                        help="dove finiscono maschere, overlay e CSV")
    parser.add_argument("--limit", type=int, default=None, help="solo le prime N clip")
    parser.add_argument("--riseleziona", action="store_true",
                        help="rifa' la selezione dei sospesi dai candidati in --out-dir,"
                             " senza caricare SAM")
    parser.add_argument("--vram-budget-gb", type=float, default=None,
                        help="stop se il picco supera questa soglia (default: frazione %.2f"
                             " della VRAM della scheda)" % gpu.VRAM_BUDGET_FRACTION)
    parser.add_argument("--vram-free-min-gb", type=float, default=gpu.VRAM_FREE_MIN_GB)
    parser.add_argument("--temp-limit-c", type=int, default=gpu.TEMP_LIMIT_C,
                        help="sopra questa temperatura la misura e' marcata contaminata")
    args = parser.parse_args()

    global OUT_DIR, VRAM_BUDGET_GB, VRAM_FREE_MIN_GB, TEMP_LIMIT_C
    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.riseleziona:
        rows = reselect(args.clips_dir)
        out_path = write_csv(rows)
        print("\n--- sintesi (riselezione, SAM non caricato) ---")
        print("frame:             %d" % len(rows))
        print("risultati:         %s" % out_path)
        summarize(rows)
        return

    VRAM_FREE_MIN_GB = args.vram_free_min_gb
    TEMP_LIMIT_C = args.temp_limit_c
    VRAM_BUDGET_GB = args.vram_budget_gb or gpu.vram_budget_gb()
    print("soglie: picco max %.1f GB, libera minima %.1f GB, temperatura %d C"
          % (VRAM_BUDGET_GB, VRAM_FREE_MIN_GB, TEMP_LIMIT_C))

    rows, elapsed, peak, contaminated, non_finite, precision_mode = evaluate(
        args.clips_dir, args.limit)
    out_path = write_csv(rows)

    print("\n--- sintesi ---")
    print("precisione:        %s" % precision_mode)
    print("frame:             %d" % len(rows))
    print("risultati:         %s" % out_path)
    print("maschere/overlay:  %s" % os.path.join(OUT_DIR, "masks|overlay"))
    print("picco VRAM:        %.2f GB su un budget di %.1f" % (peak, VRAM_BUDGET_GB))
    print("frame con output non finito (fp16): %d" % non_finite)
    print("misure contaminate da throttling: %d" % contaminated)
    print("tempo totale:      %.1f s" % elapsed)
    summarize(rows)
    print("\nNOTA: il 54,5% CARLA e' calcolato PER ISTANZA, questo PER FRAME. Nelle clip"
          " generate c'e' un solo oggetto bersaglio per frame, quindi le due misure"
          " coincidono in pratica; nel set CARLA 12 bici su 33 erano occluse dal ciclista.")
    print("NOTA: il processore ridimensiona a 1008x1008 senza preservare l'aspect ratio:"
          " 1280x704 viene compresso in orizzontale (x0,79) e stirato in verticale (x1,43),"
          " deformazione piu' forte del 4:3 di CARLA. Le maschere tornano alla risoluzione"
          " originale, ma la deformazione il modello la subisce.")


if __name__ == "__main__":
    main()
