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
import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

# gpu.py sta in questa stessa cartella, quindi va importato PRIMA della riga sotto.
import gpu

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
    "ramo_sporgente": ("tree branch", "tree trunk"),
    "insegna_cartello_basso": ("sign", "pole"),
    "ostacolo_sospeso_generico": ("horizontal bar", "pole"),
}

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
        obj_prompt, sup_prompt = TEXT_PROMPTS[cls]
        mask_dir = os.path.join(OUT_DIR, "masks", rec["clip_id"])
        ovl_dir = os.path.join(OUT_DIR, "overlay", rec["clip_id"])
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(ovl_dir, exist_ok=True)

        for png in pngs:
            frame = int(os.path.splitext(os.path.basename(png))[0])
            image = Image.open(png).convert("RGB")
            with torch.autocast("cuda", dtype=DTYPE):
                state = processor.set_image(image)
                obj, n_obj, fin_obj = best_detection(
                    processor.set_text_prompt(state=state, prompt=obj_prompt))
                sup, n_sup, fin_sup = None, 0, True
                if sup_prompt:
                    processor.reset_all_prompts(state)
                    sup, n_sup, fin_sup = best_detection(
                        processor.set_text_prompt(state=state, prompt=sup_prompt))
            # fp16 fallisce producendo NaN, non eccezioni: il frame va contato
            if not (fin_obj and fin_sup):
                non_finite += 1

            empty = np.zeros((image.height, image.width), dtype=bool)
            np.savez_compressed(
                os.path.join(mask_dir, "%03d.npz" % frame),
                oggetto=obj["mask"] if obj else empty,
                supporto=sup["mask"] if sup else empty,
                trovato=obj is not None,
                supporto_trovato=sup is not None,
                box_oggetto=obj["box"] if obj else np.zeros(4, np.float32))
            save_overlay(image, obj, sup, cls, os.path.join(ovl_dir, "%03d.jpg" % frame))

            rows.append({
                "clip_id": rec["clip_id"], "classe": cls, "frame": frame,
                "prompt_oggetto": obj_prompt, "trovato": obj is not None,
                "score": round(obj["score"], 4) if obj else "",
                "n_detections": n_obj,
                "area_px": int(obj["mask"].sum()) if obj else 0,
                "box_xyxy": " ".join("%.1f" % v for v in obj["box"]) if obj else "",
                "prompt_supporto": sup_prompt or "",
                "supporto_trovato": (sup is not None) if sup_prompt else "",
                "score_supporto": round(sup["score"], 4) if sup else "",
                "n_detections_supporto": n_sup if sup_prompt else "",
            })

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


def summarize(rows):
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["classe"]].append(r)
    print("\ntasso di individuazione per frame, prompt testuale, soglia di confidenza 0,5:")
    print("  %-28s %6s %8s %9s %10s %10s"
          % ("classe", "frame", "trovato", "tasso", "score med", "supporto"))
    for cls in TEXT_PROMPTS:
        rc = by_class.get(cls)
        if not rc:
            continue
        found = [r for r in rc if r["trovato"]]
        score = float(np.median([r["score"] for r in found])) if found else float("nan")
        sup = "-"
        if TEXT_PROMPTS[cls][1]:
            sup = "%.1f%%" % (100.0 * sum(1 for r in rc if r["supporto_trovato"]) / len(rc))
        print("  %-28s %6d %8d %8.1f%% %10.3f %10s"
              % (cls, len(rc), len(found), 100.0 * len(found) / len(rc), score, sup))
    print("  riferimento: bicicletta CARLA in modalita' testo %.1f%%" % CARLA_TEXT_RATE_BICICLETTA)


def main():
    parser = argparse.ArgumentParser(
        description="SAM 3 sui frame generati da Wan (passo 0.2 esteso).")
    parser.add_argument("--clips-dir", default=os.path.join("..", "wan2_2", "outputs", "test_clips"),
                        help="cartella con manifest.jsonl e frames/; sul pod /workspace/out/<modello>")
    parser.add_argument("--out-dir", default=os.path.join("outputs", "generated_eval"),
                        help="dove finiscono maschere, overlay e CSV")
    parser.add_argument("--limit", type=int, default=None, help="solo le prime N clip")
    parser.add_argument("--vram-budget-gb", type=float, default=None,
                        help="stop se il picco supera questa soglia (default: frazione %.2f"
                             " della VRAM della scheda)" % gpu.VRAM_BUDGET_FRACTION)
    parser.add_argument("--vram-free-min-gb", type=float, default=gpu.VRAM_FREE_MIN_GB)
    parser.add_argument("--temp-limit-c", type=int, default=gpu.TEMP_LIMIT_C,
                        help="sopra questa temperatura la misura e' marcata contaminata")
    args = parser.parse_args()

    global OUT_DIR, VRAM_BUDGET_GB, VRAM_FREE_MIN_GB, TEMP_LIMIT_C
    OUT_DIR = args.out_dir
    VRAM_FREE_MIN_GB = args.vram_free_min_gb
    TEMP_LIMIT_C = args.temp_limit_c
    VRAM_BUDGET_GB = args.vram_budget_gb or gpu.vram_budget_gb()
    os.makedirs(OUT_DIR, exist_ok=True)
    print("soglie: picco max %.1f GB, libera minima %.1f GB, temperatura %d C"
          % (VRAM_BUDGET_GB, VRAM_FREE_MIN_GB, TEMP_LIMIT_C))

    rows, elapsed, peak, contaminated, non_finite, precision_mode = evaluate(
        args.clips_dir, args.limit)

    out_path = os.path.join(OUT_DIR, "sam_generated_%s.csv"
                            % datetime.now().strftime("%Y%m%d_%H%M%S"))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

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
    print("\nNOTA: il 54,5%% CARLA e' calcolato PER ISTANZA, questo PER FRAME. Nelle clip"
          " generate c'e' un solo oggetto bersaglio per frame, quindi le due misure"
          " coincidono in pratica; nel set CARLA 12 bici su 33 erano occluse dal ciclista.")
    print("NOTA: il processore ridimensiona a 1008x1008 senza preservare l'aspect ratio:"
          " 1280x704 viene compresso in orizzontale (x0,79) e stirato in verticale (x1,43),"
          " deformazione piu' forte del 4:3 di CARLA. Le maschere tornano alla risoluzione"
          " originale, ma la deformazione il modello la subisce.")


if __name__ == "__main__":
    main()
