# eval_on_carla.py
#
# Fase B.1: misura SAM contro la ground truth di istanza di CARLA.
#
# Legge il manifest eval_set.json, da al modello SOLO il flusso rgb, e confronta
# le maschere prodotte con quelle esatte del simulatore.
#
# Due modalita' di prompt, e non sono intercambiabili:
#
#   box    prompt geometrico dai box gia' esportati. Misura la qualita' della
#          maschera DATO un box corretto, cioe' la segmentazione pura. E' l'unica
#          valida per gli ostacoli sospesi: il ramo_sporgente e' una barriera
#          stradale che fluttua a due metri, l'aspetto di ramo lo dipinge il
#          generatore in Fase D, quindi nessun prompt testuale "branch" lo
#          troverebbe mai - e sarebbe un difetto della prova, non del modello.
#
#   testo  prompt testuale, solo per le classi il cui aspetto corrisponde al nome.
#          Misura rilevamento e segmentazione insieme, che e' il modo in cui SAM
#          verra' usato davvero in Fase E, e produce anche il recall.
import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime

# La cartella di questo script contiene il clone del repo in sam3\ (senza
# __init__.py): come sys.path[0] farebbe ombra al package installato sam3,
# risolvendolo come namespace package con __file__ = None e rompendo il lookup
# delle risorse (assets/bpe_*). La si rimuove dal path.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "CARLA")))
import carla_gt  # noqa: E402

from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

# SAM 3 e non SAM 3.1: in questa release 3.1 esiste solo come modello video
# multiplex, e il suo checkpoint lascerebbe scoperto un livello FPN del modello
# immagine. Il perche' esteso, con i riferimenti al sorgente, e' in smoke_test.py.
VERSION = "sam3"

# fp16 e non bf16: occupano gli stessi byte, cambia il compromesso fra esponente
# e mantissa. fp16 puo' andare in overflow e in quel caso restituisce NaN senza
# eccezioni, per questo sotto c'e' un controllo esplicito sui valori non finiti.
DTYPE = torch.float16

VRAM_BUDGET_GB = 13.0
TEMP_LIMIT_C = 83
TELEMETRY_EVERY = 10

# Solo le classi il cui aspetto in CARLA corrisponde al nome. Gli ostacoli
# sospesi sono deliberatamente esclusi: la loro apparenza non e' quella della
# classe, e chiederla a parole misurerebbe la scenografia, non il modello.
TEXT_PROMPTS = {
    "pedone": "person",
    "automobile": "car",
    "bicicletta": "bicycle",
    "motociclo": "motorcycle",
    "palo_della_luce": "pole",
    "semaforo": "traffic light",
}

# Tetto di istanze per classe e per frame nella modalita' box. Senza, i pali
# della luce - 363 istanze su 614 nel set - si prenderebbero la maggior parte
# delle inferenze e la misura direbbe soprattutto quanto SAM segmenta i pali.
MAX_INSTANCES_PER_CLASS_PER_FRAME = 6


def gpu_telemetry():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,clocks_throttle_reasons.active,clocks.current.sm",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=10).strip().splitlines()[0]
        temp, throttle, sm = [p.strip() for p in out.split(",")]
        return {"gpu_temp_c": int(temp), "throttle_mask": throttle, "sm_clock_mhz": int(sm)}
    except Exception:
        return {"gpu_temp_c": None, "throttle_mask": None, "sm_clock_mhz": None}


def is_throttling(tel):
    if tel["throttle_mask"] is None:
        return False
    try:
        mask = int(tel["throttle_mask"], 16)
    except ValueError:
        return False
    hot = tel["gpu_temp_c"] is not None and tel["gpu_temp_c"] >= TEMP_LIMIT_C
    return mask != 0 or hot


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def box_iou(a, b):
    """IoU fra due box in formato xyxy."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def fill_ratio(inst):
    """Quanta parte del proprio box occupa la maschera dell'istanza.

    Serve a leggere gli IoU senza sbagliarsi. Misurato sul set: automobile 68%,
    gradino 81%, ostacolo_sospeso_generico 92% - ma palo_della_luce 4,3% e
    semaforo 5,2%, perche' il tag Pole di CARLA copre anche i cavi aerei e il tag
    TrafficLight i bracci dei semafori. Per quelle due classi un box e' per il 95%
    sfondo, quindi un prompt geometrico non identifica l'oggetto e uno IoU basso
    non e' un difetto del modello ma della domanda che gli si e' posta.
    """
    x, y, w, h = inst["bbox"]
    return round(inst["area_px"] / float(w * h), 4) if w * h else 0.0


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
    checkpoint_path = download_ckpt_from_hf(version=VERSION)
    model = build_sam3_image_model(checkpoint_path=checkpoint_path)

    # model.half() dimezza i pesi residenti; autocast da solo lascia i pesi in
    # fp32 e converte le sole attivazioni. Su questo modello la combinazione NON
    # regge - misurato: "mat1 and mat2 must have the same dtype, but got Float
    # and Half" - perche' gli operatori che autocast tiene deliberatamente in
    # fp32 si trovano pesi gia' convertiti. Si ripiega quindi su autocast da
    # solo, che con 4,24 GB di picco resta comunque un terzo del budget. Il
    # tentativo resta qui perche' su un modello diverso potrebbe passare, e in
    # quel caso il risparmio e' reale.
    mode = "half+autocast"
    try:
        model = model.half()
    except Exception as exc:
        print("model.half() non applicabile (%s): si resta su autocast"
              % type(exc).__name__, file=sys.stderr)
        mode = "autocast"
    return model, checkpoint_path, mode


def evaluate(manifest_path, limit):
    with open(manifest_path) as f:
        manifest = json.load(f)
    base = os.path.dirname(os.path.abspath(manifest_path))
    frames = manifest["frames"][:limit] if limit else manifest["frames"]

    model, checkpoint_path, precision_mode = build_model()
    processor = Sam3Processor(model)

    # Sondaggio su un frame prima del ciclo. model.half() puo' riuscire e poi far
    # fallire il forward - gli operatori che autocast tiene in fp32 si trovano
    # pesi gia' convertiti - e su questo modello e' quello che succede. Scoprirlo
    # al primo frame invece che a meta' run costa una inferenza; model.float()
    # riverte in memoria, senza ricaricare i 3,5 GB del checkpoint.
    if precision_mode == "half+autocast" and frames:
        probe = Image.fromarray(carla_gt.read_rgb(os.path.join(base, frames[0]["rgb"])))
        try:
            with torch.autocast("cuda", dtype=DTYPE):
                st = processor.set_image(probe)
                processor.set_text_prompt(state=st, prompt="car")
        except RuntimeError as exc:
            print("half+autocast non regge su questo modello (%s): si prosegue con il"
                  " solo autocast" % exc, file=sys.stderr)
            model = model.float()
            processor = Sam3Processor(model)
            precision_mode = "autocast"

    print("checkpoint:   %s" % os.path.basename(checkpoint_path))
    print("precisione:   %s / %s" % (precision_mode, DTYPE))
    print("manifest:     %s (%d frame)" % (manifest_path, len(frames)))

    torch.cuda.reset_peak_memory_stats()
    rows = []
    contaminated = 0
    non_finite_frames = 0
    started = time.time()

    for n, entry in enumerate(frames):
        rgb = carla_gt.read_rgb(os.path.join(base, entry["rgb"]))
        image = Image.fromarray(rgb)
        tag, ids = carla_gt.read_instance(os.path.join(base, entry["instance"]))
        H, W = entry["height"], entry["width"]

        with torch.autocast("cuda", dtype=DTYPE):
            state = processor.set_image(image)

            # --- modalita' box -------------------------------------------------
            per_class = defaultdict(int)
            for inst in entry["instances"]:
                cls = inst["boss_class"]
                if per_class[cls] >= MAX_INSTANCES_PER_CLASS_PER_FRAME:
                    continue
                per_class[cls] += 1

                gt_mask = carla_gt.instance_mask(tag, ids, inst["carla_tag"],
                                                 inst["instance_id"])
                if not gt_mask.any():
                    continue

                x, y, w, h = inst["bbox"]
                prompt_xyxy = (x, y, x + w, y + h)
                norm_cxcywh = [(x + w / 2.0) / W, (y + h / 2.0) / H, w / float(W), h / float(H)]

                processor.reset_all_prompts(state)
                out = processor.add_geometric_prompt(box=norm_cxcywh, label=True, state=state)
                masks = to_numpy_masks(out)
                boxes = out["boxes"].detach().float().cpu().numpy() if len(masks) else []

                iou, matched = 0.0, False
                if len(masks):
                    order = sorted(range(len(masks)),
                                   key=lambda i: -box_iou(prompt_xyxy, boxes[i]))
                    best = order[0]
                    if box_iou(prompt_xyxy, boxes[best]) > 0:
                        iou = mask_iou(masks[best], gt_mask)
                        matched = True

                rows.append({
                    "session": entry["session"], "frame_index": entry["frame_index"],
                    "prompt_mode": "box", "prompt": "geometrico",
                    "boss_class": cls, "source": inst["source"],
                    "instance_id": inst["instance_id"], "area_px": inst["area_px"],
                    "fill_ratio": fill_ratio(inst),
                    "truncated": inst["truncated"], "distance_m": inst["distance_m"],
                    "iou": round(iou, 4), "matched": matched, "n_detections": len(masks),
                })

            # --- modalita' testo ------------------------------------------------
            present = set(i["boss_class"] for i in entry["instances"])
            for cls, prompt in TEXT_PROMPTS.items():
                if cls not in present:
                    continue
                processor.reset_all_prompts(state)
                out = processor.set_text_prompt(state=state, prompt=prompt)
                masks = to_numpy_masks(out)
                used = set()

                gt_insts = [i for i in entry["instances"] if i["boss_class"] == cls]
                gt_insts = gt_insts[:MAX_INSTANCES_PER_CLASS_PER_FRAME]
                gt_masks = []
                for inst in gt_insts:
                    gm = carla_gt.instance_mask(tag, ids, inst["carla_tag"],
                                                inst["instance_id"])
                    if gm.any():
                        gt_masks.append((inst, gm))

                # Abbinamento greedy sulla IoU decrescente, non nell'ordine in cui
                # le istanze compaiono nel manifest: altrimenti la prima istanza
                # elencata si prende una predizione grande che apparteneva a
                # un'altra, e le due IoU risultano entrambe peggiori del vero.
                pairs = sorted(
                    ((mask_iou(masks[k], gm), k, n_gt)
                     for n_gt, (_, gm) in enumerate(gt_masks)
                     for k in range(len(masks))),
                    key=lambda t: -t[0])
                assigned = {}
                for v, k, n_gt in pairs:
                    if v <= 0 or k in used or n_gt in assigned:
                        continue
                    used.add(k)
                    assigned[n_gt] = v

                for n_gt, (inst, _) in enumerate(gt_masks):
                    iou = assigned.get(n_gt, 0.0)
                    best = n_gt in assigned
                    rows.append({
                        "session": entry["session"], "frame_index": entry["frame_index"],
                        "prompt_mode": "testo", "prompt": prompt,
                        "boss_class": cls, "source": inst["source"],
                        "instance_id": inst["instance_id"], "area_px": inst["area_px"],
                        "fill_ratio": fill_ratio(inst),
                        "truncated": inst["truncated"], "distance_m": inst["distance_m"],
                        "iou": round(iou, 4), "matched": best,
                        "n_detections": len(masks),
                    })

        peak = torch.cuda.max_memory_allocated() / 1e9
        if peak > VRAM_BUDGET_GB:
            sys.exit("picco VRAM %.2f GB oltre il budget di %.1f GB al frame %d:"
                     " il run si ferma invece di proseguire in sysmem fallback."
                     % (peak, VRAM_BUDGET_GB, n))

        if n % TELEMETRY_EVERY == 0:
            tel = gpu_telemetry()
            if is_throttling(tel):
                contaminated += 1
                print("  frame %d: throttling (temp=%s, mask=%s)"
                      % (n, tel["gpu_temp_c"], tel["throttle_mask"]), file=sys.stderr)
        if (n + 1) % 10 == 0:
            print("  %d/%d frame, %d misure" % (n + 1, len(frames), len(rows)), flush=True)

    elapsed = time.time() - started
    peak = torch.cuda.max_memory_allocated() / 1e9
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return rows, elapsed, peak, contaminated, non_finite_frames, precision_mode


def summarize(rows, mode):
    subset = [r for r in rows if r["prompt_mode"] == mode]
    if not subset:
        return
    print("\nprompt %s:" % mode)
    by_class = defaultdict(list)
    for r in subset:
        by_class[r["boss_class"]].append(r)
    print("  %-30s %5s %9s %9s %9s %9s"
          % ("classe", "n", "IoU med", "IoU>=0,7", "trovate", "riemp."))
    for cls in sorted(by_class, key=lambda c: -len(by_class[c])):
        v = [r["iou"] for r in by_class[cls]]
        found = sum(1 for r in by_class[cls] if r["matched"])
        fill = float(np.median([r["fill_ratio"] for r in by_class[cls]]))
        print("  %-30s %5d %9.3f %8.1f%% %8.1f%% %8.1f%%"
              % (cls, len(v), float(np.median(v)),
                 100.0 * np.mean([x >= 0.7 for x in v]), 100.0 * found / len(v),
                 100.0 * fill))
    allv = [r["iou"] for r in subset]
    print("  %-30s %5d %9.3f %8.1f%%"
          % ("TUTTE", len(allv), float(np.median(allv)),
             100.0 * np.mean([x >= 0.7 for x in allv])))


def main():
    parser = argparse.ArgumentParser(
        description="Misura SAM 3.1 contro la ground truth di istanza CARLA (Fase B.1).")
    parser.add_argument("--eval-set", default=os.path.join("..", "CARLA", "eval_set.json"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows, elapsed, peak, contaminated, non_finite, precision_mode = evaluate(
        args.eval_set, args.limit)

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "sam_eval_%s.csv"
                            % datetime.now().strftime("%Y%m%d_%H%M%S"))
    fields = sorted(set().union(*(set(r) for r in rows)))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\n--- sintesi ---")
    print("precisione:        %s" % precision_mode)
    print("misure:            %d" % len(rows))
    print("risultati:         %s" % out_path)
    print("picco VRAM:        %.2f GB su un budget di %.1f" % (peak, VRAM_BUDGET_GB))
    print("misure contaminate da throttling: %d" % contaminated)
    print("tempo totale:      %.1f s (registrato, NON usato come metrica di confronto)"
          % elapsed)
    print("\nNOTA: il processore ridimensiona a 1008x1008 senza preservare l'aspect"
          " ratio, quindi il 4:3 di CARLA viene schiacciato in quadrato. Le maschere"
          " tornano poi alle dimensioni originali e lo IoU e' calcolato nel sistema"
          " giusto, ma la deformazione il modello la subisce.")

    summarize(rows, "box")
    summarize(rows, "testo")


if __name__ == "__main__":
    main()
