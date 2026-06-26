"""
boss_eval_utils.py — modulo condiviso di valutazione B.O.S.S. OR4

Logica comune ai notebook di test (YOLO11, EfficientDet, NanoDet-Plus, PicoDet)
estratta in un unico punto per garantire confronto a piu modelli omogeneo per
costruzione ed evitare drift tra notebook:

  - mappa classi universale BOSS <-> alias <-> COCO 80
  - soglie identiche (CONF_THRESHOLD, IOU_THRESHOLD, MATCH_IOU)
  - preprocessing uniforme (letterbox, aspect-ratio preservato + padding)
  - greedy matching TP/FP/FN per classe a IoU = MATCH_IOU
  - generazione report markdown con titolo PARAMETRICO sul nome modello reale
  - contatore uniforme GFLOPs/GMACs/parametri su grafo ONNX

Nessuna dipendenza da variabili globali dei notebook: import puro.
"""

import re
import numpy as np
import pandas as pd

try:
    import cv2
except Exception:  # cv2 non necessario per le funzioni di sola valutazione/report
    cv2 = None


# ============================================================
# Soglie identiche per tutti i notebook (fix metodologico C)
# ============================================================
CONF_THRESHOLD = 0.25   # soglia confidenza inferenza/predizioni
IOU_THRESHOLD  = 0.45   # soglia IoU per NMS
MATCH_IOU      = 0.50   # soglia IoU per il matching TP/FP/FN greedy


# ============================================================
# Classi BOSS canoniche + alias
# ============================================================
BOSS_CLASSES = [
    "Bench", "Bicycle Rack", "Bike", "Car", "Chair", "Dustbin",
    "Electrical Box", "Electrical Pole", "Manhole", "Motorcycle",
    "Pedestrian crosswalk", "Person", "Plant Pot", "Road", "Stairs",
    "Table", "Teraffic Barrel", "Traffic sign", "Tree", "Truck",
]
NUM_CLASSES = len(BOSS_CLASSES)

BOSS_ALIASES = {
    "Bench":                ["bench"],
    "Bicycle Rack":         ["bicycle rack", "bike rack", "cycle rack"],
    "Bike":                 ["bike", "bicycle", "cycle"],
    "Car":                  ["car", "automobile"],
    "Chair":                ["chair"],
    "Dustbin":              ["dustbin", "bin", "trash can", "trashcan", "garbage can", "waste bin", "trash"],
    "Electrical Box":       ["electrical box", "electric box", "junction box", "utility box"],
    "Electrical Pole":      ["electrical pole", "electric pole", "utility pole", "power pole", "pole"],
    "Manhole":              ["manhole", "manhole cover"],
    "Motorcycle":           ["motorcycle", "motorbike", "motor bike"],
    "Pedestrian crosswalk": ["pedestrian crosswalk", "crosswalk", "cross walk", "zebra crossing", "pedestrian crossing"],
    "Person":               ["person", "pedestrian", "people", "human"],
    "Plant Pot":            ["plant pot", "potted plant", "pot plant", "flower pot", "flowerpot", "planter"],
    "Road":                 ["road", "street", "roadway"],
    "Stairs":               ["stairs", "staircase", "steps", "stair"],
    "Table":                ["table", "dining table", "desk"],
    "Teraffic Barrel":      ["teraffic barrel", "traffic barrel", "barrel", "traffic drum", "construction barrel"],
    "Traffic sign":         ["traffic sign", "road sign", "street sign", "stop sign", "traffic signal"],
    "Tree":                 ["tree"],
    "Truck":                ["truck", "lorry"],
}


def normalize_name(name):
    return re.sub(r"[\s_\-]+", " ", str(name).strip().lower())


ALIAS_TO_BOSS = {}
for _boss in BOSS_CLASSES:
    ALIAS_TO_BOSS[normalize_name(_boss)] = _boss
    for _alias in BOSS_ALIASES.get(_boss, []):
        ALIAS_TO_BOSS[normalize_name(_alias)] = _boss


def resolve_to_boss(name):
    """Nome (qualsiasi alias/grafia) -> classe BOSS canonica, o None."""
    return ALIAS_TO_BOSS.get(normalize_name(name))


def build_model_to_boss(model_names):
    """{model_id: nome} -> {model_id: boss_index} per le sole classi mappabili."""
    mapping = {}
    for mid, mname in model_names.items():
        boss = resolve_to_boss(mname)
        if boss is not None:
            mapping[int(mid)] = BOSS_CLASSES.index(boss)
    return mapping


# ============================================================
# Spazio classi COCO 80 (NanoDet-Plus, PicoDet, EfficientDet)
# ============================================================
# COCO id 1-based con gap (mancano 12, 26, 29, 30, ...). COCO_ID_TO_SEQ converte
# coco_id 1-based -> indice 0-based sequenziale 0..79: e' l'ordine standard a 80
# classi usato da NanoDet-Plus e PicoDet (class_id del modello = seq_id diretto).
_COCO_RAW = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup",
    48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana",
    53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot",
    58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair",
    63: "couch", 64: "potted plant", 65: "bed", 67: "dining table",
    70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote",
    76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven",
    80: "toaster", 81: "sink", 82: "refrigerator", 84: "book",
    85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
    89: "hair drier", 90: "toothbrush",
}

_coco_keys_sorted = sorted(_COCO_RAW.keys())
COCO_ID_TO_SEQ = {cid: idx for idx, cid in enumerate(_coco_keys_sorted)}  # 1-based -> 0-based
SEQ_TO_COCO_ID = {v: k for k, v in COCO_ID_TO_SEQ.items()}                # 0-based -> 1-based


def build_coco_model_classes():
    """Ritorna (MODEL_CLASSES, MODEL_NC) per i detector COCO 80 seq."""
    model_classes = {COCO_ID_TO_SEQ[k]: v for k, v in _COCO_RAW.items()}
    return model_classes, len(model_classes)


# ============================================================
# Preprocessing uniforme: letterbox (fix metodologico B)
# ============================================================
def letterbox(img_bgr, new_shape, color=(114, 114, 114)):
    """
    Resize con aspect-ratio preservato + padding (no stretch-resize).

    Ritorna (padded, ratio, (pad_left, pad_top)):
      - padded: immagine new_shape (H, W)
      - ratio:  fattore di scala applicato all'immagine originale
      - pad:    offset (left, top) in pixel del contenuto dentro padded

    Inversa per riportare una box dallo spazio rete a quello originale:
      x_orig = (x_net - pad_left) / ratio
      y_orig = (y_net - pad_top)  / ratio
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) non disponibile: letterbox richiede cv2.")
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    h0, w0 = img_bgr.shape[:2]
    r = min(new_shape[0] / h0, new_shape[1] / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))  # (w, h)
    dw = (new_shape[1] - new_unpad[0]) / 2.0
    dh = (new_shape[0] - new_unpad[1]) / 2.0
    resized = cv2.resize(img_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
    top    = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1))
    right  = int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)


def scale_boxes_from_letterbox(boxes_xyxy, ratio, pad, orig_hw):
    """
    Riporta box [N,4] xyxy dallo spazio rete (letterbox) all'immagine originale,
    con clipping ai bordi.
    """
    if len(boxes_xyxy) == 0:
        return boxes_xyxy
    left, top = pad
    h0, w0 = orig_hw
    out = boxes_xyxy.copy().astype(np.float64)
    out[:, [0, 2]] = (out[:, [0, 2]] - left) / ratio
    out[:, [1, 3]] = (out[:, [1, 3]] - top) / ratio
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w0)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h0)
    return out


# ============================================================
# IoU + greedy matching TP/FP/FN (fix metodologico C)
# ============================================================
def iou_one_to_many(box, boxes):
    """IoU tra una box [x1,y1,x2,y2] e un array [M,4]."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float64)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    iw = np.clip(x2 - x1, 0, None)
    ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    area_b = (box[2] - box[0]) * (box[3] - box[1])
    area_o = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_b + area_o - inter + 1e-9)


def accumulate_matches(tp_pc, fp_pc, fn_pc,
                       pred_boxes, pred_labels, pred_scores,
                       gt_boxes, gt_labels, match_iou=MATCH_IOU):
    """
    Greedy matching per classe (predizioni per confidenza decrescente) a IoU
    >= match_iou. Aggiorna in place gli accumulatori per classe tp_pc/fp_pc/fn_pc
    (array indicizzati per class_id). Identico in tutti i notebook.
    """
    pred_boxes  = np.asarray(pred_boxes,  dtype=np.float64).reshape(-1, 4)
    pred_labels = np.asarray(pred_labels, dtype=np.int64).reshape(-1)
    pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
    gt_boxes    = np.asarray(gt_boxes,    dtype=np.float64).reshape(-1, 4)
    gt_labels   = np.asarray(gt_labels,   dtype=np.int64).reshape(-1)

    if len(pred_labels) == 0 and len(gt_labels) == 0:
        return
    all_cls = np.unique(np.concatenate([pred_labels, gt_labels]))
    for cls in all_cls:
        cls = int(cls)
        pb = pred_boxes[pred_labels == cls]
        ps = pred_scores[pred_labels == cls]
        gb = gt_boxes[gt_labels == cls]
        matched = np.zeros(len(gb), dtype=bool)
        for pi in np.argsort(-ps):
            ious = iou_one_to_many(pb[pi], gb)
            if len(ious):
                ious = ious.copy()
                ious[matched] = -1.0
                best = int(ious.argmax())
                if ious[best] >= match_iou:
                    matched[best] = True
                    tp_pc[cls] += 1
                    continue
            fp_pc[cls] += 1
        fn_pc[cls] += int((~matched).sum())


def precision_recall_per_class(tp_pc, fp_pc, fn_pc):
    """Da accumulatori TP/FP/FN -> (precision_pc, recall_pc) per classe."""
    tp_pc = np.asarray(tp_pc, dtype=np.float64)
    fp_pc = np.asarray(fp_pc, dtype=np.float64)
    fn_pc = np.asarray(fn_pc, dtype=np.float64)
    denom_p = tp_pc + fp_pc
    denom_r = tp_pc + fn_pc
    prec = np.divide(tp_pc, denom_p, out=np.zeros_like(denom_p), where=denom_p > 0)
    rec  = np.divide(tp_pc, denom_r, out=np.zeros_like(denom_r), where=denom_r > 0)
    return prec, rec


# ============================================================
# Report markdown — titolo PARAMETRICO, nota metodologica corretta
# ============================================================
def df_to_md(df):
    """DataFrame -> tabella Markdown, senza dipendenze esterne."""
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep  = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = [f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + rows)


def build_report(model_name, config, metrics, df_metrics,
                 class_distribution=None, pred_total=0, frames_with_pred=0,
                 n_frames=0):
    """
    Costruisce il report markdown.

    - model_name: nome modello REALE (titolo parametrico — niente hardcode).
    - config: dict {label: valore} per la sezione Configurazione.
    - metrics: dict con chiavi map50, map5095, precision, recall, f1, match_iou,
               conf_threshold.
    - df_metrics: DataFrame metriche per classe.
    - class_distribution: pd.Series {classe BOSS: conteggio} oppure None.

    Nota metodologica corretta: Precision/Recall reali via TP/FP/FN greedy
    matching (non proxy AP/MAR di torchmetrics).
    """
    cfg_md = "\n".join(f"| {k} | {v} |" for k, v in config.items())

    if class_distribution is not None and len(class_distribution) > 0:
        dist_df = pd.DataFrame({
            "Classe": list(class_distribution.index),
            "Rilevamenti": list(class_distribution.values),
        })
        dist_md = df_to_md(dist_df)
    else:
        dist_md = "_Nessuna predizione._"

    note = (
        f"_Precision e Recall calcolate via greedy matching IoU (TP/FP/FN) a "
        f"IoU = {metrics['match_iou']} e soglia confidenza = {metrics['conf_threshold']}, "
        f"mediate sulle classi BOSS coperte dal modello._"
    )

    report = f"""# B.O.S.S. — Report Test {model_name}

## 1. Configurazione
| Parametro | Valore |
| --- | --- |
{cfg_md}

## 2. Metriche aggregate vs Ground Truth
{note}
| Metrica | Valore |
| --- | --- |
| mAP@0.5 | {metrics['map50']:.4f} |
| mAP@0.5:0.95 | {metrics['map5095']:.4f} |
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| F1 Score | {metrics['f1']:.4f} |

## 3. Metriche per classe
{df_to_md(df_metrics)}

## 4. Distribuzione predizioni sui recordings
- Predizioni totali: {pred_total}
- Frame con almeno una predizione: {frames_with_pred} / {n_frames}

{dist_md}
"""
    return report


# ============================================================
# Contatore uniforme GFLOPs/GMACs/parametri su grafo ONNX (fix D)
# ============================================================
def count_onnx_flops_params(onnx_path, input_shape=None, input_name=None,
                            csv_tmp_path=None):
    """
    Conta MACs/parametri su grafo ONNX (contatore uniforme per tutti i notebook).
    Ritorna dict {params, gmacs, gflops, method}.

    GMACs = MACs / 1e9 ; GFLOPs = 2 * GMACs (1 MAC = 2 FLOP).

    - onnx_path: percorso del modello ONNX esportato alla risoluzione di inferenza.
    - input_shape / input_name: opzionali; necessari solo se l'ONNX ha shape di
      input dinamiche. Se l'export e' a shape fissa, lasciarli None.
    - csv_tmp_path: file CSV temporaneo per il profilo per-nodo (default accanto
      all'onnx).

    Prova prima onnx-tool (profilo per-nodo completo). Se onnx-tool non riesce a
    profilare il grafo (es. detector con NMS embedded o modelli quantizzati QDQ,
    che generano errori di shape inference), ripiega su un conteggio dei soli
    layer di calcolo (Conv/ConvTranspose/Gemm/MatMul) via shape inference ONNX:
    box-decode e NMS hanno MACs trascurabili e restano esclusi per costruzione.
    """
    onnx_path = str(onnx_path)
    try:
        res = _count_onnx_tool(onnx_path, input_shape, input_name, csv_tmp_path)
        res["method"] = "onnx_tool"
        return res
    except Exception as e_tool:
        try:
            res = _count_via_shape_inference(onnx_path, input_shape, input_name)
            res["method"] = "shape_inference"
            return res
        except Exception as e_fb:
            raise RuntimeError(
                f"onnx-tool fallito ({e_tool}); fallback shape-inference fallito ({e_fb})"
            )


def _count_onnx_tool(onnx_path, input_shape=None, input_name=None,
                     csv_tmp_path=None):
    """Profilo MACs/parametri per-nodo con onnx-tool."""
    import onnx_tool
    from pathlib import Path

    onnx_path = str(onnx_path)
    if csv_tmp_path is None:
        csv_tmp_path = str(Path(onnx_path).with_suffix(".profile.csv"))

    inputs = None
    if input_shape is not None and input_name is not None:
        from onnx_tool import create_ndarray_f32
        inputs = {input_name: create_ndarray_f32(tuple(input_shape))}

    # model_profile stampa il profilo e (save_profile) salva il dettaglio per-nodo.
    onnx_tool.model_profile(onnx_path, inputs, save_profile=csv_tmp_path)

    df = pd.read_csv(csv_tmp_path)
    cols = {c.strip().lower(): c for c in df.columns}
    # onnx-tool recente nomina la colonna 'Forward_MACs'; match robusto per versione.
    mac_col = next((orig for low, orig in cols.items()
                    if "mac" in low and "back" not in low), None)
    par_col = next((orig for low, orig in cols.items() if "param" in low), None) \
        or cols.get("weight")
    name_col = df.columns[0]
    if mac_col is None or par_col is None:
        raise RuntimeError(f"Colonne MACs/Params non trovate nel CSV onnx-tool: {list(df.columns)}")

    def _num(x):
        s = re.sub(r"[^0-9]", "", str(x))
        return int(s) if s else 0

    # Esclude eventuale riga 'Total' per non raddoppiare: si somma il per-nodo.
    mask = df[name_col].astype(str).str.strip().str.lower() != "total"
    total_macs   = int(df.loc[mask, mac_col].map(_num).sum())
    total_params = int(df.loc[mask, par_col].map(_num).sum())

    gmacs = total_macs / 1e9
    return {"params": total_params, "gmacs": gmacs, "gflops": 2.0 * gmacs}


def _count_via_shape_inference(onnx_path, input_shape=None, input_name=None):
    """
    Fallback al profilo onnx-tool: MACs/parametri dei soli layer di calcolo
    (Conv/ConvTranspose/Gemm/MatMul) via shape inference ONNX. Pensato per grafi
    che onnx-tool non profila (detector con NMS embedded, modelli quantizzati
    QDQ). prod(weight_shape) e' invariante a permutazioni, quindi eventuali
    Transpose/Dequantize sul percorso dei pesi non alterano il conteggio.
    """
    import onnx
    from onnx import shape_inference

    model = onnx.load(str(onnx_path))
    graph = model.graph

    # Batch dinamico -> shape concreta sull'input, per abilitare l'inferenza.
    if input_shape is not None and input_name is not None:
        for inp in graph.input:
            if inp.name == input_name:
                dim = inp.type.tensor_type.shape.dim
                del dim[:]
                for d in input_shape:
                    dim.add().dim_value = int(d)

    inferred = shape_inference.infer_shapes(model, strict_mode=False,
                                            data_prop=True)

    shapes = {}
    for vi in (list(inferred.graph.value_info)
               + list(inferred.graph.input)
               + list(inferred.graph.output)):
        shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]

    init_shape = {init.name: list(init.dims) for init in graph.initializer}
    producer = {o: n for n in graph.node for o in n.output}

    def _prod(xs):
        p = 1
        for x in xs:
            p *= int(x)
        return p

    def _resolve_shape(name, _depth=0):
        """Shape di un tensore, risalendo Dequantize/Quantize/Cast/Transpose."""
        s = shapes.get(name)
        if s and 0 not in s:
            return s
        if name in init_shape:
            return init_shape[name]
        n = producer.get(name)
        if n is not None and _depth < 8 and n.op_type in (
                "DequantizeLinear", "QuantizeLinear", "Cast", "Transpose"):
            return _resolve_shape(n.input[0], _depth + 1)
        return s

    total_macs = 0
    total_params = 0
    for node in graph.node:
        if node.op_type in ("Conv", "ConvTranspose"):
            w = _resolve_shape(node.input[1])
            out = shapes.get(node.output[0])
            if not w or not out or len(out) < 3 or 0 in w:
                continue
            spatial_out = out[2:]
            if not spatial_out or 0 in spatial_out:
                continue
            total_macs += _prod(spatial_out) * _prod(w)
            total_params += _prod(w)
        elif node.op_type in ("Gemm", "MatMul"):
            b = _resolve_shape(node.input[1])
            out = shapes.get(node.output[0])
            if not b or not out or len(b) < 2 or 0 in out or 0 in b:
                continue
            if node.op_type == "MatMul":
                k = b[-2]
            else:  # Gemm: B e' [K,N] o [N,K]; K = dim di B diverso da N = out[-1]
                k = b[0] if b[1] == out[-1] else b[1]
            total_macs += _prod(out) * int(k)
            total_params += _prod(b)

    if total_macs == 0:
        raise RuntimeError("shape inference: nessun layer Conv/Gemm conteggiato "
                           "(shape non risolte)")

    gmacs = total_macs / 1e9
    return {"params": int(total_params), "gmacs": gmacs, "gflops": 2.0 * gmacs}
