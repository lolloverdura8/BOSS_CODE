# eval_on_generated.py
#
# Passo 0.2 esteso: Depth Anything 3 sui frame generati da wan2_2, con le maschere
# prodotte da sam3_1/eval_on_generated.py.
#
# Qui non c'e' ground truth di profondita', quindi niente AbsRel ne' delta<1,25
# contro un riferimento. Si misura una COERENZA INTERNA che non dipende dalla
# scala ignota della depth relativa: il rapporto r fra la profondita' mediana
# dell'oggetto e quella di qualcosa che nella scena sta, per costruzione, alla
# stessa distanza. Moltiplicare la mappa per una costante non cambia r.
#
#   sospesi      riferimento = il supporto (tronco per il ramo, palo per insegna
#                e barra). I prompt di generazione fissano l'oggetto al supporto e
#                lo sviluppano di traverso al percorso: stessa profondita', r ~ 1.
#                Su CARLA DA3 collocava i sospesi 3,4-4,0 volte piu' lontani
#                (B.2, Tabella 5.2): e' quel sintomo che r intercetta.
#   monopattino  riferimento = la fascia di suolo subito sotto il box: poggia a
#                terra, quindi di nuovo r ~ 1.
#
# Il criterio max(r, 1/r) < 1,25 e' la stessa soglia del delta<1,25 di B.2,
# applicata a un rapporto fra mediane invece che pixel per pixel.
import argparse
import csv
import gc
import glob
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from PIL import Image

# Prima di importare depth_anything_3, come in eval_on_carla.py: fp16 e non bf16.
torch.cuda.is_bf16_supported = lambda *a, **k: False

from depth_anything_3.api import DepthAnything3  # noqa: E402
from depth_anything_3.utils.visualize import visualize_depth  # noqa: E402

DEFAULT_MODEL = "depth-anything/DA3-BASE"
DEFAULT_PROCESS_RES = 1008

VRAM_BUDGET_GB = 13.0
TEMP_LIMIT_C = 83
TELEMETRY_EVERY = 10
# DA3-BASE ha un picco misurato di 1,49 GB (B.2): sotto 6 GB liberi la scheda e'
# ancora occupata da qualcun altro.
VRAM_FREE_MIN_GB = 6.0

DELTA_THRESHOLD = 1.25
# Fascia di suolo sotto il box del monopattino, in pixel su 704 di altezza.
BAND_PX = 15
# Sotto questo numero di pixel validi una mediana non e' una misura: il frame
# diventa non valutabile invece di produrre un r casuale.
MIN_PX = 50

SUSPENDED = ("ramo_sporgente", "insegna_cartello_basso", "ostacolo_sospeso_generico")
CLASS_ORDER = ("monopattino",) + SUSPENDED

COLOR_OBJ = np.array((255, 0, 0), np.uint8)
COLOR_REF = np.array((0, 90, 255), np.uint8)

OUT_DIR = os.path.join("outputs", "generated_eval")


def gpu_telemetry():
    """Temperatura, throttling e clock, da nvidia-smi. Vuoto se non disponibile."""
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


def ground_band(box, shape):
    """Fascia di BAND_PX righe subito sotto il bordo inferiore del box, larga quanto il box."""
    H, W = shape
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    top = min(max(y1, 0), H)
    band = np.zeros(shape, dtype=bool)
    band[top:min(top + BAND_PX, H), max(x0, 0):min(x1, W)] = True
    return band


def outline(mask):
    """Contorno di una maschera, spesso 2 px, per disegnarlo sopra la depth."""
    inner = mask.copy()
    inner[1:, :] &= mask[:-1, :]
    inner[:-1, :] &= mask[1:, :]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    edge = mask & ~inner
    thick = edge.copy()
    thick[1:, :] |= edge[:-1, :]
    thick[:, 1:] |= edge[:, :-1]
    return thick


def save_depth_vis(rgb, depth, obj, ref, path):
    """RGB a sinistra, depth colorata a destra; contorni oggetto (rosso) e
    riferimento (blu) su entrambi, per giudicare a occhio dove stanno."""
    colored = visualize_depth(depth)
    panels = []
    for panel in (rgb.copy(), colored.copy()):
        panel[outline(ref)] = COLOR_REF
        panel[outline(obj)] = COLOR_OBJ
        panels.append(panel)
    Image.fromarray(np.concatenate(panels, axis=1)).save(path, quality=90)


def latest_sam_csv(sam_dir):
    found = sorted(glob.glob(os.path.join(sam_dir, "sam_generated_*.csv")))
    if not found:
        sys.exit("nessun sam_generated_*.csv in %s: prima sam3_1/eval_on_generated.py" % sam_dir)
    return found[-1]


def evaluate(clips_dir, sam_dir, sam_csv, model_name, process_res):
    with open(sam_csv, encoding="utf-8") as f:
        sam_rows = list(csv.DictReader(f))

    free, total = torch.cuda.mem_get_info()
    print("VRAM libera prima di caricare: %.2f GB su %.2f" % (free / 1e9, total / 1e9))
    if free / 1e9 < VRAM_FREE_MIN_GB:
        sys.exit("VRAM libera %.2f GB < %.1f GB: la scheda e' ancora occupata."
                 % (free / 1e9, VRAM_FREE_MIN_GB))

    print("modello:      %s" % model_name)
    print("maschere SAM: %s (%d frame)" % (sam_csv, len(sam_rows)))
    print("process_res:  %d" % process_res)

    model = DepthAnything3.from_pretrained(model_name).to(device=torch.device("cuda"))
    torch.cuda.reset_peak_memory_stats()

    rows, review = [], []
    contaminated = 0
    shape_reported = False
    started = time.time()

    for n, sr in enumerate(sam_rows):
        clip_id, cls, frame = sr["clip_id"], sr["classe"], int(sr["frame"])
        png = os.path.join(clips_dir, "frames", clip_id, "%03d.png" % frame)
        npz = os.path.join(sam_dir, "masks", clip_id, "%03d.npz" % frame)
        rgb = np.asarray(Image.open(png).convert("RGB"))
        H, W = rgb.shape[:2]

        with torch.inference_mode():
            prediction = model.inference([rgb], process_res=process_res)
        depth = np.asarray(prediction.depth[0], dtype=np.float32)
        if depth.shape != (H, W):
            # 1280x704 non e' multiplo di 14 come lo era 1008x756 di CARLA: il
            # processor ridimensiona, e la mappa va riportata ai pixel delle maschere
            if not shape_reported:
                print("depth %s != frame %s: riportata a %dx%d con interpolazione bilineare"
                      % (depth.shape, (H, W), W, H))
                shape_reported = True
            depth = np.asarray(Image.fromarray(depth, mode="F").resize((W, H), Image.BILINEAR))

        m = np.load(npz)
        obj = m["oggetto"]
        if cls == "monopattino":
            riferimento = "suolo"
            ref = ground_band(m["box_oggetto"], (H, W)) & ~obj if m["trovato"] else np.zeros_like(obj)
            ref_found = bool(m["trovato"])
        else:
            riferimento = "supporto"
            ref = m["supporto"] & ~obj
            ref_found = bool(m["supporto_trovato"])

        valid = np.isfinite(depth) & (depth > 0)
        obj_v, ref_v = obj & valid, ref & valid
        row = {"clip_id": clip_id, "classe": cls, "frame": frame, "riferimento": riferimento,
               "n_px_oggetto": int(obj_v.sum()), "n_px_riferimento": int(ref_v.sum()),
               "med_oggetto": "", "med_riferimento": "", "r": "", "coerente": "",
               "non_valutabile": False, "motivo": ""}

        if not m["trovato"]:
            row["non_valutabile"], row["motivo"] = True, "oggetto non trovato da SAM"
        elif not ref_found:
            row["non_valutabile"], row["motivo"] = True, "riferimento non trovato da SAM"
        elif obj_v.sum() < MIN_PX or ref_v.sum() < MIN_PX:
            row["non_valutabile"], row["motivo"] = True, "meno di %d px validi" % MIN_PX
        else:
            mo, mr = float(np.median(depth[obj_v])), float(np.median(depth[ref_v]))
            r = mo / mr
            row.update(med_oggetto=round(mo, 5), med_riferimento=round(mr, 5), r=round(r, 4),
                       coerente=bool(max(r, 1.0 / r) < DELTA_THRESHOLD))

        vis_dir = os.path.join(OUT_DIR, "depth_vis", clip_id)
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, "%03d.jpg" % frame)
        save_depth_vis(rgb, depth, obj, ref, vis_path)

        peak = torch.cuda.max_memory_allocated() / 1e9
        row["vram_peak_gb"] = round(peak, 3)
        if peak > VRAM_BUDGET_GB:
            sys.exit("picco VRAM %.2f GB oltre il budget di %.1f GB al frame %d:"
                     " il run si ferma invece di proseguire in sysmem fallback."
                     % (peak, VRAM_BUDGET_GB, n))
        if n % TELEMETRY_EVERY == 0:
            tel = gpu_telemetry()
            row.update(tel)
            row["contaminated"] = is_throttling(tel)
            if row["contaminated"]:
                contaminated += 1
                print("  frame %d: throttling (temp=%s, mask=%s)"
                      % (n, tel["gpu_temp_c"], tel["throttle_mask"]), file=sys.stderr)

        rows.append(row)
        review.append({
            "clip_id": clip_id, "classe": cls, "frame": frame,
            "overlay_sam": os.path.abspath(os.path.join(sam_dir, "overlay", clip_id,
                                                        "%03d.jpg" % frame)),
            "depth_vis": os.path.abspath(vis_path),
            "maschera_ok": "", "depth_ok": "", "note": ""})
        if (n + 1) % 20 == 0:
            print("  %d/%d frame" % (n + 1, len(sam_rows)), flush=True)

    elapsed = time.time() - started
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return rows, review, elapsed, contaminated


def write_csv(path, rows):
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarize(rows):
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["classe"]].append(r)
    print("\ncoerenza di profondita' oggetto/riferimento, criterio max(r, 1/r) < %.2f:"
          % DELTA_THRESHOLD)
    print("  %-28s %-9s %6s %11s %9s %9s"
          % ("classe", "rif.", "frame", "valutabili", "coerenti", "r med"))
    for cls in CLASS_ORDER:
        rc = by_class.get(cls)
        if not rc:
            continue
        ev = [r for r in rc if not r["non_valutabile"]]
        coh = "%.1f%%" % (100.0 * sum(1 for r in ev if r["coerente"]) / len(ev)) if ev else "-"
        rmed = "%.3f" % float(np.median([r["r"] for r in ev])) if ev else "-"
        print("  %-28s %-9s %6d %11d %9s %9s"
              % (cls, rc[0]["riferimento"], len(rc), len(ev), coh, rmed))
    motivi = defaultdict(int)
    for r in rows:
        if r["non_valutabile"]:
            motivi[r["motivo"]] += 1
    for motivo, k in sorted(motivi.items(), key=lambda t: -t[1]):
        print("  non valutabili, %s: %d" % (motivo, k))


def main():
    parser = argparse.ArgumentParser(
        description="DA3 sui frame generati da Wan, con le maschere SAM (passo 0.2 esteso).")
    parser.add_argument("--clips-dir", default=os.path.join("..", "wan2_2", "outputs", "test_clips"))
    parser.add_argument("--sam-dir", default=os.path.join("..", "sam3_1", "outputs", "generated_eval"))
    parser.add_argument("--sam-csv", default=None,
                        help="default: il sam_generated_*.csv piu' recente in --sam-dir")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--process-res", type=int, default=DEFAULT_PROCESS_RES)
    args = parser.parse_args()

    sam_csv = args.sam_csv or latest_sam_csv(args.sam_dir)
    rows, review, elapsed, contaminated = evaluate(
        args.clips_dir, args.sam_dir, sam_csv, args.model, args.process_res)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUT_DIR, "da3_generated_%s.csv" % ts)
    review_path = os.path.join(OUT_DIR, "revisione_%s.csv" % ts)
    write_csv(out_path, rows)
    write_csv(review_path, review)

    print("\n--- sintesi ---")
    print("modello:           %s" % args.model)
    print("depth:             relativa, nessun allineamento (r e' indipendente dalla scala)")
    print("frame:             %d" % len(rows))
    print("risultati:         %s" % out_path)
    print("scheda revisione:  %s (colonne maschera_ok, depth_ok, note da compilare)" % review_path)
    print("picco VRAM:        %.2f GB su un budget di %.1f"
          % (max(r["vram_peak_gb"] for r in rows), VRAM_BUDGET_GB))
    print("misure contaminate da throttling: %d" % contaminated)
    print("tempo totale:      %.1f s" % elapsed)
    summarize(rows)


if __name__ == "__main__":
    main()
