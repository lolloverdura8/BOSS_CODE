# eval_on_carla.py
#
# Fase B.2: misura Depth Anything 3 contro la ground truth densa di CARLA.
#
# Lo smoke test dice se il modello gira; questo dice quanto e' preciso, ed e' il
# vero scopo del blocco B. Legge il manifest eval_set.json prodotto da
# build_eval_set.py, da al modello SOLO il flusso rgb, e confronta la depth
# stimata con quella esatta del simulatore.
#
# Due cose vanno tenute a mente leggendo i numeri che produce, ed entrambe sono
# gestite qui dentro:
#
#   1. DA3-BASE produce depth RELATIVA (is_metric = 0): i valori sono corretti a
#      meno di un fattore di scala ignoto. Confrontarli con i millimetri di CARLA
#      senza allineare non misurerebbe il modello ma l'arbitrarieta' della sua
#      scala. Si allinea, e si dichiara come.
#   2. La depth di CARLA satura a 65535 mm. Quel valore non e' una misura ma un
#      clip: il cielo e ogni edificio lontano ci finiscono dentro. I pixel saturi
#      sono esclusi da carla_gt.read_depth_m() e non entrano in nessuna metrica.
import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import torch

# Prima di importare depth_anything_3: api.py sceglie bfloat16 se la scheda lo
# supporta, e quel file e' del repo upstream clonato, che non va modificato. Si
# interviene sulla condizione. I due formati pesano uguale, quindi non e' una
# scelta di memoria ma di compromesso numerico.
torch.cuda.is_bf16_supported = lambda *a, **k: False

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "CARLA")))
import carla_gt  # noqa: E402

from depth_anything_3.api import DepthAnything3  # noqa: E402

DEFAULT_MODEL = "depth-anything/DA3-BASE"
DEFAULT_PROCESS_RES = 1008

# Oltre questa occupazione il run si ferma. Non e' prudenza generica: su Windows
# il driver NVIDIA puo' riversare su RAM di sistema invece di sollevare un OOM, e
# il run proseguirebbe centinaia di volte piu' lento senza che nulla lo segnali.
# 13 GB lascia margine sui 16,3 della scheda, di cui ~1,3 gia' presi dal desktop.
VRAM_BUDGET_GB = 13.0

# Sopra questa temperatura, o con una maschera di throttling non nulla, la misura
# e' fatta in condizioni diverse dalle altre e va marcata: un confronto fra
# modelli misurati in stati termici diversi non e' un confronto.
TEMP_LIMIT_C = 83
TELEMETRY_EVERY = 10

# La fascia che conta per un dispositivo indossato.
NEAR_BAND_M = 20.0

# delta<1,25 e' la soglia convenzionale della letteratura MDE.
DELTA_THRESHOLD = 1.25


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


def align_median(pred, gt, valid):
    """Allineamento a scala mediana: la convenzione standard per depth relativa.

    Un solo grado di liberta', quindi non puo' mascherare errori di forma: se la
    struttura della scena e' sbagliata, nessuna scala la raddrizza.
    """
    mp, mg = np.median(pred[valid]), np.median(gt[valid])
    if mp <= 0:
        return None, None
    scale = float(mg / mp)
    return pred * scale, scale


def align_scale_shift_inverse(pred, gt, valid):
    """Allineamento a scala e offset nello spazio inverso, come controprova.

    Due gradi di liberta' invece di uno: se i risultati migliorassero molto
    rispetto alla sola scala, vorrebbe dire che il modello sbaglia l'origine
    della profondita' e non solo il fattore - informazione utile, e il motivo per
    cui vale la pena calcolare entrambi.
    """
    dp = 1.0 / np.clip(pred[valid], 1e-6, None)
    dg = 1.0 / np.clip(gt[valid], 1e-6, None)
    A = np.stack([dp, np.ones_like(dp)], axis=1)
    try:
        (a, b), *_ = np.linalg.lstsq(A, dg, rcond=None)
    except np.linalg.LinAlgError:
        return None, None
    disp = a * (1.0 / np.clip(pred, 1e-6, None)) + b
    out = 1.0 / np.clip(disp, 1e-6, None)
    return out, (float(a), float(b))


def depth_metrics(pred, gt, mask):
    """AbsRel, RMSE e delta<1,25 sui soli pixel della maschera."""
    n = int(mask.sum())
    if n == 0:
        return {"n_px": 0, "absrel": None, "rmse": None, "delta1": None}
    p, g = pred[mask], gt[mask]
    ratio = np.maximum(p / g, g / p)
    return {
        "n_px": n,
        "absrel": float(np.mean(np.abs(p - g) / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "delta1": float(np.mean(ratio < DELTA_THRESHOLD)),
    }


def evaluate(manifest_path, model_name, process_res, limit):
    with open(manifest_path) as f:
        manifest = json.load(f)
    base = os.path.dirname(os.path.abspath(manifest_path))
    frames = manifest["frames"][:limit] if limit else manifest["frames"]

    print("modello:      %s" % model_name)
    print("manifest:     %s (%d frame)" % (manifest_path, len(frames)))
    print("process_res:  %d" % process_res)

    model = DepthAnything3.from_pretrained(model_name).to(device=torch.device("cuda"))
    torch.cuda.reset_peak_memory_stats()

    rows = []
    contaminated = 0
    started = time.time()

    for n, entry in enumerate(frames):
        rgb = carla_gt.read_rgb(os.path.join(base, entry["rgb"]))
        gt, valid = carla_gt.read_depth_m(os.path.join(base, entry["depth"]),
                                          entry["depth_scale"])

        prediction = model.inference([rgb], process_res=process_res)
        pred = np.asarray(prediction.depth[0], dtype=np.float64)

        if pred.shape != gt.shape:
            sys.exit("shape discordi al frame %s/%06d: pred %s, gt %s. Con process_res=%d"
                     " su %dx%d il modello dovrebbe restituire la risoluzione nativa."
                     % (entry["session"], entry["frame_index"], pred.shape, gt.shape,
                        process_res, entry["width"], entry["height"]))

        finite = np.isfinite(pred)
        usable = valid & finite & (pred > 0)
        near = usable & (gt <= NEAR_BAND_M)

        # I pixel degli ostacoli piazzati dallo scenario, separati dal resto e
        # tenuti distinti PER CLASSE. Il per-classe non e' un raffinamento: la
        # prima versione aggregava tutti gli override in una maschera sola e il
        # risultato era fuorviante, perche' quelle classi si comportano in modo
        # opposto. Un gradino poggia a terra e DA3 lo azzecca (delta1 0,99); un
        # ramo sospeso non tocca nulla e DA3 lo colloca 3,5 volte piu' lontano
        # (delta1 0,00). Aggregarli produceva una mediana che descriveva solo il
        # gruppo piu' numeroso e nascondeva entrambi i fenomeni.
        override = [i for i in entry["instances"] if i["source"] == "override"]
        prop = np.zeros_like(usable)
        prop_by_class = {}
        if override:
            tag, ids = carla_gt.read_instance(os.path.join(base, entry["instance"]))
            for inst in override:
                m = carla_gt.instance_mask(tag, ids, inst["carla_tag"],
                                           inst["instance_id"])
                prop |= m
                cls = inst["boss_class"]
                prop_by_class[cls] = prop_by_class.get(cls, np.zeros_like(usable)) | m

        row = {
            "session": entry["session"],
            "frame_index": entry["frame_index"],
            "n_valid_px": int(valid.sum()),
            "n_usable_px": int(usable.sum()),
            "n_nonfinite_px": int((~finite).sum()),
            "pct_saturated": round(100.0 * float((~valid).mean()), 2),
        }

        aligned, scale = align_median(pred, gt, usable)
        if aligned is None:
            row["align_failed"] = True
            rows.append(row)
            continue
        row["scale_median"] = round(scale, 6)
        for k, v in depth_metrics(aligned, gt, usable).items():
            row["med_%s" % k] = v
        for k, v in depth_metrics(aligned, gt, near).items():
            row["med_near_%s" % k] = v
        for k, v in depth_metrics(aligned, gt, usable & ~prop).items():
            row["rest_%s" % k] = v
        for cls, m in prop_by_class.items():
            for k, v in depth_metrics(aligned, gt, usable & m).items():
                row["prop_%s_%s" % (cls, k)] = v

        ss, coef = align_scale_shift_inverse(pred, gt, usable)
        if ss is not None:
            row["ss_a"], row["ss_b"] = round(coef[0], 6), round(coef[1], 6)
            for k, v in depth_metrics(ss, gt, usable).items():
                row["ss_%s" % k] = v
            for k, v in depth_metrics(ss, gt, near).items():
                row["ss_near_%s" % k] = v

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
        if (n + 1) % 10 == 0:
            print("  %d/%d frame" % (n + 1, len(frames)), flush=True)

    elapsed = time.time() - started
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return manifest, rows, elapsed, contaminated


def summarize(rows, prefix, label):
    vals = dict((k, [r["%s_%s" % (prefix, k)] for r in rows
                     if r.get("%s_%s" % (prefix, k)) is not None])
                for k in ("absrel", "rmse", "delta1"))
    if not vals["absrel"]:
        print("  %-28s nessun frame misurabile" % label)
        return
    print("  %-28s AbsRel %.4f | RMSE %.3f m | delta<1,25 %.4f   (su %d frame)"
          % (label, float(np.median(vals["absrel"])), float(np.median(vals["rmse"])),
             float(np.median(vals["delta1"])), len(vals["absrel"])))


def main():
    parser = argparse.ArgumentParser(
        description="Misura Depth Anything 3 contro la ground truth CARLA (Fase B.2).")
    parser.add_argument("--eval-set", default=os.path.join("..", "CARLA", "eval_set.json"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--process-res", type=int, default=DEFAULT_PROCESS_RES)
    parser.add_argument("--limit", type=int, default=None,
                        help="usa solo i primi N frame, per una prova rapida")
    args = parser.parse_args()

    manifest, rows, elapsed, contaminated = evaluate(
        args.eval_set, args.model, args.process_res, args.limit)

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "da3_eval_%s.csv"
                            % datetime.now().strftime("%Y%m%d_%H%M%S"))
    fields = sorted(set().union(*(set(r) for r in rows)))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("\n--- sintesi ---")
    print("modello:           %s" % args.model)
    print("depth:             relativa, allineata alla GT frame per frame")
    print("frame misurati:    %d" % len(rows))
    print("risultati:         %s" % out_path)
    print("picco VRAM:        %.2f GB su un budget di %.1f"
          % (max(r.get("vram_peak_gb", 0) for r in rows), VRAM_BUDGET_GB))
    print("misure contaminate da throttling: %d" % contaminated)
    print("tempo totale:      %.1f s (registrato, NON usato come metrica di confronto)"
          % elapsed)

    sat = [r["pct_saturated"] for r in rows]
    print("\npixel saturi nella GT: mediana %.1f%% (esclusi da tutte le metriche)"
          % float(np.median(sat)))

    print("\nallineamento a scala mediana:")
    summarize(rows, "med", "tutti i pixel validi")
    summarize(rows, "med_near", "entro %.0f m" % NEAR_BAND_M)
    print("\nallineamento a scala e offset (controprova):")
    summarize(rows, "ss", "tutti i pixel validi")
    summarize(rows, "ss_near", "entro %.0f m" % NEAR_BAND_M)

    classes = sorted(set(k[len("prop_"):-len("_n_px")] for r in rows for k in r
                         if k.startswith("prop_") and k.endswith("_n_px")))
    if classes:
        print("\nostacoli piazzati dallo scenario, una riga per classe.")
        print("Il discrimine e' il contatto col suolo, non l'essere un ostacolo:")
        for cls in classes:
            summarize(rows, "prop_%s" % cls, cls)
        summarize(rows, "rest", "(riferimento: tutto il resto)")


if __name__ == "__main__":
    main()
