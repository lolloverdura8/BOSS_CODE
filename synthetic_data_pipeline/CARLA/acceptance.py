# acceptance.py
#
# Controlli di accettazione A1-A9 di una sessione registrata da carla_capture.py
# (par. 14.3 del piano, piu' A8 e A9 del ramo Sim2Real). Sono verifiche di poche
# righe che intercettano errori i quali non producono eccezioni: una depth
# decodificata male e' plausibile all'occhio, dei flussi disallineati producono
# file validi, un ostacolo che non e' mai nato lascia una sessione "pulita".
#
# Il modulo non importa carla: legge solo i file della sessione. Per questo i
# controlli si possono rieseguire su una sessione gia' registrata e provare su
# sessioni sintetiche (test_acceptance.py):
#   python acceptance.py --session outputs/<sessione>
#
# L'esito finisce anche in acceptance.json accanto a session.json: chi consuma la
# sessione (carla_export.py, prepare_controls.py) puo' rifiutarla senza
# rieseguire nulla, e le finestre utili di A8 restano scritte.
import argparse
import json
import math
import os
import sys
from datetime import datetime
from itertools import pairwise

import cv2
import numpy as np

from boss_classes import TAG_SKY

DEPTH_MAX_U16 = 65535

A3_SAMPLE_FRAMES = 20       # frame campionati per le verifiche statistiche
A4_SKY_TOLERANCE = 0.99     # quota minima di pixel di cielo che deve stare al massimo

# "Visibile" vuol dire almeno questi pixel dell'istanza col suo tag dominante. E'
# la stessa soglia di carla_export.DEFAULT_MIN_AREA_PX: con un pixel solo, come
# prima, A7 e A8 certificavano finestre in cui l'export non avrebbe scritto nessuna
# annotazione (misurato su sessioni sintetiche in revisione, 05/10).
MIN_VISIBLE_PX = 64

# A7 in secondi e non in frame: i 10 frame di prima valevano 0,625 s a 16 Hz ma
# 0,21 s a 48 Hz. 0,625 s lascia invariata la soglia delle sessioni a 16 Hz.
A7_MIN_SECONDS = 0.625

# A8 tollera buchi brevi: un pedone NPC che passa davanti all'ostacolo lo copre per
# qualche frame senza che la clip smetta di contenerlo. Senza tolleranza un solo
# frame coperto dimezzava la finestra (290 frame visibili meno uno: 150).
A8_MAX_GAP_S = 0.25

# A9: la camera deve essersi mossa davvero. In una sessione di settembre il pedone
# e' rimasto fermo 13 s su 26 e nessun controllo l'ha segnalato; per il ramo
# Sim2Real una camera ferma davanti all'ostacolo passerebbe A7 e A8.
A9_MIN_SPEED_FRACTION = 0.5          # sull'intera sessione, rispetto a walker_speed_mps
A9_MIN_WINDOW_SPEED_FRACTION = 0.6   # dentro ogni finestra di A8


def load_session(out_dir):
    path = os.path.join(out_dir, "session.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_poses(out_dir):
    path = os.path.join(out_dir, "poses.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def tracked_objects(session):
    """Gli oggetti tracciati: nome -> {boss_class, ids}.

    Un gruppo (il monopattino: conducente, asta, manubrio, pedana, ruote) e' un
    oggetto solo, visibile se e' visibile l'unione dei suoi pezzi. Senza gruppo,
    ogni attore e' un oggetto, come nelle sessioni registrate prima del gruppo.
    """
    objects = {}
    for a in session.get("spawned_actors") or []:
        name = a.get("group") or "%s#%d" % (a["boss_class"], a["instance_id"])
        obj = objects.setdefault(name, {"boss_class": a["boss_class"], "ids": []})
        obj["ids"].append(a["instance_id"])
    return objects


def declared_count(session):
    """Quanti oggetti dichiara lo scenario: ostacoli piu' mezzi in movimento."""
    cfg = session.get("scenario_config") or {}
    return len(cfg.get("actors") or []) + len(cfg.get("movers") or [])


def scan_instances(out_dir, names, ids):
    """Pixel per (frame, id, tag), una lettura sola dei PNG di istanza.

    Restituisce counts[i][id] = vettore dei pixel per tag (o None se l'id manca dal
    frame i). Il tag serve perche' l'id e' a 16 bit e un oggetto della mappa puo'
    condividerlo con un attore (carla_gt.instance_mask): si conta solo il tag
    dominante dell'id sull'intera sessione.
    """
    counts = []
    for name in names:
        inst = cv2.imread(os.path.join(out_dir, "instance", name), cv2.IMREAD_UNCHANGED)
        frame = {}
        if inst is not None:
            # (B<<8)|G, non (G<<8)|B come da doc ufficiale: vedi scene_actors.actor_entry.
            pix = (inst[:, :, 0].astype(np.uint32) << 8) | inst[:, :, 1]
            tags = inst[:, :, 2]
            present = set(np.unique(pix).tolist())
            for iid in ids:
                if iid in present:
                    frame[iid] = np.bincount(tags[pix == iid].ravel(), minlength=256)
        counts.append(frame)
    return counts


def visible_frames(counts, objects):
    """Per ogni oggetto, il vettore booleano dei frame in cui e' visibile, e l'area per frame."""
    dominant = {}
    for obj in objects.values():
        for iid in obj["ids"]:
            total = np.zeros(256, dtype=np.int64)
            for frame in counts:
                if iid in frame:
                    total += frame[iid]
            dominant[iid] = int(total.argmax()) if total.any() else None
    out = {}
    for name, obj in objects.items():
        area = []
        for frame in counts:
            a = 0
            for iid in obj["ids"]:
                if iid in frame and dominant[iid] is not None:
                    a += int(frame[iid][dominant[iid]])
            area.append(a)
        out[name] = ([a >= MIN_VISIBLE_PX for a in area], area)
    return out


def best_window(visible, max_gap):
    """(inizio, fine) della finestra piu' lunga, tollerando buchi di max_gap frame.

    La finestra va dal primo all'ultimo frame visibile della corsa, buchi compresi:
    e' l'intervallo che un generatore ridipingerebbe. None se l'oggetto non si vede mai.
    """
    best = None
    start = last = None
    for i, v in enumerate(visible):
        if not v:
            continue
        if start is None or i - last - 1 > max_gap:
            start = i
        last = i
        if best is None or last - start > best[1] - best[0]:
            best = (start, last)
    return best


def path_length(poses, start, end):
    """Metri percorsi dalla camera fra i frame start e end compresi."""
    pts = [p["camera"]["location"] for p in poses[start:end + 1]]
    return sum(math.hypot(b["x"] - a["x"], b["y"] - a["y"]) for a, b in pairwise(pts))


def check_depth(out_dir, counts_dirs):
    """A3 e A4 sui frame campionati. Restituisce i due record."""
    depth_names = counts_dirs["depth"]
    n = len(depth_names)
    step = max(1, n // A3_SAMPLE_FRAMES) if n else 1
    sample = depth_names[::step][:A3_SAMPLE_FRAMES]

    lo_all, hi_all, zeros, sat = [], [], 0, 0
    for name in sample:
        d = cv2.imread(os.path.join(out_dir, "depth", name), cv2.IMREAD_UNCHANGED)
        lo_all.append(float(np.percentile(d, 5)))
        hi_all.append(float(np.percentile(d, 95)))
        zeros += int((d == 0).all())
        sat += int((d == DEPTH_MAX_U16).all())
    a3_ok = bool(sample) and zeros == 0 and sat == 0 and max(hi_all) > min(lo_all)
    a3 = ("A3", a3_ok, "p5 min=%.0f mm, p95 max=%.0f mm, frame tutti-zero=%d tutti-saturi=%d"
          % (min(lo_all) if lo_all else 0, max(hi_all) if hi_all else 0, zeros, sat)
          if sample else "nessun frame da campionare")

    # A4, invariante cielo: e' il controllo piu' informativo. Il cielo in CARLA sta a
    # 1000 m, cioe' ben oltre i 65,5 m rappresentabili: dopo il clip DEVE valere
    # esattamente 65535. Se non vale, o la decodifica R3 e' sbagliata, o i flussi
    # sono disallineati (R2/R5), e la depth non descrive l'immagine accanto.
    sky_total, sky_ok, frames_with_sky = 0, 0, 0
    for name in sample:
        sem = cv2.imread(os.path.join(out_dir, "semantic", name), cv2.IMREAD_UNCHANGED)
        dep = cv2.imread(os.path.join(out_dir, "depth", name), cv2.IMREAD_UNCHANGED)
        mask = sem[:, :, 2] == TAG_SKY          # canale rosso: cv2 legge in BGR
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        frames_with_sky += 1
        sky_total += cnt
        sky_ok += int((dep[mask] == DEPTH_MAX_U16).sum())
    if frames_with_sky == 0:
        a4 = ("A4", False, ("nessun pixel di cielo nei frame campionati:"
                            " controllo non eseguibile, inquadratura da rivedere"))
    else:
        ratio = sky_ok / float(sky_total)
        a4 = ("A4", ratio >= A4_SKY_TOLERANCE,
              "%.4f dei pixel di cielo al massimo (soglia %.2f), su %d frame"
              % (ratio, A4_SKY_TOLERANCE, frames_with_sky))
    return a3, a4


def check_objects(session, objects, vis, dt):
    """A7: gli oggetti dichiarati esistono tutti e si vedono abbastanza."""
    declared = declared_count(session)
    failures = session.get("spawn_failures") or []
    if declared and len(objects) < declared:
        detail = "dichiarati %d oggetti, nella sessione ce ne sono %d" % (declared, len(objects))
        if failures:
            detail += ": " + "; ".join(f.get("reason", "?") for f in failures)
        return ("A7", False, detail)
    if not objects:
        return ("A7", None, "nessun attore tracciato in questo scenario")
    need = math.ceil(A7_MIN_SECONDS / dt - 1e-9)
    weak = [(name, sum(v)) for name, (v, _) in vis.items() if sum(v) < need]
    if weak:
        return ("A7", False, "fuori campo o quasi (servono %d frame con >= %d px): %s"
                % (need, MIN_VISIBLE_PX, weak))
    return ("A7", True, "tutti i %d oggetti dichiarati presenti e visibili in >= %d frame (>= %d px)"
            % (len(objects), need, MIN_VISIBLE_PX))


def check_windows(session, objects, vis, dt):
    """A8: finestra contigua di visibilita' >= min_window_s. Restituisce (record, finestre)."""
    min_window_s = (session.get("scenario_config") or {}).get("min_window_s")
    if not min_window_s:
        return None, {}
    need = math.ceil(min_window_s / dt - 1e-9)
    if not objects:
        return ("A8", False, "lo scenario chiede una finestra di %.1f s ma non c'e' nessun"
                             " oggetto tracciato" % min_window_s), {}
    max_gap = round(A8_MAX_GAP_S / dt)
    windows, short = {}, []
    for name, (v, _) in vis.items():
        w = best_window(v, max_gap)
        windows[name] = w
        if w is None or w[1] - w[0] + 1 < need:
            short.append((name, None if w is None else w[1] - w[0] + 1))
    if short:
        return ("A8", False, "finestra contigua troppo corta (servono %d frame, %.1f s, buchi"
                             " <= %d frame): %s" % (need, min_window_s, max_gap, short)), windows
    detail = ", ".join("%s [%d-%d] %.1f s" % (n, w[0], w[1], (w[1] - w[0] + 1) * dt)
                       for n, w in sorted(windows.items()))
    return ("A8", True, "finestra >= %d frame per tutti: %s" % (need, detail)), windows


def check_motion(session, poses, windows, dt):
    """A9: la camera si e' mossa, sull'intera sessione e dentro le finestre di A8."""
    speed = session.get("walker_speed_mps")
    if not speed or len(poses) < 2:
        return ("A9", None, "velocita' del pedone o pose assenti: controllo non eseguibile")
    total = path_length(poses, 0, len(poses) - 1)
    duration = (len(poses) - 1) * dt
    problems = []
    if total < A9_MIN_SPEED_FRACTION * speed * duration:
        problems.append("sessione: %.1f m in %.1f s (%.2f m/s)" % (total, duration, total / duration))
    for name, w in sorted(windows.items()):
        if w is None or w[1] <= w[0]:
            continue
        length = path_length(poses, w[0], w[1])
        span = (w[1] - w[0]) * dt
        if length < A9_MIN_WINDOW_SPEED_FRACTION * speed * span:
            problems.append("finestra di %s: %.1f m in %.1f s" % (name, length, span))
    if problems:
        return ("A9", False, "camera quasi ferma (pedone a %.1f m/s): %s" % (speed, "; ".join(problems)))
    return ("A9", True, "camera in moto: %.1f m in %.1f s (%.2f m/s)" % (total, duration, total / duration))


def run_acceptance_checks(out_dir):
    """I controlli A1-A9. Restituisce (risultati, info): info contiene le finestre di A8."""
    results = []
    session = load_session(out_dir)
    if session is None:
        return [("A5", False, "session.json assente")], {}

    dirs = ["rgb", "depth", "semantic", "instance"]
    names = dict((d, sorted(os.listdir(os.path.join(out_dir, d)))) for d in dirs)
    poses = load_poses(out_dir)

    n = len(names["rgb"])
    a1 = all(len(names[d]) == n for d in dirs) and len(poses) == n
    results.append(("A1", a1, "rgb=%d depth=%d semantic=%d instance=%d poses=%d"
                    % (len(names["rgb"]), len(names["depth"]), len(names["semantic"]),
                       len(names["instance"]), len(poses))))

    bad = [p["index"] for p in poses if len(set(p["frames"].values())) != 1]
    results.append(("A2", not bad, "ok su %d frame" % len(poses) if not bad
                    else "frame id discordi agli indici %s" % bad[:5]))

    results.extend(check_depth(out_dir, names))

    required = ["intrinsics", "fov", "width", "height", "depth_scale", "carla_version",
                "map", "seed", "fixed_delta_seconds", "camera_transform_relative",
                "spawned_actors", "rgb_sha256"]
    missing = [k for k in required if k not in session or session[k] is None]
    results.append(("A5", not missing, "tutti i campi R7 presenti" if not missing
                    else "campi mancanti: %s" % missing))

    # A6 non si chiude in una sola esecuzione: si stampa l'hash e il confronto e'
    # fra due sessioni registrate con lo stesso seme.
    results.append(("A6", None, "sha256 RGB = %s (confrontare con una seconda esecuzione"
                                " dello stesso scenario)" % session.get("rgb_sha256")))

    dt = session["fixed_delta_seconds"]
    objects = tracked_objects(session)
    ids = sorted(set(i for o in objects.values() for i in o["ids"]))
    vis = visible_frames(scan_instances(out_dir, names["instance"], ids), objects) if ids else {}
    results.append(check_objects(session, objects, vis, dt))

    a8, windows = check_windows(session, objects, vis, dt)
    if a8 is not None:
        results.append(a8)
    results.append(check_motion(session, poses, windows, dt))

    info = {
        "windows": dict((k, list(w) if w else None) for k, w in windows.items()),
        "visible_frames": dict((k, int(sum(v))) for k, (v, _) in vis.items()),
    }
    return results, info


def write_acceptance(out_dir, results, info):
    """acceptance.json: l'esito leggibile da chi consuma la sessione, senza rieseguire."""
    failed = [name for name, ok, _ in results if ok is False]
    with open(os.path.join(out_dir, "acceptance.json"), "w") as f:
        json.dump({
            "valid": not failed,
            "failed": failed,
            "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in results],
            "windows": info.get("windows", {}),
            "visible_frames": info.get("visible_frames", {}),
            "checked_at": datetime.now().isoformat(timespec="seconds"),
        }, f, indent=2)


def print_acceptance(results):
    print("\n--- controlli di accettazione ---")
    failed = 0
    for name, ok, detail in results:
        if ok is None:
            status = "INFO"
        elif ok:
            status = "PASS"
        else:
            status = "FAIL"
            failed += 1
        print("%-4s %-4s %s" % (name, status, detail))
    if failed:
        print("\n%d controlli falliti: la sessione NON e' valida." % failed)
        if any(n == "A4" and ok is False for n, ok, _ in results):
            print("A4 fallito e' il segnale piu' forte: rivedere la decodifica della"
                  " depth (R3) o l'allineamento dei sensori (R2/R5) prima di rigirare.")
    else:
        print("\nTutti i controlli eseguibili superati.")
    return failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Riesegue i controlli di accettazione su una sessione gia' registrata.")
    parser.add_argument("--session", required=True, help="cartella della sessione")
    args = parser.parse_args()
    res, extra = run_acceptance_checks(args.session)
    write_acceptance(args.session, res, extra)
    sys.exit(1 if print_acceptance(res) else 0)
