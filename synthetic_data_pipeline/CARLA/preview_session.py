# preview_session.py
#
# Anteprima di una sessione registrata, per giudicarla a occhio prima di usarla:
#   - preview/preview.mp4: l'RGB a 24 fps con gli oggetti tracciati colorati, la
#     loro box, la classe e la distanza, e in basso una barra del tempo con i frame
#     in cui ciascun oggetto e' visibile e la finestra utile di A8;
#   - preview/contact_sheet.png: 12 frame distribuiti sulla sessione, in griglia.
#
# Non importa carla: gira sul pod accanto a carla_capture.py oppure sul PC, sulla
# sessione scaricata. Legge la GT con carla_gt.py e gli oggetti con acceptance.py,
# cosi' mostra esattamente cio' che i controlli e l'export vedono.
#   python preview_session.py --session outputs/<sessione> [--session ...]
import argparse
import json
import os
import sys

import cv2
import numpy as np

from acceptance import MIN_VISIBLE_PX, load_poses, load_session, tracked_objects
from carla_gt import read_depth_m, read_instance

OUT_FPS = 24
TIMELINE_H = 44
SHEET_COLS, SHEET_ROWS = 4, 3
THUMB_W = 480
# Colori BGR per gli oggetti, nell'ordine in cui compaiono in session.json.
COLORS = [(0, 80, 255), (255, 160, 0), (0, 220, 120), (230, 0, 230), (0, 230, 230)]


def object_masks(ids, objects):
    """nome -> maschera dell'oggetto (unione dei suoi id) nel frame."""
    return dict((name, np.isin(ids, obj["ids"])) for name, obj in objects.items())


def depth_label(depth_m, valid, mask):
    """Distanza z mediana sulla maschera, come il criterio median_mask dell'export."""
    vals = depth_m[mask & valid]
    return None if vals.size == 0 else float(np.median(vals))


def draw_frame(rgb, masks, objects, depth_m, valid, header):
    img = rgb.copy()
    for k, (name, mask) in enumerate(masks.items()):
        if mask.sum() < 1:
            continue
        color = np.array(COLORS[k % len(COLORS)], dtype=np.float32)
        img[mask] = (0.55 * img[mask] + 0.45 * color).astype(np.uint8)
        rows, cols = np.nonzero(mask)
        x0, y0, x1, y1 = int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())
        cv2.rectangle(img, (x0, y0), (x1, y1), COLORS[k % len(COLORS)], 2)
        dist = depth_label(depth_m, valid, mask)
        label = "%s  %s" % (objects[name]["boss_class"], "-" if dist is None else "%.1f m" % dist)
        # Sotto la box se sopra finirebbe nell'intestazione: i sospesi stanno in alto.
        ty = y0 - 8 if y0 > 60 else min(img.shape[0] - 8, y1 + 22)
        cv2.putText(img, label, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, label, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[k % len(COLORS)], 2)
    cv2.putText(img, header, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
    cv2.putText(img, header, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def timeline(width, n_frames, visible, windows, current):
    """Barra in basso: una riga per oggetto, verde dove e' visibile, cornice sulla finestra A8."""
    bar = np.full((TIMELINE_H, width, 3), 30, dtype=np.uint8)
    names = list(visible)
    row_h = max(1, (TIMELINE_H - 6) // max(1, len(names)))
    for r, name in enumerate(names):
        y0 = 3 + r * row_h
        for i, v in visible[name].items():
            if v:
                x = int(i * width / n_frames)
                bar[y0:y0 + row_h - 1, x:x + max(1, width // n_frames + 1)] = COLORS[r % len(COLORS)]
        w = windows.get(name)
        if w:
            x0, x1 = int(w[0] * width / n_frames), int((w[1] + 1) * width / n_frames)
            cv2.rectangle(bar, (x0, y0), (min(width - 1, x1), y0 + row_h - 2), (255, 255, 255), 1)
    xc = int(current * width / n_frames)
    cv2.line(bar, (xc, 0), (xc, TIMELINE_H - 1), (0, 0, 255), 2)
    return bar


def open_writer(path, fps, size):
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, codec
    sys.exit("nessun codec video disponibile per %s" % path)


def preview(session_dir, make_video=True):
    session = load_session(session_dir)
    if session is None:
        sys.exit("session.json assente in %s" % session_dir)
    poses = load_poses(session_dir)
    n = len(poses)
    if n == 0:
        sys.exit("poses.jsonl vuoto in %s" % session_dir)
    dt = session["fixed_delta_seconds"]
    stride = max(1, round(1.0 / dt / OUT_FPS))
    objects = tracked_objects(session)
    windows = {}
    acc_path = os.path.join(session_dir, "acceptance.json")
    if os.path.exists(acc_path):
        with open(acc_path) as f:
            windows = json.load(f).get("windows", {}) or {}

    sampled = list(range(0, n, stride))
    visible = dict((name, {}) for name in objects)
    first_last = dict((name, [None, None]) for name in objects)
    for i in sampled:
        _, ids = read_instance(os.path.join(session_dir, "instance", "%06d.png" % i))
        for name, mask in object_masks(ids, objects).items():
            seen = int(mask.sum()) >= MIN_VISIBLE_PX
            visible[name][i] = seen
            if seen:
                first_last[name][0] = i if first_last[name][0] is None else first_last[name][0]
                first_last[name][1] = i

    out_dir = os.path.join(session_dir, "preview")
    os.makedirs(out_dir, exist_ok=True)
    sheet_idx = set(sampled[int(k * (len(sampled) - 1) / (SHEET_COLS * SHEET_ROWS - 1))]
                    for k in range(SHEET_COLS * SHEET_ROWS))
    thumbs, writer, closest = {}, None, dict((name, None) for name in objects)
    width, height = session["width"], session["height"]
    if make_video:
        writer, codec = open_writer(os.path.join(out_dir, "preview.mp4"), OUT_FPS,
                                    (width, height + TIMELINE_H))
    for i in sampled:
        stem = "%06d.png" % i
        rgb = cv2.imread(os.path.join(session_dir, "rgb", stem), cv2.IMREAD_COLOR)
        _, ids = read_instance(os.path.join(session_dir, "instance", stem))
        depth_m, valid = read_depth_m(os.path.join(session_dir, "depth", stem), session["depth_scale"])
        masks = object_masks(ids, objects)
        for name, mask in masks.items():
            d = depth_label(depth_m, valid, mask)
            if d is not None and visible[name].get(i):
                closest[name] = d if closest[name] is None else min(closest[name], d)
        header = "%s   frame %d/%d   t=%.2f s" % (session["scenario"], i, n - 1, i * dt)
        frame = draw_frame(rgb, masks, objects, depth_m, valid, header)
        if i in sheet_idx:
            thumbs[i] = frame
        if writer is not None:
            writer.write(np.vstack([frame, timeline(width, n, visible, windows, i)]))
    if writer is not None:
        writer.release()

    th = int(THUMB_W * height / width)
    tiles = [cv2.resize(thumbs[i], (THUMB_W, th), interpolation=cv2.INTER_AREA) for i in sorted(thumbs)]
    while len(tiles) < SHEET_COLS * SHEET_ROWS:
        tiles.append(np.zeros((th, THUMB_W, 3), dtype=np.uint8))
    rows = [np.hstack(tiles[r * SHEET_COLS:(r + 1) * SHEET_COLS]) for r in range(SHEET_ROWS)]
    cv2.imwrite(os.path.join(out_dir, "contact_sheet.png"), np.vstack(rows))

    print("\n%s" % session_dir)
    if writer is not None:
        print("  video:   %s (%d frame a %d fps, codec %s)"
              % (os.path.join(out_dir, "preview.mp4"), len(sampled), OUT_FPS, codec))
    print("  provini: %s" % os.path.join(out_dir, "contact_sheet.png"))
    for name, (a, b) in first_last.items():
        span = "mai visibile" if a is None else "visibile dal frame %d al %d (%.1f-%.1f s)" % (a, b, a * dt, b * dt)
        near = closest[name]
        print("  %-28s %s, distanza minima %s" % (name, span, "-" if near is None else "%.1f m" % near))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anteprima video e provini di una sessione CARLA.")
    parser.add_argument("--session", action="append", required=True,
                        help="cartella della sessione (ripetibile)")
    parser.add_argument("--no-video", action="store_true", help="solo il foglio di provini")
    args = parser.parse_args()
    for s in args.session:
        preview(s, make_video=not args.no_video)
