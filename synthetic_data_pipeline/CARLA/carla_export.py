# carla_export.py
#
# Trasforma una sessione registrata da carla_capture.py in annotazioni: classe
# BOSS, box 2D per istanza e distanza metrica per box. E' il secondo tempo della
# separazione del par. 14.4, e la separazione ha uno scopo preciso: questo script
# NON importa carla e non tocca il server, quindi si possono rivedere i criteri di
# esportazione - la mappatura delle classi, la soglia di area, il criterio di
# distanza - senza ri-registrare nulla.
#
# Il par. 5.8.5 chiede che il criterio di calcolo della distanza sia "parametrico e
# documentato". E' il difetto che il par. 3 imputa alle recordings esistenti: li'
# il criterio era ignoto, e per un'auto di sbieco il box conteneva sfondo, con un
# valore distorto in misura non quantificabile. Qui il criterio si sceglie da riga
# di comando e viene scritto in ogni singola riga di annotazione.
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

from boss_classes import (BOSS_CLASSES, BOSS_CLASS_BY_NAME, BOSS_NOT_COVERED_BY_TAG,
                          CARLA_TAG_NAMES, CARLA_TAG_TO_BOSS, CARLA_TAGS_DISCARDED,
                          PRIORITY_ORDER)

# Soglia di area sotto la quale un'istanza non viene esportata. Serve a togliere i
# frammenti di pochi pixel sul bordo dell'inquadratura, che non sono ostacoli
# annotabili ma resti di occlusione, e che gonfierebbero il conteggio di istanze
# per classe rendendo la Matrice di Copertura ottimista.
DEFAULT_MIN_AREA_PX = 64

# Soglia di riferimento del par. 5.1 del D4.1.3: 100-200 istanze annotate per
# classe, con minimo 80-100 prima di qualunque augmentation.
TARGET_INSTANCES_MIN = 100

# I quattro criteri del par. 5.8.5. Il default e' median_mask, e non e' una
# preferenza: e' l'unico che non sbaglia.
#
# Misura su fixture sintetico (oggetto a 8 m, sfondo a 40 m, barra diagonale di
# spessore decrescente cosi' che la maschera occupi una frazione sempre minore del
# proprio bounding box). Errore in metri:
#
#   riempimento del box   median_mask   median_center_third   median_box   p10_box
#            52 %            +0,000            +0,000           +0,000     +0,000
#            39 %            +0,000            +0,000          +32,000     +0,000
#            18 %            +0,000           +32,000          +32,000     +0,000
#             8 %            +0,000           +32,000          +32,000    +32,000
#
# Due cose da leggere in questa tabella. La prima e' che l'errore non degrada in
# modo graduale: sotto la soglia la mediana scatta di colpo sullo sfondo, e la
# distanza dell'ostacolo diventa la distanza dell'edificio dietro. La seconda e'
# che la soglia dipende dalla forma dell'oggetto, quindi non e' prevedibile a
# priori: un palo, un ramo sporgente o una bicicletta di profilo riempiono poco il
# proprio box, e sono esattamente le classi critiche di questo progetto.
# La maschera di istanza c'e' gia' e costa zero: non usarla e' l'errore del par. 3.
DISTANCE_CRITERIA = ("median_mask", "median_center_third", "median_box", "p10_box")


def load_session(session_dir):
    path = os.path.join(session_dir, "session.json")
    if not os.path.exists(path):
        sys.exit("session.json non trovato in %s: la sessione non e' completa" % session_dir)
    with open(path) as f:
        return json.load(f)


def build_override(session):
    """instance_id -> classe BOSS, dagli attori spawnati registrati in session.json.

    E' il meccanismo del par. 5.4, e non un ripiego: un prop spawnato porta il tag
    semantico della sua mesh - tipicamente Static (20) o Other (22) - e mai una
    classe BOSS. Un veicolo elettrico porta lo stesso tag Car di una qualsiasi
    utilitaria. Senza questa tabella, cinque classi su quattordici sarebbero
    invisibili all'export, incluse tutte quelle sospese.
    """
    return dict((a["instance_id"], a["boss_class"]) for a in session.get("spawned_actors", []))


def distance_from_depth(depth_mm, mask, bbox, criterion, depth_scale):
    """Distanza in metri dell'istanza, secondo il criterio scelto.

    I pixel a zero vengono esclusi da tutte le statistiche: zero metri e' una
    distanza fisicamente impossibile per un oggetto inquadrato, ed e' cio' che la
    depth riporta dove non c'e' informazione. Includerli tirerebbe la mediana
    verso il basso di una quantita' che dipende dall'occlusione, non dalla scena.
    """
    x, y, w, h = bbox
    if criterion == "median_mask":
        vals = depth_mm[mask]
    elif criterion == "median_box":
        vals = depth_mm[y:y + h, x:x + w].ravel()
    elif criterion == "p10_box":
        vals = depth_mm[y:y + h, x:x + w].ravel()
    elif criterion == "median_center_third":
        y0, y1 = y + h // 3, y + (2 * h) // 3 + 1
        x0, x1 = x + w // 3, x + (2 * w) // 3 + 1
        vals = depth_mm[y0:y1, x0:x1].ravel()
    else:
        sys.exit("criterio di distanza sconosciuto: %s" % criterion)

    vals = vals[vals > 0]
    if vals.size == 0:
        return None
    if criterion == "p10_box":
        return float(np.percentile(vals, 10)) / depth_scale
    return float(np.median(vals)) / depth_scale


def instances_in_frame(instance_img, override, min_area_px):
    """Istanze presenti in un frame, come (tag, instance_id, maschera, bbox, area).

    cv2 legge in BGR, quindi il canale del tag semantico e' l'indice 2. Per
    l'identita', la documentazione ufficiale CARLA descrive (G<<8)|B, ma la
    verifica empirica su CARLA 0.9.16 (attori spawnati con actor.id noto,
    confrontati contro i pixel effettivamente renderizzati - vedi
    carla_capture.py, spawn_one_actor) mostra l'opposto: (B<<8)|G, cioe' B
    (indice 0) e' il byte alto e G (indice 1) il byte basso. Questa funzione e
    carla_capture.py devono usare la stessa formula: se una delle due cambia
    senza l'altra, l'override smette di risolvere silenziosamente.
    Confondere l'ordine non solleva alcun errore: produce istanze inesistenti.

    Si filtrano subito i pixel irrilevanti - strada, edifici, cielo - prima di
    raggruppare: su una scena urbana sono la larga maggioranza, e tenerli
    significherebbe ordinare 762.000 elementi per frame invece di poche decine di
    migliaia.
    """
    tags = instance_img[:, :, 2].astype(np.uint32)
    ids = (instance_img[:, :, 0].astype(np.uint32) << 8) | instance_img[:, :, 1].astype(np.uint32)

    relevant = np.isin(tags, list(CARLA_TAG_TO_BOSS.keys()))
    if override:
        relevant |= np.isin(ids, list(override.keys()))
    flat = np.flatnonzero(relevant)
    if flat.size == 0:
        return []

    keys = tags.ravel()[flat] * 65536 + ids.ravel()[flat]
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    pix_sorted = flat[order]

    uniq, starts = np.unique(keys_sorted, return_index=True)
    ends = np.append(starts[1:], keys_sorted.size)

    height, width = instance_img.shape[:2]
    out = []
    for key, s, e in zip(uniq, starts, ends):
        area = int(e - s)
        if area < min_area_px:
            continue
        pix = pix_sorted[s:e]
        rows, cols = np.divmod(pix, width)
        y0, y1 = int(rows.min()), int(rows.max())
        x0, x1 = int(cols.min()), int(cols.max())
        mask = np.zeros(height * width, dtype=bool)
        mask[pix] = True
        out.append({
            "carla_tag": int(key // 65536),
            "instance_id": int(key % 65536),
            "mask": mask.reshape(height, width),
            "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
            "area_px": area,
            "truncated": bool(x0 == 0 or y0 == 0 or x1 == width - 1 or y1 == height - 1),
        })
    return out


def export(session_dir, criterion, min_area_px):
    session = load_session(session_dir)
    override = build_override(session)
    depth_scale = session["depth_scale"]

    out_dir = os.path.join(session_dir, "export")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(session_dir, "poses.jsonl")) as f:
        poses = [json.loads(l) for l in f if l.strip()]

    per_class_instances = defaultdict(int)
    per_class_frames = defaultdict(set)
    unmapped_tags = defaultdict(int)
    total_instances = 0
    no_distance = 0

    ann_path = os.path.join(out_dir, "annotations.jsonl")
    with open(ann_path, "w") as ann:
        for pose in poses:
            stem = "%06d.png" % pose["index"]
            inst_img = cv2.imread(os.path.join(session_dir, "instance", stem), cv2.IMREAD_UNCHANGED)
            depth_mm = cv2.imread(os.path.join(session_dir, "depth", stem), cv2.IMREAD_UNCHANGED)
            if inst_img is None or depth_mm is None:
                sys.exit("frame %s incompleto: instance o depth mancante" % stem)

            records = []
            for inst in instances_in_frame(inst_img, override, min_area_px):
                # L'override viene PRIMA del tag: e' l'unica precedenza corretta,
                # perche' un'elettrica e un'utilitaria condividono il tag Car e
                # solo l'id le distingue.
                boss = override.get(inst["instance_id"])
                source = "override"
                if boss is None:
                    boss = CARLA_TAG_TO_BOSS.get(inst["carla_tag"])
                    source = "tag"
                if boss is None:
                    unmapped_tags[inst["carla_tag"]] += 1
                    continue

                dist = distance_from_depth(depth_mm, inst["mask"], inst["bbox"],
                                           criterion, depth_scale)
                if dist is None:
                    no_distance += 1

                records.append({
                    "instance_id": inst["instance_id"],
                    "carla_tag": inst["carla_tag"],
                    "carla_tag_name": CARLA_TAG_NAMES.get(inst["carla_tag"], "?"),
                    "boss_class": boss,
                    "boss_class_id": BOSS_CLASS_BY_NAME[boss]["id"],
                    "bbox": inst["bbox"],
                    "area_px": inst["area_px"],
                    "truncated": inst["truncated"],
                    "distance_m": dist,
                    # Il criterio viaggia con ogni singola annotazione: confrontare
                    # due criteri diventa cosi' una lettura, non una ri-esportazione.
                    "distance_criterion": criterion,
                    "source": source,
                })
                per_class_instances[boss] += 1
                per_class_frames[boss].add(pose["index"])
                total_instances += 1

            ann.write(json.dumps({
                "frame_index": pose["index"],
                "frame_id": pose["frame"],
                "instances": records,
            }) + "\n")

    mapping_path = os.path.join(out_dir, "class_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump({
            "note": "Tabella di corrispondenza applicata a questa esportazione. Il par."
                    " 5.8.4 del piano la richiede nel repository e nel deliverable, con"
                    " le classi CARLA scartate e quelle BOSS non copribili tracciate e"
                    " non nascoste: sono assunzioni, non dimenticanze.",
            "carla_tag_to_boss": dict((str(k), v) for k, v in CARLA_TAG_TO_BOSS.items()),
            "carla_tags_discarded": dict((str(k), v) for k, v in CARLA_TAGS_DISCARDED.items()),
            "boss_not_covered_by_tag": BOSS_NOT_COVERED_BY_TAG,
            "override_applied": dict((str(k), v) for k, v in override.items()),
            "distance_criterion": criterion,
            "min_area_px": min_area_px,
            "unmapped_tags_seen": dict(
                ("%d (%s)" % (k, CARLA_TAG_NAMES.get(k, "?")), v)
                for k, v in sorted(unmapped_tags.items())),
        }, f, indent=2)

    # Riepilogo ordinato per priorita': in testa le due categorie che il report
    # interviste indica come le sole degne di nota, perche' sono quelle su cui la
    # soglia di copertura va verificata davvero.
    summary_path = os.path.join(out_dir, "export_summary.csv")
    rows = sorted(BOSS_CLASSES,
                  key=lambda c: (PRIORITY_ORDER.index(c["priority"]), c["id"]))
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["boss_class", "boss_class_id", "priority", "macro",
                    "n_instances", "n_frames", "target_min", "status"])
        for c in rows:
            n = per_class_instances[c["name"]]
            w.writerow([c["name"], c["id"], c["priority"], c["macro"], n,
                        len(per_class_frames[c["name"]]), TARGET_INSTANCES_MIN,
                        "ok" if n >= TARGET_INSTANCES_MIN else
                        ("sotto_soglia" if n else "assente")])

    print("\n--- sintesi ---")
    print("sessione:        %s" % session_dir)
    print("scenario:        %s" % session.get("scenario", "?"))
    print("frame:           %d" % len(poses))
    print("istanze:         %d (criterio distanza: %s, area minima: %d px)"
          % (total_instances, criterion, min_area_px))
    if no_distance:
        print("senza distanza:  %d istanze con depth tutta a zero nella regione scelta"
              % no_distance)
    print("annotazioni:     %s" % ann_path)
    print("mappatura:       %s" % mapping_path)
    print("copertura:       %s" % summary_path)

    print("\ncopertura per classe (soglia %d istanze, par. 5.1 del D4.1.3):"
          % TARGET_INSTANCES_MIN)
    current = None
    for c in rows:
        if c["priority"] != current:
            current = c["priority"]
            print("  [%s]" % current)
        n = per_class_instances[c["name"]]
        print("    %-32s %6d istanze in %4d frame  %s"
              % (c["name"], n, len(per_class_frames[c["name"]]),
                 "" if n >= TARGET_INSTANCES_MIN else ("<-- sotto soglia" if n else "<-- ASSENTE")))

    if unmapped_tags:
        print("\ntag CARLA visti e non mappati (atteso: sono in carla_tags_discarded):")
        for tag, cnt in sorted(unmapped_tags.items(), key=lambda kv: -kv[1]):
            print("    %2d %-14s %d istanze" % (tag, CARLA_TAG_NAMES.get(tag, "?"), cnt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Esporta classi BOSS, box 2D e distanze da una sessione CARLA.")
    parser.add_argument("--session", required=True,
                        help="cartella della sessione prodotta da carla_capture.py")
    parser.add_argument("--distance-criterion", default="median_mask", choices=DISTANCE_CRITERIA,
                        help="median_mask (default) usa i soli pixel della maschera di"
                             " istanza ed e' l'unico misurato come immune alla"
                             " contaminazione da sfondo nel box; gli altri tre sono i"
                             " criteri elencati dal par. 5.8.5 e sbagliano di colpo quando"
                             " la maschera scende sotto il 50/20/10 % del box")
    parser.add_argument("--min-area-px", type=int, default=DEFAULT_MIN_AREA_PX,
                        help="area minima in pixel di un'istanza esportata"
                             " (default: %d)" % DEFAULT_MIN_AREA_PX)
    args = parser.parse_args()

    export(args.session, args.distance_criterion, args.min_area_px)
