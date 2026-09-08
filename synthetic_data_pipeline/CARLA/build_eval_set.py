# build_eval_set.py
#
# Sceglie i frame su cui si misurano SAM 3.1 e Depth Anything 3 (Fase B.1-B.2) e
# scrive il manifest eval_set.json.
#
# Perche' un manifest e non "tutti i frame": misurare su 900 frame quasi-duplicati
# non aggiunge informazione e rende ogni giro lento. Perche' un file e non una
# regola nel codice: le immagini stanno sotto outputs/, che e' gitignorata, quindi
# il manifest e' l'unica cosa che finisce in git e l'unica che rende la misura
# ripetibile fra mesi. Contiene lo sha256 di ogni RGB proprio per questo: se un
# frame cambiasse, il confronto se ne accorgerebbe.
#
# Il manifest porta con se' anche la ground truth di ogni istanza (classe, box,
# area, distanza), cosi' gli script di valutazione non devono ri-parsare
# annotations.jsonl ne' conoscere la struttura di una sessione: leggono un file e
# basta.
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import carla_gt

# Le tre sessioni che hanno superato il Gate A e sono state esportate. Le altre
# nove in outputs/ sono run precedenti senza export/, tenute come storico.
SESSIONS = [
    "outputs/urban_day_20260907_110348",
    "outputs/overhead_obstacle_20260907_111201",
    "outputs/silent_vehicles_20260907_112134",
]

# Tetto di sicurezza per sessione, non un bersaglio: la dimensione del set la
# decide la copertura per classe qui sotto. Un budget fisso e' stato provato e
# scartato - a 25 frame la sessione overhead_obstacle lo esauriva prima di
# completare ramo_sporgente, che si fermava a 7 frame su 13 disponibili.
MAX_FRAMES_PER_SESSION = 40

# Quanti frame garantire a ciascuna classe BOSS presente nella sessione, e con
# quale distanza minima fra loro. A 16 Hz, quattro frame sono un quarto di
# secondo di cammino: sotto questa soglia si misurerebbe piu' volte la stessa
# inquadratura, gonfiando la confidenza senza aggiungere scene diverse.
#
# Serve perche' la selezione a solo punteggio, misurata, dava 4-7 istanze alle
# classi degli ostacoli sospesi: una mediana di IoU su quattro valori non e' un
# risultato, ed erano proprio le classi prioritarie del report interviste. La
# causa e' che i prop sono piazzati just-in-time e restano inquadrati per una
# finestra breve, mentre pali e automobili compaiono ovunque.
#
# Il gap non puo' essere molto piu' largo, o il tetto diventa invalicabile:
# misurato sulle sessioni, con gap 4 i frame disponibili sono 13
# per ramo_sporgente, 12 per gradino, 8 per insegna_cartello_basso e 7 per
# ostacolo_sospeso_generico. Quei due ultimi numeri sono il tetto fisico della
# registrazione, non un limite della selezione: si prende quel che c'e' e lo si
# dichiara.
TARGET_FRAMES_PER_CLASS = 12
COVERAGE_FRAME_GAP = 4

# La fascia che conta per un dispositivo indossato. Un ostacolo a 60 m non e' un
# problema di sicurezza; uno a 5 m si'. Entra nel punteggio di selezione e sara'
# poi l'intervallo su cui si guardano per prime le metriche di profondita'.
NEAR_DISTANCE_M = 20.0

# Sopra questa soglia la distanza non e' una misura ma il clip a 16 bit della
# depth. Le istanze cosi' marcate restano nel manifest - servono a SAM, che
# segmenta anche cio' che e' lontano - ma vanno escluse da qualunque statistica
# metrica.
SATURATED_DISTANCE_M = 65.5

OUT_PATH = "eval_set.json"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def score_frame(instances):
    """Quanto un frame e' utile alla misura.

    Tre contributi, in ordine di peso. Gli ostacoli in override valgono il triplo
    perche' sono le classi che esistono solo grazie allo scenario - rami, insegne,
    gradini, veicoli elettrici - e sono l'unica ragione per cui quelle sessioni
    sono state registrate. Il numero di classi distinte premia i frame ricchi, che
    dicono di piu' per singola inferenza. Le istanze vicine sono limitate a cinque
    perche' oltre e' saturazione: un frame con venti oggetti a due metri non e'
    piu' informativo di uno con cinque.
    """
    n_override = sum(1 for i in instances if i["source"] == "override")
    n_classes = len(set(i["boss_class"] for i in instances))
    n_near = sum(1 for i in instances
                 if i["distance_m"] is not None and i["distance_m"] <= NEAR_DISTANCE_M)
    return 3 * n_override + n_classes + min(n_near, 5)


def select_frames(records):
    """I frame su cui misurare, scelti per copertura delle classi.

    La fase di copertura scorre le classi dalla piu' rara alla piu' comune e
    prende frame finche' ciascuna non raggiunge TARGET_FRAMES_PER_CLASS. L'ordine
    non e' un dettaglio: partire dalle rare fa si' che i frame scelti per un ramo
    sporgente portino in dote anche i pali e le automobili che ci sono dentro,
    mentre l'ordine inverso riempirebbe il budget di scene comuni e lascerebbe le
    classi prioritarie a mani vuote.

    Il punteggio non sceglie quali frame entrano, ma in quale ordine si guardano
    i candidati di una classe: a parita' di classe da coprire si prende la scena
    piu' ricca. La dimensione del set non e' fissata a priori, la determina la
    copertura.

    Greedy e deterministico: non e' l'ottimo globale, ma e' ripetibile, che qui
    conta di piu'.
    """
    ranked = sorted(records, key=lambda r: (-r["score"], r["frame_index"]))
    classes_of = dict((r["frame_index"], set(i["boss_class"] for i in r["instances"]))
                      for r in records)

    frames_with = defaultdict(int)
    for r in records:
        for cls in classes_of[r["frame_index"]]:
            frames_with[cls] += 1

    picked, picked_idx = [], []
    covered = defaultdict(int)

    def take(rec):
        picked.append(rec)
        picked_idx.append(rec["frame_index"])
        for cls in classes_of[rec["frame_index"]]:
            covered[cls] += 1

    def free(fi, gap):
        return all(abs(fi - p) >= gap for p in picked_idx)

    # Un giro alla volta sulla classe piu' indietro rispetto al bersaglio, non una
    # classe per volta fino a saturarla. La differenza e' concreta: scorrendo le
    # classi in sequenza, quelle in fondo trovavano il budget gia' speso, e
    # ramo_sporgente si fermava a 7 frame su 13 disponibili. A deficit nessuna
    # classe viene affamata da quelle processate prima.
    exhausted = set()
    while len(picked) < MAX_FRAMES_PER_SESSION:
        candidates = [c for c in frames_with
                      if covered[c] < TARGET_FRAMES_PER_CLASS and c not in exhausted]
        if not candidates:
            break
        cls = min(candidates, key=lambda c: (covered[c], frames_with[c], c))
        for rec in ranked:
            fi = rec["frame_index"]
            if fi in picked_idx or cls not in classes_of[fi]:
                continue
            if free(fi, COVERAGE_FRAME_GAP):
                take(rec)
                break
        else:
            # Nessun frame ancora disponibile per questa classe: e' il tetto della
            # registrazione, non un problema di budget.
            exhausted.add(cls)

    return sorted(picked, key=lambda r: r["frame_index"])


def collect_session(session_dir):
    session = carla_gt._load_session(session_dir)
    ann_path = os.path.join(session_dir, "export", "annotations.jsonl")
    if not os.path.exists(ann_path):
        sys.exit("%s non ha un export: lanciare prima carla_export.py" % session_dir)

    records = []
    with open(ann_path) as f:
        for line in f:
            rec = json.loads(line)
            instances = []
            for i in rec["instances"]:
                dist = i["distance_m"]
                instances.append({
                    "carla_tag": i["carla_tag"],
                    "carla_tag_name": i["carla_tag_name"],
                    "instance_id": i["instance_id"],
                    "boss_class": i["boss_class"],
                    "boss_class_id": i["boss_class_id"],
                    "bbox": i["bbox"],
                    "area_px": i["area_px"],
                    "truncated": i["truncated"],
                    "distance_m": dist,
                    "distance_saturated": dist is not None and dist >= SATURATED_DISTANCE_M,
                    "source": i["source"],
                })
            records.append({
                "frame_index": rec["frame_index"],
                "frame_id": rec["frame_id"],
                "instances": instances,
                "score": score_frame(instances),
            })
    return session, records


def build():
    frames = []
    per_class = defaultdict(int)
    per_class_near = defaultdict(int)

    for session_dir in SESSIONS:
        if not os.path.isdir(session_dir):
            sys.exit("sessione non trovata: %s" % session_dir)

        # Prima di leggere un solo pixel: se le convenzioni non tornano, tutto
        # cio' che segue sarebbe numericamente plausibile e falso.
        carla_gt.verify_conventions(session_dir)

        session, records = collect_session(session_dir)
        picked = select_frames(records)
        if len(picked) >= MAX_FRAMES_PER_SESSION:
            print("ATTENZIONE: %s ha raggiunto il tetto di %d frame: la copertura per"
                  " classe potrebbe essere stata troncata"
                  % (session_dir, MAX_FRAMES_PER_SESSION), file=sys.stderr)

        name = os.path.basename(session_dir)
        for rec in picked:
            stem = "%06d.png" % rec["frame_index"]
            rgb = os.path.join(session_dir, "rgb", stem).replace("\\", "/")
            frames.append({
                "session": name,
                "scenario": session.get("scenario"),
                "frame_index": rec["frame_index"],
                "frame_id": rec["frame_id"],
                "rgb": rgb,
                "depth": os.path.join(session_dir, "depth", stem).replace("\\", "/"),
                "semantic": os.path.join(session_dir, "semantic", stem).replace("\\", "/"),
                "instance": os.path.join(session_dir, "instance", stem).replace("\\", "/"),
                "rgb_sha256": sha256_of(rgb),
                "width": session["width"],
                "height": session["height"],
                "depth_scale": session["depth_scale"],
                "intrinsics": session["intrinsics"],
                "instances": rec["instances"],
            })
            for i in rec["instances"]:
                per_class[i["boss_class"]] += 1
                if i["distance_m"] is not None and not i["distance_saturated"] \
                        and i["distance_m"] <= NEAR_DISTANCE_M:
                    per_class_near[i["boss_class"]] += 1

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Set di riferimento per le verifiche B.1-B.2: misura di SAM 3.1 e"
                   " Depth Anything 3 contro la ground truth CARLA.",
        "selection": {
            "sessions": SESSIONS,
            "max_frames_per_session": MAX_FRAMES_PER_SESSION,
            "target_frames_per_class": TARGET_FRAMES_PER_CLASS,
            "coverage_frame_gap": COVERAGE_FRAME_GAP,
            "near_distance_m": NEAR_DISTANCE_M,
            "saturated_distance_m": SATURATED_DISTANCE_M,
            "score": "3*n_override + n_classi_distinte + min(n_istanze_vicine, 5)",
        },
        "reading": {
            "paths": "relativi alla cartella che contiene questo manifest"
                     " (synthetic_data_pipeline/CARLA), non alla cwd di chi legge",
            "input": "solo il flusso rgb va dato ai modelli",
            "ground_truth": "depth (uint16 mm), instance (tag in R, id = (B<<8)|G),"
                            " semantic (tag in R). Isolare un'istanza richiede la"
                            " coppia (carla_tag, instance_id), non il solo id.",
            "reader": "carla_gt.py, unica implementazione condivisa dai tre venv",
        },
        "frames": frames,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n--- sintesi ---")
    print("frame selezionati: %d da %d sessioni" % (len(frames), len(SESSIONS)))
    print("manifest:          %s" % OUT_PATH)
    print("istanze totali:    %d" % sum(per_class.values()))
    print("\ncopertura del set (istanze totali / entro %.0f m e non sature):" % NEAR_DISTANCE_M)
    for name in sorted(per_class, key=lambda k: -per_class[k]):
        print("    %-32s %5d / %4d" % (name, per_class[name], per_class_near[name]))

    absent = [c for c in per_class_near if per_class_near[c] == 0]
    if absent:
        print("\nclassi presenti ma mai entro %.0f m con distanza misurabile: %s"
              % (NEAR_DISTANCE_M, ", ".join(sorted(absent))))


if __name__ == "__main__":
    build()
