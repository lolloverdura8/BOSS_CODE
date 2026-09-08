# carla_gt.py
#
# Lettore unico della ground truth di una sessione registrata da carla_capture.py.
# Esiste per una ragione sola: la lettura della GT deve avere UNA implementazione,
# non una copia per ogni script che la usa. Tre copie divergono, e divergono in
# silenzio - nessuna delle trappole qui sotto solleva eccezioni, producono tutte
# risultati plausibili e sbagliati.
#
# Lo usano build_eval_set.py (venv CARLA), sam3_1/eval_on_carla.py (venv Python
# 3.12) e depth_anything3/eval_on_carla.py (venv Python 3.11). Per questo dipende
# solo da cv2 e numpy, che sono presenti in tutti e tre, e non importa carla.
#
# Le tre convenzioni di lettura NON sono dichiarate come costanti da credere sulla
# parola: verify_conventions() le ricava dai dati e si ferma se non tornano. Vedi
# la funzione per il perche' di ciascuna.
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

# I tag semantici di CARLA 0.9.16 sono 29, indicizzati 0..28. E' questo intervallo
# che rende identificabile il canale del tag: gli altri due canali portano l'id di
# istanza e sfondano regolarmente la soglia.
MAX_VALID_TAG = 28

# La depth e' salvata in millimetri su 16 bit. 65535 non significa "65,535 m":
# significa "almeno 65,535 m", perche' CARLA codifica fino a 1000 m e la scrittura
# clippa. Il cielo ci finisce sempre, e con lui ogni edificio lontano. Trattarli
# come misure e' l'errore che falsa AbsRel e delta<1,25 senza lasciare traccia.
DEPTH_SATURATED_U16 = 65535
DEPTH_SCALE_DEFAULT = 1000.0

# cv2.imread restituisce BGR. Il tag semantico sta nel canale R, che qui e'
# l'indice 2. L'identita' dell'istanza sta sui due canali restanti, con B come
# byte alto: la documentazione ufficiale CARLA dichiara (G<<8)|B, ma su questa
# installazione e' l'opposto, ed e' verificato da verify_conventions().
TAG_CHANNEL = 2
ID_HIGH_CHANNEL = 0     # B
ID_LOW_CHANNEL = 1      # G


class GroundTruthError(RuntimeError):
    """Convenzione di lettura non verificata: proseguire produrrebbe numeri falsi."""


# --- lettura -----------------------------------------------------------------

def read_rgb(path):
    """(H, W, 3) uint8 in ordine RGB.

    E' l'unico flusso che i modelli devono vedere. La conversione da BGR e' fatta
    qui una volta per tutte perche' PIL, HuggingFace e torchvision si aspettano
    RGB, mentre cv2 legge BGR: e' lo scambio che inverte rosso e blu senza che
    nulla lo segnali, e su una scena urbana il risultato resta credibile.
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise GroundTruthError("RGB non leggibile: %s" % path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_depth_m(path, depth_scale=DEPTH_SCALE_DEFAULT):
    """(depth_m, valid): metri float32 e la maschera dei pixel utilizzabili.

    `valid` esclude sia gli zeri (assenza di informazione) sia i saturi (oltre il
    rappresentabile). Ogni metrica di profondita' va calcolata SOLO dove valid e'
    True: sulle sessioni di riferimento i saturi sono il 4-26% delle istanze e
    fino al 13% dei pixel di un frame.
    """
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise GroundTruthError("depth non leggibile: %s" % path)
    if raw.dtype != np.uint16 or raw.ndim != 2:
        raise GroundTruthError(
            "depth con formato inatteso in %s: dtype=%s shape=%s, attesi uint16 e 2 assi."
            " Un PNG a 8 bit qui significa che la sessione e' stata scritta senza"
            " cv2.IMREAD_UNCHANGED o con un backend che ha fatto downcast."
            % (path, raw.dtype, raw.shape))
    valid = (raw > 0) & (raw < DEPTH_SATURATED_U16)
    return raw.astype(np.float32) / float(depth_scale), valid


def read_instance(path):
    """(tag, ids): i due piani dell'instance segmentation, gia' decodificati.

    `tag` e' il tag semantico per pixel, `ids` l'identita' a 16 bit dell'attore.
    Nessuno dei due basta da solo a isolare un'istanza: serve la coppia, vedi
    instance_mask().
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise GroundTruthError("instance non leggibile: %s" % path)
    if img.ndim != 3 or img.shape[2] < 3:
        raise GroundTruthError("instance con shape inattesa in %s: %s" % (path, img.shape))
    tag = img[:, :, TAG_CHANNEL].astype(np.uint32)
    ids = (img[:, :, ID_HIGH_CHANNEL].astype(np.uint32) << 8) | img[:, :, ID_LOW_CHANNEL]
    return tag, ids


def read_semantic(path):
    """Tag semantico per pixel, dalla semantic segmentation."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise GroundTruthError("semantic non leggibile: %s" % path)
    return img[:, :, TAG_CHANNEL].astype(np.uint32)


def instance_mask(tag, ids, carla_tag, instance_id):
    """Maschera booleana di UNA istanza, con la chiave composta (tag, id).

    Raggruppare per solo instance_id e' l'errore che rovina lo IoU: l'id e' a 16
    bit e oggetti con tag semantici diversi possono condividerlo. Misurato sul
    frame 236 di silent_vehicles_20260907_112134, un'istanza da 271 px ne
    misurava 1672 raggruppando per solo id, e 271 esatti con la coppia.
    carla_export.py raggruppa per coppia: chi rilegge la GT deve fare lo stesso o
    i due non parlano della stessa cosa.
    """
    return (ids == instance_id) & (tag == carla_tag)


# --- verifica delle convenzioni ----------------------------------------------

def _load_session(session_dir):
    path = os.path.join(session_dir, "session.json")
    if not os.path.exists(path):
        raise GroundTruthError("session.json assente in %s" % session_dir)
    with open(path) as f:
        return json.load(f)


def _instance_files(session_dir):
    return sorted(glob.glob(os.path.join(session_dir, "instance", "*.png")))


def _check_tag_channel(session_dir, sample):
    """Quale canale porta il tag semantico.

    Criterio: i tag validi stanno in 0..28. Il canale del tag e' l'unico che non
    sfonda quella soglia; gli altri due portano meta' di un id a 16 bit e la
    sfondano su qualunque scena non vuota. Non e' un ragionamento a priori: se in
    un frame anche gli altri canali restassero sotto 28 il criterio non
    discriminerebbe, e in quel caso il controllo lo dichiara invece di passare.
    """
    maxima = [0, 0, 0]
    for path in sample:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        for c in range(3):
            m = int(img[:, :, c].max())
            if m > maxima[c]:
                maxima[c] = m

    ok = maxima[TAG_CHANNEL] <= MAX_VALID_TAG
    others = [maxima[c] for c in range(3) if c != TAG_CHANNEL]
    discriminating = all(m > MAX_VALID_TAG for m in others)

    detail = ("max per canale B/G/R = %d/%d/%d, tag atteso nel canale R (indice %d)"
              % (maxima[0], maxima[1], maxima[2], TAG_CHANNEL))
    if not ok:
        detail += (" -- il canale del tag sfonda 28: la convenzione di canale e' sbagliata"
                   " oppure queste non sono immagini di instance segmentation")
    elif not discriminating:
        detail += (" -- ATTENZIONE: anche gli altri canali stanno sotto 28, quindi questo"
                   " campione non discrimina. Ampliare il campione prima di fidarsi")
    return ok, detail


def _check_id_formula(session_dir, sample):
    """Quale formula ricostruisce l'identita' dell'attore.

    Criterio: session.json contiene gli id degli attori spawnati dallo scenario,
    che sono noti indipendentemente dai pixel. La formula giusta li ritrova nei
    frame in cui A7 ha certificato che sono inquadrati; quella sbagliata no.
    E' un confronto contro un dato esterno, non contro un'assunzione.
    """
    session = _load_session(session_dir)
    tracked = session.get("spawned_actors") or []
    if not tracked:
        return None, "nessun attore tracciato in questa sessione: controllo non eseguibile"

    known = set(a["instance_id"] for a in tracked)
    hits_ok, hits_swapped = set(), set()
    for path in sample:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        a = (img[:, :, ID_HIGH_CHANNEL].astype(np.uint32) << 8) | img[:, :, ID_LOW_CHANNEL]
        b = (img[:, :, ID_LOW_CHANNEL].astype(np.uint32) << 8) | img[:, :, ID_HIGH_CHANNEL]
        hits_ok |= known & set(np.unique(a).tolist())
        hits_swapped |= known & set(np.unique(b).tolist())
        if hits_ok and len(hits_ok) == len(known):
            break

    ok = len(hits_ok) > 0 and len(hits_ok) > len(hits_swapped)
    detail = ("(B<<8)|G ritrova %d id noti su %d, (G<<8)|B ne ritrova %d"
              % (len(hits_ok), len(known), len(hits_swapped)))
    if not hits_ok and not hits_swapped:
        detail += (" -- nessuna delle due formule trova nulla: il campione di frame potrebbe"
                   " non inquadrare gli attori, oppure la codifica e' cambiata")
    return ok, detail


def _check_composite_key(session_dir):
    """Che il raggruppamento sia per la coppia (tag, id) e non per il solo id.

    Criterio: carla_export.py ha gia' scritto bbox e area per ogni istanza. Le
    maschere ricostruite qui devono riprodurle al pixel. Il controllo riporta
    anche quante istanze sbaglierebbe la chiave per solo id, cosi' si vede se sta
    davvero discriminando su questa sessione o se il frame scelto e' innocuo.
    """
    ann_path = os.path.join(session_dir, "export", "annotations.jsonl")
    if not os.path.exists(ann_path):
        return None, "export/annotations.jsonl assente: controllo non eseguibile"

    best = None
    with open(ann_path) as f:
        for line in f:
            rec = json.loads(line)
            if best is None or len(rec["instances"]) > len(best["instances"]):
                best = rec
            if best is not None and len(best["instances"]) >= 20:
                break
    if best is None or not best["instances"]:
        return None, "nessuna istanza esportata: controllo non eseguibile"

    stem = "%06d.png" % best["frame_index"]
    tag, ids = read_instance(os.path.join(session_dir, "instance", stem))

    matched = mismatched = 0
    id_only_wrong = 0
    for inst in best["instances"]:
        m = instance_mask(tag, ids, inst["carla_tag"], inst["instance_id"])
        if not m.any():
            mismatched += 1
            continue
        rows, cols = np.where(m)
        bbox = [int(cols.min()), int(rows.min()),
                int(cols.max() - cols.min() + 1), int(rows.max() - rows.min() + 1)]
        if bbox == inst["bbox"] and int(m.sum()) == inst["area_px"]:
            matched += 1
        else:
            mismatched += 1
        if int((ids == inst["instance_id"]).sum()) != inst["area_px"]:
            id_only_wrong += 1

    ok = mismatched == 0 and matched > 0
    detail = ("frame %06d: %d istanze su %d riprodotte al pixel con la coppia (tag, id);"
              " la chiave per solo id ne sbaglierebbe %d"
              % (best["frame_index"], matched, matched + mismatched, id_only_wrong))
    return ok, detail


def verify_conventions(session_dir, sample_size=12, raise_on_fail=True):
    """Verifica le tre convenzioni di lettura sui dati di questa sessione.

    Restituisce una lista di (nome, esito, dettaglio) dove esito e' True, False o
    None (controllo non eseguibile). Con raise_on_fail solleva GroundTruthError
    al primo False: e' il comportamento voluto negli script di misura, perche' un
    numero prodotto con una convenzione sbagliata e' peggio di nessun numero.
    """
    files = _instance_files(session_dir)
    if not files:
        raise GroundTruthError("nessuna immagine in %s/instance" % session_dir)
    step = max(1, len(files) // sample_size)
    sample = files[::step][:sample_size]

    results = [
        ("canale del tag", ) + _check_tag_channel(session_dir, sample),
        ("formula dell'id", ) + _check_id_formula(session_dir, sample),
        ("chiave (tag, id)", ) + _check_composite_key(session_dir),
    ]
    if raise_on_fail:
        failed = [r for r in results if r[1] is False]
        if failed:
            raise GroundTruthError(
                "convenzione di lettura non verificata in %s:\n  %s"
                % (session_dir, "\n  ".join("%s: %s" % (n, d) for n, _, d in failed)))
    return results


def print_verification(session_dir, results):
    print("\n%s" % session_dir)
    for name, ok, detail in results:
        status = "SKIP" if ok is None else ("OK" if ok else "FALLITO")
        print("  %-18s %-8s %s" % (name, status, detail))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verifica le convenzioni di lettura della ground truth CARLA.")
    parser.add_argument("--self-test", action="store_true",
                        help="verifica tutte le sessioni con un export in outputs/")
    parser.add_argument("--session", default=None,
                        help="verifica una singola sessione")
    args = parser.parse_args()

    if args.session:
        targets = [args.session]
    else:
        targets = sorted(os.path.dirname(p) for p in
                         glob.glob(os.path.join("outputs", "*", "export", "annotations.jsonl")))
        targets = [os.path.dirname(t) for t in targets]
    if not targets:
        sys.exit("nessuna sessione esportata trovata in outputs/")

    failed = 0
    for session_dir in targets:
        try:
            results = verify_conventions(session_dir, raise_on_fail=False)
        except GroundTruthError as exc:
            print("\n%s\n  ERRORE %s" % (session_dir, exc))
            failed += 1
            continue
        print_verification(session_dir, results)
        failed += sum(1 for _, ok, _ in results if ok is False)

    print("\n--- sintesi ---")
    print("sessioni verificate: %d" % len(targets))
    if failed:
        sys.exit("%d controlli falliti: non usare questa GT per misurare." % failed)
    print("Tutte le convenzioni verificate.")
