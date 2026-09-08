# quality_gate.py
#
# Fase B.4: il controllo rapido che decide se una clip generata vale la pena di
# essere scaricata dal pod.
#
# Il contesto e' economico prima che tecnico. La generazione video girera' su GPU
# noleggiata, dove si paga il pod acceso: scaricare centinaia di clip per poi
# scoprire in locale che alcune sono inutilizzabili e' tempo di GPU buttato.
# Questo gate gira sulla CPU del pod mentre la GPU produce la clip successiva,
# quindi in pratica non costa nulla, e risponde a una domanda sola: vale la pena
# scaricarla?
#
# Guarda due cose indipendenti:
#
#   e' a fuoco?   varianza del Laplaciano, che conta quanti contorni netti ci
#                 sono. Tanti contorni = immagine a fuoco, pochi = sfocata.
#                 Soglia INFERIORE: sotto si scarta.
#
#   si muove?     SSIM fra frame consecutivi, che dice quanto due immagini si
#                 somigliano (1,000 = identiche). Soglia SUPERIORE, e il verso
#                 conta: una SSIM bassa e' NORMALE, la camera cammina e la scena
#                 cambia - misurato, i frame CARLA a 16 Hz stanno a 0,455, cioe'
#                 sono meno simili fra loro di una clip generata. Il difetto da
#                 intercettare e' l'opposto: una clip congelata, che da SSIM
#                 vicina a 1 ed e' inutile come dato di addestramento perche' non
#                 contiene movimento.
#
# Dipende solo da opencv-python e numpy. E' un vincolo, non un caso: questo file
# va installato sul pod, e ogni dipendenza in piu' e' tempo di GPU fatturato
# mentre pip risolve. SSIM e' quindi implementata qui invece di importare
# scikit-image.
import argparse
import csv
import os
from datetime import datetime

import cv2
import numpy as np

# Soglie calibrate con --calibrate. Valgono per la risoluzione a cui sono state
# misurate: la varianza del Laplaciano dipende dalla scala - gli stessi frame
# CARLA danno mediana 4736 a 1008x756 e 2150 a 720x480 - quindi ricalibrare
# quando cambia la risoluzione di generazione.
CALIBRATION_RESOLUTION = (720, 480)

# Calibrata: i frame rovinati arrivano al massimo a 161, i buoni peggiori stanno
# a 848 (la clip CogVideoX; il render CARLA parte da 1003). Fra le due
# popolazioni c'e' un fattore 5, e 370 e' il centro geometrico del vuoto.
SHARPNESS_MIN = 370.0

# Qui la soglia NON e' il centro del vuoto, che la calibrazione propone a 0,83, e
# la deroga e' deliberata. I negativi che si possono fabbricare - frame duplicati
# o quasi - sono estremi e stanno tutti sopra 0,9989: il bordo inferiore della
# popolazione cattiva non rappresenta il caso realistico, cioe' una clip lenta ma
# non ferma. A 0,83 si scarterebbero clip legittime con poco movimento; 0,95
# resta ben sotto i negativi misurati e lascia margine al caso vero.
# Da rivedere quando esisteranno clip generate davvero difettose da misurare.
SSIM_MAX = 0.95

# Parametri SSIM standard (Wang et al.): finestra gaussiana 11x11, sigma 1,5.
SSIM_WIN, SSIM_SIGMA = 11, 1.5
SSIM_C1, SSIM_C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2


def to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def sharpness(gray):
    """Varianza del Laplaciano: quanto e' netta l'immagine."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def ssim(a, b):
    """SSIM media fra due immagini in scala di grigi."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    k = (SSIM_WIN, SSIM_WIN)
    mu1 = cv2.GaussianBlur(a, k, SSIM_SIGMA)
    mu2 = cv2.GaussianBlur(b, k, SSIM_SIGMA)
    s11 = cv2.GaussianBlur(a * a, k, SSIM_SIGMA) - mu1 * mu1
    s22 = cv2.GaussianBlur(b * b, k, SSIM_SIGMA) - mu2 * mu2
    s12 = cv2.GaussianBlur(a * b, k, SSIM_SIGMA) - mu1 * mu2
    num = (2 * mu1 * mu2 + SSIM_C1) * (2 * s12 + SSIM_C2)
    den = (mu1 ** 2 + mu2 ** 2 + SSIM_C1) * (s11 + s22 + SSIM_C2)
    return float((num / den).mean())


def read_video_gray(path):
    """I frame di un video in scala di grigi. Lista vuota se non apribile."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(to_gray(frame))
    cap.release()
    return frames


def evaluate_clip(path):
    """Verdetto su una clip. E' la funzione che run_batch.py importera' sul pod.

    Restituisce sempre un dizionario, anche in caso di errore: sul pod un gate
    che solleva eccezioni fermerebbe il lotto invece di scartare una clip.
    """
    frames = read_video_gray(path)
    if len(frames) < 2:
        return {"path": path, "n_frames": len(frames), "verdict": "illeggibile",
                "sharpness_median": None, "ssim_median": None, "reason":
                "meno di due frame leggibili"}

    sharp = [sharpness(f) for f in frames]
    sims = [ssim(frames[i - 1], frames[i]) for i in range(1, len(frames))]

    sharp_med = float(np.median(sharp))
    ssim_med = float(np.median(sims))

    reasons = []
    if sharp_med < SHARPNESS_MIN:
        reasons.append("sfocata (nitidezza %.0f < %.0f)" % (sharp_med, SHARPNESS_MIN))
    if ssim_med > SSIM_MAX:
        reasons.append("statica (SSIM %.3f > %.3f)" % (ssim_med, SSIM_MAX))

    return {
        "path": path,
        "n_frames": len(frames),
        "sharpness_median": round(sharp_med, 1),
        "sharpness_min": round(float(np.min(sharp)), 1),
        "ssim_median": round(ssim_med, 4),
        "ssim_max": round(float(np.max(sims)), 4),
        "verdict": "scarta" if reasons else "tieni",
        "reason": "; ".join(reasons),
    }


# --- calibrazione -------------------------------------------------------------

def load_good_frames(directory, size, limit):
    """Frame sicuramente buoni: render CARLA, portati alla risoluzione bersaglio."""
    names = sorted(n for n in os.listdir(directory) if n.lower().endswith((".png", ".jpg")))
    step = max(1, len(names) // limit)
    out = []
    for name in names[::step][:limit]:
        img = cv2.imread(os.path.join(directory, name))
        if img is None:
            continue
        out.append(to_gray(cv2.resize(img, size, interpolation=cv2.INTER_AREA)))
    return out


def degrade(frames):
    """I negativi, costruiti rovinando i buoni in modo controllato.

    Senza negativi una soglia e' arbitraria: si sa dove stanno i buoni ma non
    dove cade il confine. Con due popolazioni separate la soglia si mette nel
    vuoto in mezzo, e la larghezza di quel vuoto diventa essa stessa un dato: se
    fosse stretto, vorrebbe dire che il criterio non discrimina.
    """
    out = {}
    for sigma in (1.0, 2.0, 3.0):
        out["blur_sigma_%.0f" % sigma] = [
            cv2.GaussianBlur(f, (0, 0), sigma) for f in frames]
    for factor in (2, 3):
        out["downscale_%dx" % factor] = [
            cv2.resize(cv2.resize(f, (f.shape[1] // factor, f.shape[0] // factor),
                                  interpolation=cv2.INTER_AREA),
                       (f.shape[1], f.shape[0]), interpolation=cv2.INTER_LINEAR)
            for f in frames]
    return out


def gap(low_population, high_population):
    """Il vuoto fra due popolazioni e la soglia al centro geometrico.

    Centro geometrico e non aritmetico perche' la nitidezza si muove per ordini
    di grandezza: fra 7 e 1029 la media aritmetica darebbe 518, che sta appiccicato
    al bordo dei buoni.
    """
    hi = max(low_population)
    lo = min(high_population)
    if hi >= lo:
        return None, hi, lo
    return float(np.sqrt(hi * lo)), hi, lo


def calibrate(good_dir, clip_path, size, limit, out_dir):
    print("risoluzione di calibrazione: %dx%d" % size)
    good = load_good_frames(good_dir, size, limit)
    if not good:
        raise SystemExit("nessun frame leggibile in %s" % good_dir)
    clip = read_video_gray(clip_path)
    print("frame buoni (render):  %d da %s" % (len(good), good_dir))
    print("frame generati:        %d da %s" % (len(clip), clip_path))

    rows = []

    def record(population, name, values, metric):
        for v in values:
            rows.append({"popolazione": population, "insieme": name,
                         "metrica": metric, "valore": round(v, 4)})

    good_sharp = [sharpness(f) for f in good]
    clip_sharp = [sharpness(f) for f in clip]
    record("buoni", "render CARLA", good_sharp, "nitidezza")
    record("buoni", "clip generata", clip_sharp, "nitidezza")

    bad_sharp = []
    for name, frames in degrade(good).items():
        v = [sharpness(f) for f in frames]
        bad_sharp += v
        record("rovinati", name, v, "nitidezza")

    good_ssim = [ssim(clip[i - 1], clip[i]) for i in range(1, len(clip))]
    good_ssim += [ssim(good[i - 1], good[i]) for i in range(1, len(good))]
    record("buoni", "frame consecutivi", good_ssim, "ssim")

    # Il negativo per la SSIM e' la clip congelata: frame identici, e frame quasi
    # identici, che e' il caso realistico di un generatore che si pianta.
    frozen = [ssim(f, f) for f in clip[:20]]
    near_frozen = [ssim(f, cv2.GaussianBlur(f, (3, 3), 0.4)) for f in clip[:20]]
    record("rovinati", "congelata", frozen, "ssim")
    record("rovinati", "quasi congelata", near_frozen, "ssim")
    bad_ssim = frozen + near_frozen

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "gate_calibration_%s.csv"
                        % datetime.now().strftime("%Y%m%d_%H%M%S"))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["popolazione", "insieme", "metrica", "valore"])
        w.writeheader()
        w.writerows(rows)

    def stats(name, v):
        v = np.array(v)
        print("  %-22s n=%4d  min %10.4f  mediana %10.4f  max %10.4f"
              % (name, len(v), v.min(), np.median(v), v.max()))

    print("\n--- nitidezza (varianza del Laplaciano) ---")
    stats("render CARLA", good_sharp)
    stats("clip generata", clip_sharp)
    stats("rovinati", bad_sharp)
    thr_sharp, bad_hi, good_lo = gap(bad_sharp, good_sharp + clip_sharp)
    if thr_sharp is None:
        print("  POPOLAZIONI SOVRAPPOSTE (rovinati fino a %.1f, buoni da %.1f):"
              " il criterio non discrimina" % (bad_hi, good_lo))
    else:
        print("  vuoto fra %.1f (peggiore rovinato) e %.1f (peggiore buono):"
              " fattore %.0fx" % (bad_hi, good_lo, good_lo / max(bad_hi, 1e-9)))
        print("  SOGLIA INFERIORE proposta: %.0f" % thr_sharp)

    print("\n--- movimento (SSIM fra frame consecutivi) ---")
    stats("buoni", good_ssim)
    stats("rovinati", bad_ssim)
    thr_ssim, good_hi, bad_lo = gap(good_ssim, bad_ssim)
    if thr_ssim is None:
        print("  POPOLAZIONI SOVRAPPOSTE (buoni fino a %.4f, rovinati da %.4f):"
              " il criterio non discrimina" % (good_hi, bad_lo))
    else:
        print("  vuoto fra %.4f (piu' statico fra i buoni) e %.4f (meno statico fra"
              " i rovinati)" % (good_hi, bad_lo))
        print("  SOGLIA SUPERIORE proposta: %.4f" % thr_ssim)

    print("\ndistribuzioni: %s" % path)
    print("soglie attualmente nel codice: nitidezza >= %.0f, SSIM <= %.4f"
          % (SHARPNESS_MIN, SSIM_MAX))


def main():
    parser = argparse.ArgumentParser(
        description="Gate di qualita' per le clip generate (Fase B.4).")
    parser.add_argument("--calibrate", action="store_true",
                        help="ricava le soglie confrontando buoni e rovinati")
    parser.add_argument("--good", default=None,
                        help="cartella di frame sicuramente buoni (render CARLA)")
    parser.add_argument("--clip", default=None, help="una clip generata")
    parser.add_argument("--limit", type=int, default=40,
                        help="quanti frame buoni campionare")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    if args.calibrate:
        if not args.good or not args.clip:
            raise SystemExit("--calibrate richiede --good e --clip")
        calibrate(args.good, args.clip, CALIBRATION_RESOLUTION, args.limit, args.out_dir)
        return

    if not args.clip:
        raise SystemExit("indicare --clip da valutare, oppure --calibrate")
    result = evaluate_clip(args.clip)
    for k, v in result.items():
        print("%-18s %s" % (k + ":", v))


if __name__ == "__main__":
    main()
