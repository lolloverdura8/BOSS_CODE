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
# COME SI USA
#
#   una clip, alla sua risoluzione nativa (e' la chiamata storica):
#     python quality_gate.py --clip out/wan22_5b/monopattino_00.mp4
#
#   tutte le clip di tutti i modelli, normalizzate e confrontabili fra loro:
#     python quality_gate.py --clips-dir scaricato/out --out-dir eval
#
# Dipende solo da opencv-python e numpy. E' un vincolo, non un caso: questo file
# va installato sul pod, e ogni dipendenza in piu' e' tempo di GPU fatturato
# mentre pip risolve. SSIM e' quindi implementata qui invece di importare
# scikit-image.
import argparse
import csv
import glob
import json
import os
from datetime import datetime

import cv2
import numpy as np

# Soglie calibrate con --calibrate. Valgono per la risoluzione a cui sono state
# misurate: la varianza del Laplaciano dipende dalla scala, quindi ricalibrare
# quando cambia la risoluzione di generazione.
#
# VALORI STORICI, NON APPLICATI DA NESSUNA PARTE. Restano per non perdere la
# provenienza di un numero che si e' visto in giro nei manifest fino al
# 21/09/2026: 370 era il centro geometrico del vuoto fra i frame rovinati (fino
# a 161) e i buoni peggiori (848), misurato pero' a 720x480. A quella
# risoluzione voleva dire qualcosa; applicato alle clip del bake-off, che stanno
# fra 0,90 e 1,04 Mpx, ne marcava "sfocate" tre su cinque confrontando grandezze
# non commensurabili. Il verdetto sulla nitidezza e' stato tolto: vedi
# evaluate_clip.
CALIBRATION_RESOLUTION = (720, 480)
SHARPNESS_MIN_STORICA = 370.0

# Qui la soglia NON e' il centro del vuoto, che la calibrazione propone a 0,83, e
# la deroga e' deliberata. I negativi che si possono fabbricare - frame duplicati
# o quasi - sono estremi e stanno tutti sopra 0,9989: il bordo inferiore della
# popolazione cattiva non rappresenta il caso realistico, cioe' una clip lenta ma
# non ferma. A 0,83 si scarterebbero clip legittime con poco movimento; 0,95
# resta ben sotto i negativi misurati e lascia margine al caso vero.
# Da rivedere quando esisteranno clip generate davvero difettose da misurare.
#
# QUESTA SOGLIA SOPRAVVIVE ALLA RINUNCIA A TUTTE LE ALTRE, e la ragione non e'
# storica. Il suo negativo non ha bisogno di una popolazione di riferimento
# esterna: si fabbrica dalla clip stessa, duplicandone i fotogrammi. Il difetto
# "clip congelata" e' definito per confronto della clip con se' stessa, quindi
# misurabile senza niente d'altro. La nitidezza no, e infatti li' la soglia ad
# altezza normalizzata non c'e'.
#
# Verificata il 20/09/2026 ad altezza 704: i fotogrammi duplicati stanno sopra
# 0,9988, quelli della clip vera arrivano al massimo a 0,8394. La SSIM inoltre
# e' normalizzata e cambia poco con la risoluzione - misurato fra 0,794 e 0,819
# su quattro altezze diverse della stessa clip - quindi una soglia sola vale a
# qualunque altezza di valutazione.
SSIM_MAX = 0.95

# Altezza a cui si normalizza per confrontare modelli diversi. Le clip del
# bake-off stanno fra 1280x704 e 1360x768: 704 e' la minima fra le altezze
# native, quindi portare tutto li' preservando l'aspect ratio e' l'unica scelta
# in cui NIENTE viene ingrandito - ogni clip subisce al massimo una riduzione di
# fattore 0,92. Ingrandire non aggiunge dettaglio ma aggiunge interpolazione, che
# la varianza del Laplaciano legge come sfocatura.
#
# Che la normalizzazione serva davvero e' misurato, non supposto: la stessa clip
# (monopattino_00, 20/09/2026) da' nitidezza 645,6 a 704 di altezza, 468,7 a 576,
# 621,7 a 480 e 1209,9 a 352. Varia di 2,6 volte, e per giunta NON in modo
# monotono - la riduzione toglie dettaglio fine ma concentra i contorni su meno
# pixel, e i due effetti si invertono. Due modelli misurati ad altezze diverse
# non sono confrontabili, e non si puo' nemmeno correggere a mente il verso
# dell'errore. La SSIM invece resta fra 0,794 e 0,819 sulle stesse quattro
# altezze: e' normalizzata, e non ha bisogno di questa cautela.
NORMALIZED_HEIGHT = 704

# Passo temporale di riferimento per la SSIM confrontabile fra modelli, in
# secondi. Gli fps nativi vanno da 8 (CogVideoX) a 24 (Wan, LTX, Hunyuan): fra
# due frame consecutivi passa il triplo del tempo a 8 fps che a 24, quindi la
# SSIM consecutiva misura anche gli fps, non solo il movimento. A 1/8 s lo scarto
# in frame e' intero per tutti - 3 a 24 fps, 2 a 16, 1 a 8 - e nessuna clip va
# interpolata. E' il passo nativo del modello piu' lento: piu' corto di cosi'
# non si puo' andare senza inventare frame.
REFERENCE_DT = 0.125

# Ritaglio centrale: il modo di confrontare la nitidezza SENZA interpolare.
#
# Misurato il 21/09/2026 sulla clip wan22_5b, nativa 1280x704: ridurla a 700 di
# altezza, cioe' dello 0,6 per cento, porta la nitidezza da 1133,4 a 602,1. Meno
# della meta'. E a 640 risale a 706,4, quindi non e' la perdita di dettaglio: e'
# l'interpolazione in se' che smussa i contorni e paga un prezzo fisso appena la
# si tocca. Conseguenza: un ridimensionamento "normalizzante" regala il valore
# pieno ai modelli gia' nati all'altezza bersaglio e taglia del 25-47 per cento
# tutti gli altri, che e' un confondimento piu' grande di quello che curava.
#
# 1280x704 e' il rettangolo piu' grande che entra in tutte le risoluzioni native
# del bake-off (1280x704, 1280x720, 1360x768). Il ritaglio non interpola: i pixel
# sono gli originali, alla loro scala, e sono lo stesso numero per ogni modello.
# Per chi e' gia' 1280x704 il ritaglio e' l'identita', e qui va bene - a
# differenza del ridimensionamento, ritagliare non introduce nessun artefatto,
# quindi identita' e ritaglio sono la stessa misura e non due trattamenti
# diversi.
CROP_SIZE = (1280, 704)

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


def fit_height(gray, h):
    """Porta il frame ad altezza h preservando l'aspect ratio.

    NON ingrandisce mai: se il frame e' gia' piu' basso di h lo restituisce
    intatto. Un ingrandimento non aggiunge dettaglio ma aggiunge interpolazione,
    e la varianza del Laplaciano lo legge come sfocatura - falserebbe verso il
    basso proprio la popolazione che deve fare da riferimento.
    """
    if h is None or gray.shape[0] <= h:
        return gray
    w = int(round(gray.shape[1] * h / float(gray.shape[0])))
    return cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)


def center_crop(gray, size):
    """Ritaglio centrale a dimensione fissa. Non interpola e non ridimensiona.

    Se il frame e' piu' piccolo del bersaglio lo restituisce intatto: allargarlo
    vorrebbe dire inventare pixel, che e' il difetto da cui questa funzione
    esiste per scappare.
    """
    if size is None:
        return gray
    w, h = size
    H, W = gray.shape[:2]
    if W < w or H < h:
        return gray
    x, y = (W - w) // 2, (H - h) // 2
    return gray[y:y + h, x:x + w]


def read_video_gray(path, target_height=None, crop=None):
    """I frame di un video in scala di grigi. Lista vuota se non apribile."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(center_crop(fit_height(to_gray(frame), target_height), crop))
    cap.release()
    return frames


def sharpness_only(path, target_height=None, crop=None):
    """La sola nitidezza mediana, per le passate di confronto.

    Esiste per non ricalcolare la SSIM, che e' il pezzo caro: una gaussiana per
    ogni coppia di fotogrammi. Le colonne "ridimensionata" e "ritaglio" servono a
    confrontare la nitidezza, e la SSIM li' sarebbe scartata comunque.
    """
    frames = read_video_gray(path, target_height, crop)
    if not frames:
        return None, None
    med = float(np.median([sharpness(f) for f in frames]))
    return round(med, 1), "%dx%d" % (frames[0].shape[1], frames[0].shape[0])


def evaluate_clip(path, target_height=None, fps=None, ref_dt=REFERENCE_DT, crop=None):
    """Verdetto su una clip. E' la funzione che runner.py chiama sul pod.

    Restituisce sempre un dizionario, anche in caso di errore: sul pod un gate
    che solleva eccezioni fermerebbe il lotto invece di scartare una clip.

    I DUE PARAMETRI NUOVI SONO OPZIONALI E INERTI SE NON PASSATI. runner.py
    chiama evaluate_clip(path) con un argomento solo, e con quella chiamata le
    chiavi restituite e i loro valori sono identici a prima: le chiavi in piu'
    compaiono solo se si chiede la normalizzazione.

      target_height  altezza a cui portare i frame prima di misurare, aspect
                     ratio preservato. Serve a confrontare modelli che generano
                     a risoluzioni diverse: la varianza del Laplaciano scala con
                     la risoluzione, quindi senza questo i numeri di due modelli
                     non stanno sulla stessa scala.
      fps            fotogrammi al secondo NATIVI della clip. Se dato, si calcola
                     anche ssim_dt_median, cioe' la SSIM fra frame distanti
                     ref_dt secondi invece che un fotogramma.
      crop           (larghezza, altezza) del ritaglio centrale. E' l'altro modo
                     di mettere i modelli sulla stessa scala, e l'unico che non
                     interpola: vedi CROP_SIZE.
    """
    frames = read_video_gray(path, target_height, crop)
    if len(frames) < 2:
        return {"path": path, "n_frames": len(frames), "verdict": "illeggibile",
                "sharpness_median": None, "ssim_median": None, "reason":
                "meno di due frame leggibili"}

    sharp = [sharpness(f) for f in frames]
    sims = [ssim(frames[i - 1], frames[i]) for i in range(1, len(frames))]

    sharp_med = float(np.median(sharp))
    ssim_med = float(np.median(sims))

    # LA NITIDEZZA SI MISURA MA NON GIUDICA, e la ragione non e' rinuncia.
    #
    # Una soglia su questa grandezza deve dire "sotto questo valore la clip
    # danneggia il dataset". Quella frase si puo' verificare, ma con i tassi di
    # annotazione: se le clip piu' morbide producono meno istanze utili, il
    # legame e' misurabile e la soglia esce da li'. Finche' M1 e M4 non ci sono
    # per tutti i modelli, qualunque numero sarebbe scelto e non ricavato.
    #
    # Il valore storico 370 era peggio che arbitrario: calibrato a 720x480, ha
    # marcato "sfocate" tre clip su cinque del bake-off, che stanno fra 0,90 e
    # 1,04 Mpx. La varianza del Laplaciano cambia di 2,6 volte con la scala
    # (vedi CROP_SIZE), quindi confrontava grandezze non commensurabili.
    #
    # Resta il solo criterio della clip congelata, che e' l'unico che non ha
    # bisogno di un riferimento esterno: il suo negativo si fabbrica dalla clip
    # stessa duplicandone i fotogrammi.
    reasons = []
    # La SSIM si controlla sui frame CONSECUTIVI anche quando si conosce ref_dt:
    # ssim_dt e' sempre <= ssim consecutiva, perche' fra i due frame passa piu'
    # tempo, quindi la consecutiva e' il test conservativo per il difetto che
    # questa soglia intercetta, cioe' la clip congelata.
    if ssim_med > SSIM_MAX:
        reasons.append("statica (SSIM %.3f > %.3f)" % (ssim_med, SSIM_MAX))

    out = {
        "path": path,
        "n_frames": len(frames),
        "sharpness_median": round(sharp_med, 1),
        "sharpness_min": round(float(np.min(sharp)), 1),
        "ssim_median": round(ssim_med, 4),
        "ssim_max": round(float(np.max(sims)), 4),
        # Binario, e riguarda solo la clip congelata: "tieni" non vuol dire
        # "clip buona", vuol dire "non e' ferma". Il giudizio sulla nitidezza
        # non c'e' e non e' sottinteso.
        "verdict": "scarta" if reasons else "tieni",
        "reason": "; ".join(reasons),
    }

    if target_height is not None or crop is not None:
        out["altezza_valutata"] = frames[0].shape[0]
        out["larghezza_valutata"] = frames[0].shape[1]

    if fps:
        step = max(1, int(round(fps * ref_dt)))
        sims_dt = [ssim(frames[i - step], frames[i])
                   for i in range(step, len(frames))]
        out["ssim_dt_median"] = round(float(np.median(sims_dt)), 4) if sims_dt else None
        out["ssim_dt_step"] = step
        out["ssim_dt_s"] = round(step / float(fps), 4)

    return out


# --- calibrazione -------------------------------------------------------------

def load_good_frames(directory, target_height, limit):
    """Frame sicuramente buoni, portati all'altezza bersaglio.

    Restituisce DUE popolazioni, e la distinzione non e' un dettaglio:

      sparsi   campionati a passo largo lungo tutta la sequenza, per la
               nitidezza.
               Serve varieta' di scene, e due frame adiacenti sono quasi la
               stessa immagine.
      contigui una corsa di frame consecutivi, per la SSIM. La SSIM misura
               quanto cambia la scena da un fotogramma al successivo: calcolarla
               su frame campionati a passo 7, come faceva la versione
               precedente, misura il movimento di 7 fotogrammi e fa sembrare la
               popolazione buona molto piu' mobile di quanto sia.
    """
    names = sorted(n for n in os.listdir(directory) if n.lower().endswith((".png", ".jpg")))

    def leggi(selezione):
        out = []
        for name in selezione:
            img = cv2.imread(os.path.join(directory, name))
            if img is not None:
                out.append(fit_height(to_gray(img), target_height))
        return out

    step = max(1, len(names) // limit)
    return leggi(names[::step][:limit]), leggi(names[:limit])


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


def calibrate(good_dir, clip_path, target_height, limit, out_dir):
    print("altezza di calibrazione: %s"
          % (target_height or "nativa, nessuna normalizzazione"))
    good, good_contigui = load_good_frames(good_dir, target_height, limit)
    if not good:
        raise SystemExit("nessun frame leggibile in %s" % good_dir)
    clip = read_video_gray(clip_path, target_height)
    if not clip:
        raise SystemExit("clip non leggibile: %s" % clip_path)
    print("frame buoni (render):  %d da %s  (%dx%d)"
          % (len(good), good_dir, good[0].shape[1], good[0].shape[0]))
    print("frame generati:        %d da %s  (%dx%d)"
          % (len(clip), clip_path, clip[0].shape[1], clip[0].shape[0]))

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
    good_ssim += [ssim(good_contigui[i - 1], good_contigui[i])
                  for i in range(1, len(good_contigui))]
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
    print("soglia attualmente applicata: solo SSIM <= %.4f, per la clip"
          " congelata. Sulla nitidezza non c'e' verdetto: il valore storico"
          " %.0f a 720x480 non e' piu' applicato."
          % (SSIM_MAX, SHARPNESS_MIN_STORICA))


# --- modo batch ---------------------------------------------------------------

# TRE COLONNE DI NITIDEZZA, E NON UNA, perche' misurano tre cose diverse e
# nessuna delle tre e' quella giusta in assoluto:
#
#   nativa       i pixel come escono dal modello. Confrontabile solo nella misura
#                in cui le risoluzioni si somigliano - qui stanno fra 0,90 e 1,04
#                Mpx, cioe' un 16 per cento di scarto.
#   ridimension. tutte alla stessa altezza. Mette d'accordo le risoluzioni ma
#                interpola, e l'interpolazione da sola costa il 25-47 per cento
#                (vedi CROP_SIZE): chi e' gia' all'altezza bersaglio non la paga.
#   ritaglio     stesso numero di pixel originali per tutti, nessuna
#                interpolazione. Il prezzo e' che si guarda una porzione di scena
#                leggermente diversa da un modello all'altro.
CSV_COLONNE = ["modello", "clip_id", "n_frames", "fps", "risoluzione_nativa",
               "nitidezza_nativa", "nitidezza_ridimensionata", "nitidezza_ritaglio",
               "ridimensionata_a", "ritaglio_a", "nitidezza_min_nativa",
               "ssim_median", "ssim_max", "ssim_dt_median", "ssim_dt_step",
               "ssim_dt_s", "verdict", "reason", "path"]


def leggi_specs(clips_dir):
    """clip_id -> specifiche native, dal manifest accanto alle clip.

    Gli fps e la risoluzione NON si indovinano dal file video: il contenitore
    mp4 riporta quello che ci ha scritto l'encoder, mentre qui serve il dato
    nativo del modello, che e' l'unico a dire quanto tempo separa davvero due
    fotogrammi. Sta nel manifest, scritto da runner.py al momento della
    generazione.

    Regge due forme: adapter_spec annidato (runner.py) e campi in cima
    (manifest della clip locale dell'11/09, che l'adapter non l'aveva ancora).
    L'ultima riga vince, come in compare_models.read_manifest: un rilancio dopo
    un OOM riscrive l'esito.
    """
    path = os.path.join(clips_dir, "manifest.jsonl")
    specs = {}
    if not os.path.exists(path):
        return specs
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            # Il manifest si scrive in append durante la generazione: se quel
            # processo viene ucciso a meta' riga - watchdog, OOM, pod revocato -
            # l'ultima riga resta JSON parziale. Farla propagare fermerebbe la
            # valutazione anche dei modelli che vengono dopo in ordine
            # alfabetico, e la loro colpa sarebbe di chiamarsi in un certo modo.
            try:
                r = json.loads(line)
            except ValueError as e:
                print("  manifest %s riga %d illeggibile, saltata: %s"
                      % (path, n, e))
                continue
            spec = r.get("adapter_spec") or r
            specs[r.get("clip_id")] = {
                "fps": spec.get("fps"),
                "height": spec.get("height"),
                "width": spec.get("width"),
            }
    return specs


def cartelle_modello(root):
    """Le cartelle da valutare: root stessa se contiene clip, altrimenti le sue
    sottocartelle. Cosi' un solo comando copre tutti i modelli del bake-off."""
    if glob.glob(os.path.join(root, "*.mp4")):
        return [root]
    return [d for d in sorted(glob.glob(os.path.join(root, "*")))
            if os.path.isdir(d) and glob.glob(os.path.join(d, "*.mp4"))]


def run_batch(root, target_height, ref_dt, out_dir, crop=CROP_SIZE):
    cartelle = cartelle_modello(root)
    if not cartelle:
        raise SystemExit("nessuna clip trovata sotto %s" % root)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for cartella in cartelle:
        modello = os.path.basename(os.path.normpath(cartella))
        specs = leggi_specs(cartella)
        clips = [c for c in sorted(glob.glob(os.path.join(cartella, "*.mp4")))
                 if not c.endswith(".tmp.mp4")]
        print("")
        print("%s: %d clip, %d voci nel manifest"
              % (modello, len(clips), len(specs)))

        righe = []
        for clip in clips:
            clip_id = os.path.splitext(os.path.basename(clip))[0]
            spec = specs.get(clip_id, {})
            # La passata completa e' una sola, sui pixel nativi: e' da li' che
            # vengono la SSIM e il verdetto, cosi' nessuna interpolazione entra
            # nel criterio della clip congelata. Le altre due servono solo alla
            # nitidezza.
            nat = evaluate_clip(clip, fps=spec.get("fps"), ref_dt=ref_dt)
            nit_riz, dim_riz = sharpness_only(clip, target_height=target_height)
            nit_rit, dim_rit = sharpness_only(clip, crop=crop)

            r = dict(nat)
            r["nitidezza_nativa"] = nat.get("sharpness_median")
            r["nitidezza_min_nativa"] = nat.get("sharpness_min")
            r["nitidezza_ridimensionata"] = nit_riz
            r["nitidezza_ritaglio"] = nit_rit
            r["ridimensionata_a"] = dim_riz
            r["ritaglio_a"] = dim_rit
            r["modello"] = modello
            r["clip_id"] = clip_id
            r["fps"] = spec.get("fps")
            r["risoluzione_nativa"] = ("%sx%s" % (spec.get("width"), spec.get("height"))
                                       if spec.get("width") else None)
            if not spec.get("fps"):
                r["reason"] = ((r.get("reason") or "") +
                               "; fps assente dal manifest: SSIM a passo fisso non"
                               " calcolata").strip("; ")
            righe.append(r)
            print("  %-18s nit nat %8s  ridim %8s  ritaglio %8s |  SSIM %7s"
                  "  SSIM dt %7s"
                  % (clip_id, r.get("nitidezza_nativa"),
                     r.get("nitidezza_ridimensionata"), r.get("nitidezza_ritaglio"),
                     r.get("ssim_median"), r.get("ssim_dt_median", "-")))

        destinazione = os.path.join(out_dir, modello)
        os.makedirs(destinazione, exist_ok=True)
        csv_path = os.path.join(destinazione, "gate_generated_%s.csv" % ts)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLONNE, extrasaction="ignore",
                               restval="")
            w.writeheader()
            w.writerows(righe)
        print("  -> %s" % csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="Gate di qualita' per le clip generate (Fase B.4).")
    parser.add_argument("--calibrate", action="store_true",
                        help="ricava le soglie confrontando buoni e rovinati")
    parser.add_argument("--good", default=None,
                        help="cartella di frame sicuramente buoni")
    parser.add_argument("--clip", default=None, help="una clip generata")
    parser.add_argument("--clips-dir", default=None,
                        help="cartella di un modello (o la radice di out/): valuta"
                             " tutte le clip e scrive un CSV per modello")
    parser.add_argument("--target-height", type=int, default=None,
                        help="altezza a cui normalizzare prima di misurare, aspect"
                             " ratio preservato. 0 = risoluzione nativa. Default:"
                             " %d in modo batch e in calibrazione, nativa con --clip"
                             % NORMALIZED_HEIGHT)
    parser.add_argument("--ref-dt", type=float, default=REFERENCE_DT,
                        help="secondi fra i due frame della SSIM confrontabile")
    parser.add_argument("--fps", type=float, default=None,
                        help="fps nativi, per la SSIM a passo fisso con --clip"
                             " (in modo batch si leggono dal manifest)")
    parser.add_argument("--limit", type=int, default=40,
                        help="quanti frame buoni campionare")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    # Il default dipende dal modo: --clip da solo deve continuare a comportarsi
    # come prima, cioe' misurare alla risoluzione nativa, perche' e' la chiamata
    # con cui sono stati prodotti i valori storici. Batch e calibrazione invece
    # esistono per confrontare, e senza normalizzazione il confronto non sta in
    # piedi.
    def altezza(default):
        if args.target_height is None:
            return default
        return args.target_height or None

    if args.calibrate:
        if not args.good or not args.clip:
            raise SystemExit("--calibrate richiede --good e --clip")
        calibrate(args.good, args.clip, altezza(NORMALIZED_HEIGHT), args.limit,
                  args.out_dir)
        return

    if args.clips_dir:
        run_batch(args.clips_dir, altezza(NORMALIZED_HEIGHT), args.ref_dt,
                  args.out_dir, CROP_SIZE)
        return

    if not args.clip:
        raise SystemExit("indicare --clip o --clips-dir da valutare, oppure --calibrate")
    result = evaluate_clip(args.clip, target_height=altezza(None), fps=args.fps,
                           ref_dt=args.ref_dt)
    for k, v in result.items():
        print("%-18s %s" % (k + ":", v))


if __name__ == "__main__":
    main()
