# sam_selezione.py
#
# Sceglie oggetto e supporto fra TUTTI i candidati che SAM 3 restituisce per i
# prompt testuali di una classe sospesa (ramo, insegna, ostacolo generico).
#
# Prima del 28/09/2026 annotate_sam.py prendeva, per oggetto e supporto, l'istanza
# di score massimo nell'intera immagine, in modo indipendente. Sulla scheda manuale
# revisione_ramo_20260928.csv (57 frame, ramo visibile in 57) la maschera era
# completa in 4 e il supporto giusto in 4: un ramo caduto a terra preso per ramo,
# le sole foglie senza il tratto legnoso, un tronco di un altro albero come supporto.
# Qui la scelta usa la posizione e il contatto, non solo lo score.
#
# Solo numpy e scipy.ndimage (presente anche nel venv del pod, scikit-image no):
# gira su CPU e si puo' rieseguire sui candidati salvati senza ricaricare SAM.
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

# Un candidato interamente nella meta' bassa del frame giace a terra: non e' un
# ostacolo sospeso. Scarta il ramo caduto di ramo_sporgente_00_s1 (bordo alto a
# 0,64 H) e lascia passare la fronda di ramo_sporgente_00 (bordo alto a 0).
SEME_Y1_MAX_FRAC = 0.5

# "Tocca" = distanza <= 6 px, circa lo 0,5 % della larghezza di 1280.
ADIACENZA_PX = 6

# Un candidato da unire non puo' superare 3 volte l'area del seme: evita che un
# ramo si mangi la chioma intera.
AREA_MAX_SU_SEME = 3.0

# Un candidato che perde oltre meta' dei pixel quando si tolgono i supporti e'
# per lo piu' tronco o palo, non oggetto.
QUOTA_MIN_DOPO_SUPPORTI = 0.5

# Oltre questa distanza dall'oggetto il supporto e' "non trovato": meglio un
# supporto mancante che uno non collegato, che falsa M4 in annotate_da3.py.
SUPPORTO_DIST_MAX_PX = 40

# Il supporto regge l'oggetto a un estremo: il contatto conta solo se cade nel 15 %
# terminale dell'asse principale dell'oggetto. Un lampione davanti a meta' barra
# non la sostiene.
ESTREMO_FRAC = 0.15

# Oltre questo numero di pixel l'asse principale si stima su un sottocampione.
PCA_MAX_PUNTI = 20000

# Sotto questo rapporto fra i due valori singolari l'oggetto non e' allungato (una
# chioma, una fronda larga): non ha un estremo a cui attaccarsi, e il contatto vale
# ovunque. Il ripiego si dichiara nel CSV, colonna terminale_degenere.
ALLUNGAMENTO_MIN = 2.0

# Un candidato di supporto con IoU oltre 0,5 con un candidato dell'oggetto e' la
# stessa istanza vista da due prompt, non un supporto. Sospeso di wan22_5b, frame
# 60-68: "pole" restituiva anche la barra (score 0,50), e togliere i supporti
# cancellava l'oggetto intero.
IOU_DOPPIONE = 0.5

# Un supporto che l'oggetto scavalca da un bordo all'altro gli sta davanti o dietro,
# non lo regge. La quota e' il numero di colonne in cui il supporto sporge sopra E
# sotto l'oggetto, diviso per la larghezza del supporto appena fuori dall'oggetto
# (fascia di BANDA_LARGHEZZA_PX righe). Supporti verticali, come tronchi e pali.
# Misurata il 28/09 sui candidati entro 40 px: il tronco dietro di
# ramo_sporgente_00 (wan22_5b), che il ramo attraversa, sta fra 0,91 e 0,94 in tutti i
# 19 frame; i supporti giudicati giusti a occhio fra 0 e 0,29 (palo del sospeso di
# ltx25 al frame 33, dove la barra finisce dentro il palo). Nessuna ragione per
# preferire un punto del vuoto fra i due: 0,6.
# Limite voluto: se l'oggetto finisce dentro la larghezza del supporto la quota e'
# bassa e il supporto resta idoneo, anche quando in realta' e' un tronco dietro. In
# 2D quel caso non si distingue da un ramo che esce dal tronco o da una barra che
# finisce contro il palo (ltx25, frame 33), che vanno tenuti.
QUOTA_SCAVALCATO = 0.6
BANDA_LARGHEZZA_PX = 10


@dataclass(frozen=True)
class Selezione:
    oggetto: dict | None
    supporto: dict | None
    diag: dict


def box_da_maschera(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


def _bordi_y(mask: np.ndarray) -> tuple[int, int]:
    rows = np.flatnonzero(mask.any(axis=1))
    return int(rows[0]), int(rows[-1]) + 1


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    unione = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / unione) if unione else 0.0


def _senza_doppioni(cand_sup: list, cand_oggetto: list) -> tuple[list, int]:
    tenuti = [s for s in cand_sup
              if all(_iou(s["mask"], o["mask"]) <= IOU_DOPPIONE for o in cand_oggetto)]
    return tenuti, len(cand_sup) - len(tenuti)


def _pulito(c: dict, tutti_supporti: np.ndarray) -> np.ndarray | None:
    """Il candidato senza i pixel dei supporti, o None se era per lo piu' supporto."""
    area = int(c["mask"].sum())
    resto = c["mask"] & ~tutti_supporti
    if area == 0 or resto.sum() < QUOTA_MIN_DOPO_SUPPORTI * area:
        return None
    return resto


def _scegli_seme(cand_obj, cand_extra, tutti_supporti, H):
    """(maschera pulita, candidato, "principale"|"extra") oppure (None, None, "")."""
    y1_max = SEME_Y1_MAX_FRAC * H
    principali = sorted(cand_obj, key=lambda c: -c["score"])
    # nel ripiego si preferisce il ramo che scende di piu': il nudo di LTX
    # scende a 0,68 H, i rami della chioma restano in alto
    extra = sorted(cand_extra, key=lambda c: -_bordi_y(c["mask"])[1])
    for gruppo, nome in ((principali, "principale"), (extra, "extra")):
        for c in gruppo:
            m = _pulito(c, tutti_supporti)
            if m is not None and _bordi_y(m)[0] <= y1_max:
                return m, c, nome
    return None, None, ""


def _unisci(seme: np.ndarray, altri: list, tutti_supporti: np.ndarray):
    """Seme + i candidati dell'altro prompt che lo toccano. Un passo solo, non a
    catena, altrimenti la chioma intera entra per contiguita'."""
    dist = distance_transform_edt(~seme)
    area_max = AREA_MAX_SU_SEME * seme.sum()
    unione, originali = seme.copy(), np.zeros_like(seme)
    n_uniti = n_scartati = 0
    for c in altri:
        m = _pulito(c, tutti_supporti)
        if m is None or m.sum() > area_max:
            n_scartati += 1
            continue
        if dist[m].min() <= ADIACENZA_PX:
            unione |= m
            originali |= c["mask"]
            n_uniti += 1
    return unione, originali, n_uniti, n_scartati


def _tutto_terminale(p: np.ndarray) -> np.ndarray:
    return np.ones(len(p), dtype=bool)


def _zona_terminale(obj: np.ndarray):
    """(funzione punti (N,2 in y,x) -> bool, ripiego). Il punto e' terminale se
    cade nel 15 % estremo dell'asse principale dell'oggetto, o oltre. Per un
    oggetto puntiforme o poco allungato tutto e' terminale, e il ripiego si dice."""
    pts = np.argwhere(obj).astype(np.float64)
    if len(pts) > PCA_MAX_PUNTI:
        pts = pts[np.linspace(0, len(pts) - 1, PCA_MAX_PUNTI).astype(int)]
    if len(pts) < 2:
        return _tutto_terminale, "puntiforme"
    centro = pts.mean(axis=0)
    _, s, vt = np.linalg.svd(pts - centro, full_matrices=False)
    if s[1] > 0 and s[0] / s[1] < ALLUNGAMENTO_MIN:
        return _tutto_terminale, "poco_allungato"
    asse = vt[0]
    t = (pts - centro) @ asse
    t_min, t_max = t.min(), t.max()
    if t_max - t_min < 1.0:
        return _tutto_terminale, "puntiforme"
    margine = ESTREMO_FRAC * (t_max - t_min)

    def terminale(p):
        tp = (p.astype(np.float64) - centro) @ asse
        return (tp <= t_min + margine) | (tp >= t_max - margine)
    return terminale, ""


def _quota_scavalcata(obj: np.ndarray, sup: np.ndarray) -> float:
    """Colonne in cui sup sporge sopra e sotto obj, sulla larghezza di sup appena
    fuori da obj. 0 se obj non lo attraversa in nessuna colonna."""
    cols = np.flatnonzero(obj.any(axis=0) & sup.any(axis=0))
    scavalcate = []
    for x in cols:
        yo, ys = np.flatnonzero(obj[:, x]), np.flatnonzero(sup[:, x])
        if ys[0] < yo[0] and ys[-1] > yo[-1]:
            scavalcate.append(x)
    if not scavalcate:
        return 0.0
    rows = np.flatnonzero(obj[:, scavalcate].any(axis=1))
    y0, y1 = rows[0], rows[-1] + 1
    larghezza = max(int(sup[max(0, y0 - BANDA_LARGHEZZA_PX):y0].any(axis=0).sum()),
                    int(sup[y1:y1 + BANDA_LARGHEZZA_PX].any(axis=0).sum()))
    if larghezza == 0:
        # sup sporge sopra e sotto ma con un buco oltre la fascia su entrambi i lati
        # (bordi erosi dall'occlusione): la larghezza non si misura, e il candidato
        # resta idoneo come prima della regola
        return 0.0
    return len(scavalcate) / larghezza


def _scegli_supporto(obj: np.ndarray, cand_sup: list):
    """(candidato, criterio, distanza, n a contatto, ripiego, n scavalcati) sulla
    maschera finale."""
    if not cand_sup:
        return None, "nessuno", None, 0, "", 0
    dist = distance_transform_edt(~obj)
    terminale, ripiego = _zona_terminale(obj)
    idonei = []
    n_scavalcati = 0
    for c in cand_sup:
        d = float(dist[c["mask"]].min())
        if d > SUPPORTO_DIST_MAX_PX:
            continue
        if _quota_scavalcata(obj, c["mask"]) >= QUOTA_SCAVALCATO:
            n_scavalcati += 1
            continue
        vicini = np.argwhere(c["mask"] & (dist <= d + ADIACENZA_PX))
        n_term = int(terminale(vicini).sum())
        if n_term:
            idonei.append((c, d, n_term))
    contatto = [x for x in idonei if x[1] <= ADIACENZA_PX]
    if contatto:
        c, d, _ = max(contatto, key=lambda x: x[2])
        return c, "contatto_estremo", d, len(contatto), ripiego, n_scavalcati
    if idonei:
        c, d, _ = min(idonei, key=lambda x: x[1])
        return c, "vicino", d, 0, ripiego, n_scavalcati
    return None, "nessuno", None, 0, ripiego, n_scavalcati


def _non_vuoti(cands: list) -> list:
    """Le funzioni sopra assumono maschere non vuote: all_detections le scarta gia',
    ma i candidati possono anche arrivare da file con --riseleziona."""
    return [c for c in cands if c["mask"].any()]


def seleziona(cand_obj: list, cand_extra: list, cand_sup: list,
              shape: tuple[int, int]) -> Selezione:
    """Ordine: seme -> unione (senza i pixel di tutti i supporti) -> supporto sulla
    maschera unita -> oggetto privato del supporto scelto. Il supporto si cerca
    dopo l'unione perche' e' il tratto legnoso, non le foglie, a toccare il tronco.
    Le maschere dei candidati non vengono modificate."""
    H, W = shape
    cand_obj, cand_extra, cand_sup = (_non_vuoti(cand_obj), _non_vuoti(cand_extra),
                                      _non_vuoti(cand_sup))
    n_cand_sup = len(cand_sup)
    cand_sup, n_doppioni = _senza_doppioni(cand_sup, cand_obj + cand_extra)
    tutti_supporti = np.zeros((H, W), dtype=bool)
    for c in cand_sup:
        tutti_supporti |= c["mask"]
    diag = {"esito_selezione": "nessun_seme",
            "n_cand_oggetto": len(cand_obj), "n_cand_extra": len(cand_extra),
            "n_cand_supporto": n_cand_sup, "n_doppioni_supporto": n_doppioni,
            "seme_prompt": "", "seme_y1": "", "seme_y2": "", "area_seme": 0,
            "area_finale": 0, "n_uniti": 0, "n_scartati_unione": 0,
            "px_sottratti_supporto": 0, "n_supporti_contatto": 0,
            "n_supporti_scavalcati": 0,
            "criterio_supporto": "nessuno", "dist_supporto_px": "",
            "terminale_degenere": ""}

    seme, c_seme, nome = _scegli_seme(cand_obj, cand_extra, tutti_supporti, H)
    if seme is None:
        return Selezione(None, None, diag)
    altri = cand_extra if nome == "principale" else cand_obj
    unione, originali, n_uniti, n_scartati = _unisci(seme, altri, tutti_supporti)
    # tutti i pixel dei candidati scelti che non arrivano nella maschera finale
    # sono stati tolti perche' supporto
    originali |= c_seme["mask"]

    sup, criterio, d, n_contatto, ripiego, n_scavalcati = _scegli_supporto(unione, cand_sup)
    finale = unione & ~sup["mask"] if sup else unione.copy()
    y1, y2 = _bordi_y(seme)
    diag.update({"esito_selezione": "ok" if finale.any() else "cancellato_dal_supporto",
                 "seme_prompt": nome, "seme_y1": y1, "seme_y2": y2,
                 "area_seme": int(seme.sum()), "area_finale": int(finale.sum()),
                 "n_uniti": n_uniti, "n_scartati_unione": n_scartati,
                 "px_sottratti_supporto": int(originali.sum() - finale.sum()),
                 "n_supporti_contatto": n_contatto, "n_supporti_scavalcati": n_scavalcati,
                 "criterio_supporto": criterio,
                 "dist_supporto_px": round(d, 1) if d is not None else "",
                 "terminale_degenere": ripiego})
    if not finale.any():
        return Selezione(None, None, diag)
    oggetto = {"mask": finale, "score": float(c_seme["score"]), "box": box_da_maschera(finale)}
    supporto = ({"mask": sup["mask"], "score": float(sup["score"]), "box": sup["box"]}
                if sup else None)
    return Selezione(oggetto, supporto, diag)
