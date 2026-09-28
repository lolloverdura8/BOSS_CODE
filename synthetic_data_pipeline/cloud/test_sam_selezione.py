# test_sam_selezione.py
#
# Casi sintetici per sam_selezione.seleziona(), su maschere rettangolari 704x1280
# costruite a mano. Ogni caso riproduce un errore visto sulle clip del bake-off il
# 28/09 (scheda revisione_ramo_20260928.csv): ramo a terra scelto come ramo, ramo
# preso senza il tratto legnoso, supporto su un albero o un palo non collegato.
#
# pytest non e' installato: si esegue con  python cloud/test_sam_selezione.py
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sam_selezione as S

H, W = 704, 1280


def rect(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def cand(mask: np.ndarray, score: float) -> dict:
    ys, xs = np.nonzero(mask)
    box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
    return {"mask": mask, "score": score, "box": box}


def test_ramo_a_terra_non_e_seme():
    a_terra = cand(rect(500, 540, 700, 1000), 0.9)
    sospeso = cand(rect(100, 300, 300, 600), 0.6)
    sel = S.seleziona([a_terra, sospeso], [], [], (H, W))
    assert sel.oggetto is not None
    assert not (sel.oggetto["mask"] & a_terra["mask"]).any()
    assert sel.diag["seme_prompt"] == "principale"


def test_ripiego_sul_ramo_piu_pendente():
    chioma = cand(rect(20, 150, 200, 700), 0.9)
    pendente = cand(rect(100, 450, 400, 460), 0.6)
    sel = S.seleziona([], [chioma, pendente], [], (H, W))
    assert sel.diag["seme_prompt"] == "extra"
    assert (sel.oggetto["mask"] & pendente["mask"]).sum() == pendente["mask"].sum()


def test_foglie_e_legno_che_si_toccano_vengono_uniti():
    foglie = cand(rect(200, 300, 300, 600), 0.7)
    legno = cand(rect(150, 205, 590, 800), 0.8)
    sel = S.seleziona([foglie], [legno], [], (H, W))
    assert sel.diag["n_uniti"] == 1
    assert sel.oggetto["mask"].sum() == (foglie["mask"] | legno["mask"]).sum()
    x0, y0, x1, y1 = sel.oggetto["box"]
    assert (x0, y0, x1, y1) == (300, 150, 800, 300)


def test_candidato_lontano_non_viene_unito():
    foglie = cand(rect(200, 300, 300, 600), 0.7)
    lontano = cand(rect(150, 205, 900, 1000), 0.8)
    sel = S.seleziona([foglie], [lontano], [], (H, W))
    assert sel.diag["n_uniti"] == 0
    assert not (sel.oggetto["mask"] & lontano["mask"]).any()


def test_candidato_enorme_non_viene_unito():
    foglie = cand(rect(200, 300, 300, 600), 0.7)
    chioma = cand(rect(0, 299, 0, 1280), 0.8)
    sel = S.seleziona([foglie], [chioma], [], (H, W))
    assert sel.diag["n_uniti"] == 0


def test_il_tronco_non_finisce_dentro_il_ramo():
    foglie = cand(rect(200, 300, 300, 600), 0.7)
    legno = cand(rect(150, 205, 590, 830), 0.8)
    tronco = cand(rect(100, 704, 800, 860), 0.9)
    sel = S.seleziona([foglie], [legno], [tronco], (H, W))
    assert sel.diag["n_uniti"] == 1
    assert sel.supporto is not None
    assert not (sel.oggetto["mask"] & sel.supporto["mask"]).any()
    assert sel.diag["px_sottratti_supporto"] > 0


def test_palo_a_meta_barra_non_conta():
    barra = cand(rect(200, 220, 200, 1000), 0.9)
    a_meta = cand(rect(220, 704, 590, 610), 0.9)
    sel = S.seleziona([barra], [], [a_meta], (H, W))
    assert sel.supporto is None
    assert sel.diag["criterio_supporto"] == "nessuno"


def test_palo_all_estremo_conta():
    barra = cand(rect(200, 220, 200, 1000), 0.9)
    a_meta = cand(rect(220, 704, 590, 610), 0.95)
    all_estremo = cand(rect(220, 704, 985, 1005), 0.6)
    sel = S.seleziona([barra], [], [a_meta, all_estremo], (H, W))
    assert sel.supporto is not None
    assert (sel.supporto["mask"] == all_estremo["mask"]).all()
    assert sel.diag["criterio_supporto"] == "contatto_estremo"


def test_barra_fra_due_pali_sceglie_il_contatto_maggiore():
    barra = cand(rect(200, 220, 200, 1000), 0.9)
    sinistro = cand(rect(220, 704, 190, 240), 0.5)
    destro = cand(rect(220, 704, 990, 1000), 0.9)
    sel = S.seleziona([barra], [], [sinistro, destro], (H, W))
    assert (sel.supporto["mask"] == sinistro["mask"]).all()
    assert sel.diag["n_supporti_contatto"] == 2


def test_supporto_vicino_ma_non_a_contatto():
    barra = cand(rect(200, 220, 200, 1000), 0.9)
    staccato = cand(rect(240, 704, 985, 1005), 0.9)
    sel = S.seleziona([barra], [], [staccato], (H, W))
    assert sel.supporto is not None
    assert sel.diag["criterio_supporto"] == "vicino"
    assert 15 <= sel.diag["dist_supporto_px"] <= 25


def test_supporto_oltre_la_soglia_non_trovato():
    barra = cand(rect(200, 220, 200, 1000), 0.9)
    lontano = cand(rect(280, 704, 985, 1005), 0.9)
    sel = S.seleziona([barra], [], [lontano], (H, W))
    assert sel.supporto is None


def test_supporto_doppione_dell_oggetto_non_lo_cancella():
    # sospeso di wan22_5b, frame 60-68: "pole" restituisce anche la barra stessa
    barra = cand(rect(99, 126, 297, 919), 0.92)
    doppione = cand(rect(99, 126, 297, 918), 0.50)
    palo = cand(rect(51, 423, 901, 989), 0.92)
    sel = S.seleziona([barra], [], [palo, doppione], (H, W))
    assert sel.oggetto is not None
    assert (sel.supporto["mask"] == palo["mask"]).all()
    assert sel.diag["n_doppioni_supporto"] == 1


def test_oggetto_poco_allungato_accetta_il_supporto_ovunque():
    # una chioma quasi quadrata non ha un "estremo": il tronco che la tocca a
    # meta' del lato basso e' il suo supporto, e il ripiego va dichiarato
    chioma = cand(rect(50, 250, 500, 700), 0.8)
    tronco = cand(rect(250, 704, 590, 610), 0.9)
    sel = S.seleziona([chioma], [], [tronco], (H, W))
    assert sel.supporto is not None
    assert sel.diag["terminale_degenere"] == "poco_allungato"


def test_scarti_dell_unione_contati():
    foglie = cand(rect(200, 300, 300, 600), 0.7)
    enorme = cand(rect(0, 299, 0, 1280), 0.8)
    lontano = cand(rect(150, 205, 900, 1000), 0.8)
    sel = S.seleziona([foglie], [enorme, lontano], [], (H, W))
    assert sel.diag["n_scartati_unione"] == 1
    assert sel.diag["n_uniti"] == 0


def test_candidati_vuoti_ignorati():
    vuoto = {"mask": np.zeros((H, W), bool), "score": 0.99, "box": np.zeros(4, np.float32)}
    foglie = cand(rect(200, 300, 300, 600), 0.7)
    sel = S.seleziona([vuoto, foglie], [vuoto], [vuoto], (H, W))
    assert sel.oggetto is not None and sel.diag["esito_selezione"] == "ok"


def test_nessun_candidato():
    sel = S.seleziona([], [], [], (H, W))
    assert sel.oggetto is None and sel.supporto is None
    assert sel.diag["seme_prompt"] == ""
    assert sel.diag["esito_selezione"] == "nessun_seme"


def test_non_modifica_i_candidati():
    foglie = cand(rect(200, 300, 300, 600), 0.7)
    legno = cand(rect(150, 205, 590, 830), 0.8)
    tronco = cand(rect(100, 704, 800, 860), 0.9)
    copie = [c["mask"].copy() for c in (foglie, legno, tronco)]
    S.seleziona([foglie], [legno], [tronco], (H, W))
    for c, m in zip((foglie, legno, tronco), copie):
        assert (c["mask"] == m).all()


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok     %s" % name)
        except AssertionError as exc:
            failed += 1
            print("FALLITO %s %s" % (name, exc))
    print("\n%d test, %d falliti" % (len(tests), failed))
    sys.exit(1 if failed else 0)
