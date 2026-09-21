# compare_models.py
#
# Unisce manifest.jsonl, sam_generated_*.csv e da3_generated_*.csv e stampa la
# tabella modello x metriche del bake-off. Di sola lettura: nessuna soglia
# cablata dentro, nessun verdetto. Aggrega e basta.
#
#   python compare_models.py --root /workspace/out --eval-root /workspace/eval
#   python compare_models.py --add wan22_5b <manifest> <sam.csv> <da3.csv>
#
# Solo standard library, cosi' gira in qualunque venv e anche senza.
#
# LE METRICHE
#
#   M1  tasso di individuazione SAM per frame. Riferimento: 54,5%, il tasso con
#       cui SAM trova la bicicletta CARLA in modalita' testo. Dice se il modello
#       ha davvero disegnato la cosa richiesta, in modo riconoscibile.
#   M2  score mediano e area mediana fra i frame trovati.
#   M3  tasso di supporto trovato, solo per le tre classi sospese: un ramo senza
#       tronco o un'insegna senza palo e' una scena fisicamente sbagliata.
#   M4  coerenza di profondita', max(r, 1/r) < 1,25, sui frame VALUTABILI.
#       Un frame non valutabile lo e' perche' SAM non ha trovato nulla, quindi
#       e' gia' escluso da M1: contarlo di nuovo qui lo pesherebbe due volte.
#   M5  nitidezza e SSIM dal gate, VALORI GREZZI. Le soglie di quality_gate.py
#       sono calibrate a 720x480: alle risoluzioni native degli altri modelli
#       non vogliono dire niente come pass/fail.
#   M6  stabilita' temporale: coefficiente di variazione dell'area della
#       maschera SAM dentro una clip, mediano fra le clip. Misura il flicker.
#       Qui non c'e' una sorgente con cui fare IoU, quindi si guarda quanto
#       l'oggetto sfarfalla su se' stesso.
#   M7  secondi e dollari per clip.
#
# LA METRICA CHE DECIDE
#
#   istanze_utili_per_clip = frame_valutati x M1 x M4                      (1)
#   $_per_istanza_utile    = ($/h x secondi_per_clip / 3600) / (1)         (2)
#
# E' il conto del PIANO_ATTACCO_OR4.1 §3, ma con i due fattori misurati invece
# che stimati. Il vincitore e' il valore piu' basso della (2), PURCHE' superi il
# 54,5% su M1 e passi la revisione visiva: il dataset e' il deliverable, non il
# fotogramma.
import argparse
import csv
import glob
import json
import os
import statistics as st

RIFERIMENTO_SAM_CARLA = 54.5

SUSPENDED = ("ramo_sporgente", "insegna_cartello_basso", "ostacolo_sospeso_generico")


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_manifest(path):
    """L'ultima riga vince: un rilancio dopo un OOM riscrive l'esito."""
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r["clip_id"]] = r
    return list(records.values())


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def yes(s):
    return str(s).strip().lower() == "true"


def newest(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


def metrics(manifest, sam_rows, da3_rows, usd_per_hour, gate_rows=None):
    m = {}

    # --- M1, M2, M3 ---------------------------------------------------------
    found = [r for r in sam_rows if yes(r["trovato"])]
    m["n_frame_sam"] = len(sam_rows)
    m["m1_tasso_sam"] = 100.0 * len(found) / len(sam_rows) if sam_rows else None
    scores = [num(r["score"]) for r in found]
    scores = [s for s in scores if s is not None]
    areas = [num(r["area_px"]) for r in found]
    areas = [a for a in areas if a]
    m["m2_score_med"] = st.median(scores) if scores else None
    m["m2_area_med"] = st.median(areas) if areas else None

    sosp = [r for r in sam_rows if r["classe"] in SUSPENDED]
    m["m3_tasso_supporto"] = (100.0 * sum(1 for r in sosp if yes(r["supporto_trovato"]))
                              / len(sosp)) if sosp else None

    # --- M4 -----------------------------------------------------------------
    valutabili = [r for r in da3_rows if not yes(r["non_valutabile"])]
    m["n_frame_da3_valutabili"] = len(valutabili)
    m["m4_coerenza"] = (100.0 * sum(1 for r in valutabili if yes(r["coerente"]))
                        / len(valutabili)) if valutabili else None
    rs = [num(r["r"]) for r in valutabili]
    rs = [x for x in rs if x is not None]
    m["m4_r_med"] = st.median(rs) if rs else None

    # --- M6: flicker, per clip e poi mediana fra le clip ---------------------
    per_clip = {}
    for r in found:
        a = num(r["area_px"])
        if a:
            per_clip.setdefault(r["clip_id"], []).append(a)
    cvs = []
    for areas_clip in per_clip.values():
        if len(areas_clip) >= 3:
            mu = st.mean(areas_clip)
            if mu:
                cvs.append(st.pstdev(areas_clip) / mu)
    m["m6_cv_area_med"] = st.median(cvs) if cvs else None

    # --- M5, M7: dal manifest -----------------------------------------------
    ok = [r for r in manifest if r.get("esito") == "ok"]
    m["n_clip_ok"] = len(ok)
    m["n_clip_oom"] = sum(1 for r in manifest if r.get("esito") == "oom")
    m["n_clip_errore"] = sum(1 for r in manifest if r.get("esito") == "errore")
    tempi = [r["tempo_s"] for r in ok if r.get("tempo_s")]
    m["m7_s_per_clip"] = st.mean(tempi) if tempi else None

    # M5 ha due fonti possibili, e quale sia cambia cosa significa il numero.
    #
    #   il CSV del gate   valori ricalcolati ad altezza normalizzata: la varianza
    #                     del Laplaciano cambia con la risoluzione a cui e'
    #                     calcolata, quindi solo questi sono confrontabili fra
    #                     modelli che generano a risoluzioni diverse. Porta anche
    #                     la SSIM a passo temporale fisso.
    #   il manifest       valori grezzi alla risoluzione nativa, scritti da
    #                     run_gate() al momento della generazione. Confrontabili
    #                     con lo storico, NON fra modelli.
    #
    # Si preferisce il CSV quando c'e'; il manifest resta la rete, cosi' un
    # modello senza gate rigirato non sparisce dalla tabella.
    gate_rows = gate_rows or []
    if gate_rows:
        sharp = [num(r.get("sharpness_median")) for r in gate_rows]
        ssim = [num(r.get("ssim_median")) for r in gate_rows]
        ssim_dt = [num(r.get("ssim_dt_median")) for r in gate_rows]
        m["m5_fonte"] = "gate normalizzato"
        alt = {r.get("altezza_valutata") for r in gate_rows if r.get("altezza_valutata")}
        m["m5_altezza_valutata"] = alt.pop() if len(alt) == 1 else None
        dt = [x for x in ssim_dt if x is not None]
        m["m5_ssim_dt_med"] = st.median(dt) if dt else None
        m["m5_scartate_dal_gate"] = sum(1 for r in gate_rows
                                        if r.get("verdict") == "scarta")
    else:
        sharp = [r["gate_sharpness"] for r in ok if r.get("gate_sharpness") is not None]
        ssim = [r["gate_ssim"] for r in ok if r.get("gate_ssim") is not None]
        m["m5_fonte"] = "manifest, risoluzione nativa"
        m["m5_altezza_valutata"] = None
        m["m5_ssim_dt_med"] = None
        m["m5_scartate_dal_gate"] = None
    sharp = [x for x in sharp if x is not None]
    ssim = [x for x in ssim if x is not None]
    # UNA DEFINIZIONE SOLA, in entrambi i rami: le clip con un valore di
    # nitidezza utilizzabile. Contare invece le righe del CSV includerebbe anche
    # le clip illeggibili, e un modello sembrerebbe avere piu' campioni di
    # un altro solo per aver fallito di piu'. Finisce in --json-out, che e'
    # l'output su cui si fanno i confronti fini.
    m["n_clip_gate"] = len(sharp)
    m["m5_sharpness_med"] = st.median(sharp) if sharp else None
    m["m5_ssim_med"] = st.median(ssim) if ssim else None

    picchi = [r["vram_alloc_peak_gb"] for r in ok if r.get("vram_alloc_peak_gb")]
    m["vram_peak_max"] = max(picchi) if picchi else None

    # $/istanza_utile dipende dai secondi per clip, una proprieta' della GPU:
    # le otto clip di un modello vanno misurate sulla stessa scheda, altrimenti
    # m7_s_per_clip e' una media fra hardware diversi e il confronto e' falsato
    # senza che il numero lo segnali. Le clip precedenti a questo campo (17/09)
    # non hanno "gpu_model": contano come sconosciute, non come un terzo valore.
    gpu_visti = {r["gpu_model"] for r in ok if r.get("gpu_model")}
    if len(gpu_visti) == 1:
        m["gpu_model"] = gpu_visti.pop()
    elif len(gpu_visti) > 1:
        m["gpu_model"] = "MISTE: %s" % ", ".join(sorted(gpu_visti))
    else:
        m["gpu_model"] = None

    # --- (1) e (2) ----------------------------------------------------------
    n_eval = ok[0].get("eval_frames") if ok and ok[0].get("eval_frames") else None
    if n_eval is None and manifest:
        # le clip prodotte prima del runner non hanno eval_frames: si ricava dai
        # frame per clip effettivamente valutati da SAM
        n_eval = len(sam_rows) / len(set(r["clip_id"] for r in sam_rows)) if sam_rows else None
    m["frame_valutati"] = n_eval

    if None not in (n_eval, m["m1_tasso_sam"], m["m4_coerenza"]):
        m["istanze_utili_per_clip"] = n_eval * m["m1_tasso_sam"] / 100.0 * m["m4_coerenza"] / 100.0
    else:
        m["istanze_utili_per_clip"] = None

    if m["istanze_utili_per_clip"] and m["m7_s_per_clip"] and usd_per_hour:
        usd_clip = usd_per_hour * m["m7_s_per_clip"] / 3600.0
        m["usd_per_clip"] = usd_clip
        m["usd_per_istanza_utile"] = usd_clip / m["istanze_utili_per_clip"]
    else:
        m["usd_per_clip"] = None
        m["usd_per_istanza_utile"] = None
    return m


def fmt(v, spec="%.1f"):
    return "-" if v is None else spec % v


def render(results, usd_per_hour):
    print("\n=== BAKE-OFF: %d modelli ===" % len(results))
    print("frame valutati per clip, tasso SAM e coerenza DA3 sono le tre quantita' che"
          " entrano nella (1).\n")

    cols = [
        ("modello", "%-16s", lambda k, m: k[:16]),
        ("clip ok", "%7s", lambda k, m: "%d" % m["n_clip_ok"]),
        ("s/clip", "%8s", lambda k, m: fmt(m["m7_s_per_clip"])),
        ("M1 SAM%", "%8s", lambda k, m: fmt(m["m1_tasso_sam"])),
        ("M2 score", "%9s", lambda k, m: fmt(m["m2_score_med"], "%.3f")),
        ("M3 sup%", "%8s", lambda k, m: fmt(m["m3_tasso_supporto"])),
        ("M4 coer%", "%9s", lambda k, m: fmt(m["m4_coerenza"])),
        ("M4 r med", "%9s", lambda k, m: fmt(m["m4_r_med"], "%.3f")),
        ("M5 nit.", "%8s", lambda k, m: fmt(m["m5_sharpness_med"], "%.0f")),
        ("M5 SSIM", "%8s", lambda k, m: fmt(m["m5_ssim_med"], "%.3f")),
        ("M5 SSIMdt", "%10s", lambda k, m: fmt(m.get("m5_ssim_dt_med"), "%.3f")),
        ("M6 cv", "%7s", lambda k, m: fmt(m["m6_cv_area_med"], "%.3f")),
        ("ist/clip", "%9s", lambda k, m: fmt(m["istanze_utili_per_clip"], "%.2f")),
        ("$/ist.", "%8s", lambda k, m: fmt(m["usd_per_istanza_utile"], "%.4f")),
        ("GPU", " %-18s", lambda k, m: (m["gpu_model"] or "-")[:18]),
    ]
    head = "".join(f % h for h, f, _ in cols)
    print(head)
    print("-" * len(head))
    for k, m in results.items():
        print("".join(f % g(k, m) for _, f, g in cols))

    miste = [k for k, m in results.items()
             if isinstance(m.get("gpu_model"), str) and m["gpu_model"].startswith("MISTE")]
    if miste:
        print("\n!!! %s: clip ok generate su schede diverse. s/clip e $/ist. sono una"
              " media fra hardware diversi e NON vanno confrontati con gli altri"
              " modelli. Rigenerare sulla stessa GPU prima di trarre conclusioni."
              % ", ".join(miste))

    nativi = [k for k, m in results.items()
              if m.get("m5_fonte") == "manifest, risoluzione nativa"]
    if nativi:
        print("")
        print("!!! %s: M5 viene dal manifest, cioe' dalla risoluzione NATIVA di ogni"
              " modello. La varianza del Laplaciano cambia con la risoluzione a cui"
              " e' calcolata - misurato 2,6 volte sulla stessa clip, e non in modo"
              " monotono - quindi quelle nitidezze non si confrontano fra righe."
              " Rigirare quality_gate.py --clips-dir per averle normalizzate."
              % ", ".join(nativi))

    print("\nRiferimento M1: %.1f%% (bicicletta CARLA in modalita' testo, B.1-B.4)."
          % RIFERIMENTO_SAM_CARLA)
    sopra = [k for k, m in results.items()
             if m["m1_tasso_sam"] is not None and m["m1_tasso_sam"] >= RIFERIMENTO_SAM_CARLA]
    print("Sopra il riferimento: %s" % (", ".join(sopra) if sopra else "nessuno"))
    if not usd_per_hour:
        print("$/ist. vuota: passare --usd-per-hour con la tariffa del pod usato.")
    print("\nLa (2) non decide da sola: il vincitore deve anche passare la revisione"
          " visiva degli overlay e dei depth_vis.")


def discover(root, eval_root):
    """Un modello per sottocartella di --root, con i CSV omonimi sotto --eval-root."""
    out = []
    for name in sorted(os.listdir(root)):
        manifest = os.path.join(root, name, "manifest.jsonl")
        if not os.path.exists(manifest):
            continue
        ev = os.path.join(eval_root, name)
        out.append((name, manifest,
                    newest(os.path.join(ev, "sam_generated_*.csv")),
                    newest(os.path.join(ev, "da3_generated_*.csv")),
                    newest(os.path.join(ev, "gate_generated_*.csv"))))
    return out


def main():
    parser = argparse.ArgumentParser(description="Tabella modello x metriche del bake-off.")
    parser.add_argument("--root", default=None, help="cartella con una sottocartella per modello")
    parser.add_argument("--eval-root", default=None, help="idem, per i CSV di SAM e DA3")
    parser.add_argument("--add", nargs=4, action="append", metavar=("NOME", "MANIFEST", "SAM_CSV", "DA3_CSV"),
                        help="un modello con i suoi tre file espliciti; ripetibile")
    parser.add_argument("--usd-per-hour", type=float, default=0.0)
    parser.add_argument("--json-out", default=None, help="scrive anche le metriche grezze")
    args = parser.parse_args()

    # --add porta tre file espliciti; il quarto, il CSV del gate, lo trova solo
    # discover(). Si completa con None per tenere una forma sola.
    entries = [tuple(a) + (None,) for a in (args.add or [])]
    if args.root:
        entries += discover(args.root, args.eval_root or args.root)
    if not entries:
        parser.error("serve --root oppure almeno un --add")

    def leggi(path):
        return read_csv(path) if path and os.path.exists(path) else []

    results = {}
    for name, manifest, sam_csv, da3_csv, gate_csv in entries:
        sam_rows, da3_rows = leggi(sam_csv), leggi(da3_csv)
        gate_rows = leggi(gate_csv)
        # UN MODELLO SENZA SAM E DA3 NON SI SALTA PIU'. Le sue M1-M4 e M6 restano
        # vuote - metrics() ha una guardia per ognuna - ma M5 e M7 esistono lo
        # stesso, e sono esattamente le due colonne che servono per confrontare
        # la qualita' dei pixel e il costo. Saltare la riga nascondeva dati che
        # c'erano gia' nel manifest.
        mancanti = [n for n, r in (("SAM", sam_rows), ("DA3", da3_rows)) if not r]
        if mancanti:
            print("%s: senza CSV di %s, in tabella solo M5 e M7"
                  % (name, " e ".join(mancanti)))
        results[name] = metrics(read_manifest(manifest), sam_rows, da3_rows,
                                args.usd_per_hour, gate_rows)
        results[name]["_files"] = {"manifest": manifest, "sam": sam_csv,
                                   "da3": da3_csv, "gate": gate_csv}

    if not results:
        parser.error("nessun modello con dati completi")

    render(results, args.usd_per_hour)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("\nmetriche grezze: %s" % args.json_out)


if __name__ == "__main__":
    main()
