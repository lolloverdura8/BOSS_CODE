# build_jobs.py
#
# Scrive jobs.jsonl: una riga per clip da generare, prodotto cartesiano
# modelli x prompt x seed. Lo legge runner.py.
#
# Perche' un file invece di una lista dentro il runner: sul pod il lavoro si
# interrompe e riprende (SIGTERM, pod revocato, notte), e la lista dei job deve
# restare la stessa fra un rilancio e l'altro. Se fosse costruita in memoria da
# una configurazione, cambiare una costante cambierebbe silenziosamente cosa
# manca da fare.
#
#   python build_jobs.py --models wan22_5b,cogvideox15 --set bakeoff
#   python build_jobs.py --models wan22_5b --set produzione --seeds 3
#
# IL SEED NON DIPENDE DAL MODELLO. E' l'indice del prompt nell'ordine stabile di
# prompts.all_prompts(), quindi la stessa scena chiede lo stesso rumore iniziale
# a tutti i modelli: e' una delle variabili controllate del bake-off. L'indice 0
# e' il prompt di monopattino_00, che infatti fu generata con seed 0.
import argparse
import json
import os
import sys

import prompts

ADAPTERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adapters")

# Scostamento fra una replica e l'altra quando si chiedono piu' seed per prompt.
# Largo abbastanza da non sovrapporsi mai agli indici dei prompt (20 in tutto).
SEED_STRIDE = 1000


def known_adapters():
    if not os.path.isdir(ADAPTERS_DIR):
        return []
    return sorted(f[:-3] for f in os.listdir(ADAPTERS_DIR)
                  if f.endswith(".py") and not f.startswith("_"))


def build(models, which, n_seeds):
    pairs = prompts.all_prompts()
    # L'indice globale nell'ordine di all_prompts() e' il seed: si calcola qui e
    # resta lo stesso anche quando il sottoinsieme e' quello del bake-off.
    seed_of = {(cls, i): n for n, (cls, i, _) in enumerate(pairs)}

    selected = prompts.bakeoff_prompts() if which == "bakeoff" else pairs

    jobs = []
    for modello in models:
        for cls, idx, prompt in selected:
            base = seed_of[(cls, idx)]
            for k in range(n_seeds):
                seed = base + k * SEED_STRIDE
                clip_id = "%s_%02d" % (cls, idx) if n_seeds == 1 \
                    else "%s_%02d_s%d" % (cls, idx, k)
                jobs.append({"modello": modello, "clip_id": clip_id, "classe": cls,
                             "prompt_idx": idx, "prompt": prompt, "seed": seed})
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Costruisce jobs.jsonl per runner.py.")
    parser.add_argument("--models", required=True,
                        help="nomi degli adapter separati da virgola, es. wan22_5b,ltx25")
    parser.add_argument("--set", dest="which", choices=("bakeoff", "produzione"),
                        default="bakeoff",
                        help="bakeoff: %d prompt (%d per classe). produzione: tutti e %d"
                             % (len(prompts.bakeoff_prompts()),
                                len(prompts.BAKEOFF_PROMPT_INDICES),
                                len(prompts.all_prompts())))
    parser.add_argument("--seeds", type=int, default=1,
                        help="repliche per prompt, con seed diversi (default 1)")
    parser.add_argument("--limit", type=int, default=None,
                        help="solo i primi N job, per una prova")
    parser.add_argument("--out", default="jobs.jsonl")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        sys.exit("--models vuoto")

    # Un adapter che non esiste va scoperto adesso, non dopo che il pod e' acceso
    # e il runner ha gia' caricato il modello precedente.
    known = known_adapters()
    if known:
        mancanti = [m for m in models if m not in known]
        if mancanti:
            sys.exit("adapter inesistenti: %s. Disponibili: %s"
                     % (", ".join(mancanti), ", ".join(known) or "nessuno"))
    else:
        print("ATTENZIONE: %s non contiene adapter, nessun controllo sui nomi."
              % ADAPTERS_DIR)

    jobs = build(models, args.which, args.seeds)
    if args.limit:
        jobs = jobs[:args.limit]

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    print("%s: %d job" % (args.out, len(jobs)))
    print("  modelli:  %d (%s)" % (len(models), ", ".join(models)))
    print("  prompt:   %d per modello (insieme %s)" % (len(jobs) // len(models), args.which))
    print("  seed:     %d per prompt, da %d a %d"
          % (args.seeds, min(j["seed"] for j in jobs), max(j["seed"] for j in jobs)))


if __name__ == "__main__":
    main()
