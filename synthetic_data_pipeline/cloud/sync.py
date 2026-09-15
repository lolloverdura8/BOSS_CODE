# sync.py
#
# Porta fuori dal pod cio' che il pod ha prodotto, verso un repo dataset
# HuggingFace PRIVATO, e lo riporta giu' sul PC.
#
#   python sync.py push --repo ITS/boss-sintetico --src /workspace/out  --dest out
#   python sync.py push --repo ITS/boss-sintetico --src /workspace/eval --dest eval
#   python sync.py pull --repo ITS/boss-sintetico --dest ./scaricato
#   python sync.py info --repo ITS/boss-sintetico
#
# PERCHE' NON runpodctl E NON git.
# - git: .gitignore esclude gia' outputs/ e *.mp4, e sono ~15-20 GB. Un repo di
#   codice non e' un posto dove mettere un dataset.
# - runpodctl send: peer-to-peer, va bene per un file singolo da guardare
#   subito. Per decine di GB non e' ripartibile: se cade a meta' si ricomincia.
# - dataset HF: upload_large_folder riprende da dove era rimasto, e RunPod non
#   fa pagare l'egress. Il PC lo riprende quando vuole, o mai.
#
# IL REPO VA CREATO PRIVATO E RESTA PRIVATO. Sono dati di progetto: il default
# di --private e' vero e per renderlo pubblico bisogna dirlo a mano, non
# dimenticarselo.
import argparse
import os
import sys

from huggingface_hub import HfApi, snapshot_download

# I .tmp sono clip a meta' scrittura: il runner le rinomina solo quando sono
# complete, quindi caricarle vorrebbe dire pubblicare file troncati.
IGNORE = ["*.tmp", "*.tmp.mp4", "**/__pycache__/**", ".DS_Store"]


def push(api, repo, src, dest, private, large):
    if not os.path.isdir(src):
        sys.exit("sorgente inesistente: %s" % src)
    api.create_repo(repo_id=repo, repo_type="dataset", private=private, exist_ok=True)

    n = sum(len(f) for _, _, f in os.walk(src))
    size = sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(src) for f in fs)
    print("push %s -> %s:%s  (%d file, %.2f GB)" % (src, repo, dest, n, size / 1e9))

    if large:
        # upload_large_folder tiene uno stato sul disco e riprende da solo dopo
        # un'interruzione. E' la modalita' giusta per la produzione.
        api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=src,
                                ignore_patterns=IGNORE)
    else:
        api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=src,
                          path_in_repo=dest, ignore_patterns=IGNORE,
                          commit_message="sync %s" % dest)
    print("fatto: https://huggingface.co/datasets/%s" % repo)


def pull(repo, dest, patterns):
    p = snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest,
                          allow_patterns=patterns or None)
    print("scaricato in %s" % p)


def info(api, repo):
    d = api.dataset_info(repo_id=repo, files_metadata=True)
    files = d.siblings or []
    size = sum(f.size or 0 for f in files)
    print("%s  privato=%s  file=%d  %.2f GB" % (repo, d.private, len(files), size / 1e9))
    per_top = {}
    for f in files:
        top = f.rfilename.split("/")[0]
        per_top[top] = per_top.get(top, 0) + 1
    for k in sorted(per_top):
        print("  %-24s %d file" % (k, per_top[k]))


def main():
    parser = argparse.ArgumentParser(description="Sincronizza i dati generati con un dataset HF privato.")
    parser.add_argument("azione", choices=("push", "pull", "info"))
    parser.add_argument("--repo", required=True, help="es. ITS/boss-sintetico")
    parser.add_argument("--src", default=None, help="push: cartella locale da caricare")
    parser.add_argument("--dest", default=None, help="push: percorso dentro il repo. pull: cartella locale")
    parser.add_argument("--patterns", nargs="*", default=None,
                        help="pull: scarica solo questi glob, es. 'eval/**' '*.csv'")
    parser.add_argument("--public", action="store_true",
                        help="crea il repo pubblico invece che privato (da dire a mano)")
    parser.add_argument("--large", action="store_true",
                        help="push riprendibile per cartelle grandi (produzione)")
    args = parser.parse_args()

    if args.azione == "pull":
        return pull(args.repo, args.dest or "./scaricato", args.patterns)

    # Il token arriva da HF_TOKEN o da huggingface-cli login: con HF_HOME sul
    # network volume, "login" si digita una volta e i pod successivi lo ritrovano.
    api = HfApi()
    try:
        who = api.whoami()
        print("autenticato come %s" % who.get("name"))
    except Exception:
        sys.exit("non autenticato: HF_TOKEN nell'ambiente, oppure huggingface-cli login.")

    if args.azione == "info":
        return info(api, args.repo)
    if not args.src:
        sys.exit("push richiede --src")
    push(api, args.repo, args.src, args.dest or os.path.basename(args.src.rstrip("/\\")),
         not args.public, args.large)


if __name__ == "__main__":
    main()
