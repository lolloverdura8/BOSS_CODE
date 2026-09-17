#!/usr/bin/env bash
# setup_pod.sh — prepara un pod RunPod da zero.
#
#   bash setup_pod.sh              tutto
#   bash setup_pod.sh gen ann      solo questi blocchi
#   bash setup_pod.sh --list       cosa fa e basta
#
# Blocchi: base, gen, a14b, ann, opensora, check
#
# TUTTO FINISCE SUL NETWORK VOLUME (/workspace), NIENTE SUL CONTAINER DISK.
# Il container disk sparisce a ogni Stop o Terminate. Il volume di rete no: e'
# per questo che i venv, la cache HuggingFace e il clone del repo stanno li' e
# la seconda sessione parte in un minuto invece che in un'ora.
#
# Prima di lanciarlo, una volta sola:
#   hf auth login
# NON "huggingface-cli login": da huggingface_hub 1.x quel comando e' deprecato e
# si rifiuta di eseguire ("deprecated and no longer works"). I requirements
# pinnano 1.28.0, quindi sul pod fallisce di sicuro, e fallisce nel passo che
# autentica l'account: cioe' prima di qualunque download.
# Con HF_HOME sul volume il token resta li' e tutti i pod successivi che
# montano lo stesso volume lo ritrovano gia' fatto.
#
# ACCESSI DA SISTEMARE PRIMA, ALTRIMENTI SI SCOPRONO A GPU ACCESA:
#   - facebook/sam3 e' gated: accesso gia' approvato per l'account del progetto
#   - Lightricks/LTX-2.5-Diffusers e' gated (auto): le condizioni vanno accettate
#     una volta sulla pagina del modello. Fatto: verificato il 16/09/2026 con
#     l'account lolloverdura8, "hf download ... model_index.json" scarica.
#     Con un altro account va rifatto: il gate segue l'account, non la macchina.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/lolloverdura8/BOSS_CODE.git}"
BRANCH="${BRANCH:-LorenzoV}"

REPO="$WORKSPACE/BOSS_CODE"
CLOUD="$REPO/synthetic_data_pipeline/cloud"
ENVS_SRC="$REPO/synthetic_data_pipeline/envs"
VENVS="$WORKSPACE/envs"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

# Commit misurati in locale. Non sono "l'ultimo main": sono quelli con cui sono
# stati prodotti i numeri di B.1-B.4, e cambiarli significa rimisurare.
SAM3_COMMIT="8f0b7f4d4e7eda2ed606ebde6702c93359ad01da"
DA3_COMMIT="3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"

export HF_HOME="$WORKSPACE/hf"

say() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

want() {
  [ "${#SELECTED[@]}" -eq 0 ] && return 0
  for s in "${SELECTED[@]}"; do [ "$s" = "$1" ] && return 0; done
  return 1
}

mkvenv() {  # mkvenv <nome> <requirements>
  local name="$1" req="$2" dir="$VENVS/$1"
  if [ -d "$dir" ]; then
    echo "venv $name gia' presente, aggiorno soltanto"
  else
    python3 -m venv "$dir"
  fi
  "$dir/bin/pip" install --upgrade pip -q
  "$dir/bin/pip" install -r "$req" --extra-index-url "$TORCH_INDEX"
  echo "venv $name pronto: $dir"
}

SELECTED=()
for a in "$@"; do
  case "$a" in
    --list) sed -n '2,29p' "$0"; exit 0 ;;
    *) SELECTED+=("$a") ;;
  esac
done

# --------------------------------------------------------------------------
if want base; then
  say "base: volume, HF_HOME, repo"
  mkdir -p "$HF_HOME" "$VENVS" "$WORKSPACE/out" "$WORKSPACE/eval"

  # Cosi' vale anche nelle shell aperte dopo, senza doverlo ricordare.
  grep -q 'HF_HOME' ~/.bashrc 2>/dev/null || {
    echo "export HF_HOME=$HF_HOME" >> ~/.bashrc
    echo "export WORKSPACE=$WORKSPACE" >> ~/.bashrc
  }

  if [ -d "$REPO/.git" ]; then
    git -C "$REPO" fetch --quiet origin
    git -C "$REPO" checkout --quiet "$BRANCH"
    git -C "$REPO" pull --quiet
    echo "repo aggiornato: $(git -C "$REPO" log -1 --format='%h %s')"
  else
    git clone --branch "$BRANCH" "$REPO_URL" "$REPO"
    echo "repo clonato in $REPO"
  fi

  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  df -h "$WORKSPACE" | tail -1
fi

# --------------------------------------------------------------------------
if want gen; then
  say "gen: generatori (numpy 2.x, diffusers 0.40.0)"
  mkvenv gen "$ENVS_SRC/requirements-gen.txt"
fi

if want a14b; then
  say "a14b: Wan2.2-T2V-A14B (diffusers pinnato a 37ce5add)"
  mkvenv a14b "$ENVS_SRC/requirements-a14b.txt"
  # Lo SHA e' stato registrato il 17/09/2026 e ora sta nel requirements: questo
  # pip show serve a confermare che l'ambiente monti davvero quel commit.
  "$VENVS/a14b/bin/pip" show diffusers | sed -n '1,3p'
  echo ">>> atteso: 0.41.0.dev0 dal commit 37ce5add"
fi

# --------------------------------------------------------------------------
if want ann; then
  say "ann: SAM 3 + Depth Anything 3 (numpy <2)"
  mkvenv ann "$ENVS_SRC/requirements-ann.txt"

  # I due pacchetti upstream sono install editabili da git: vanno clonati e
  # installati ai commit misurati, non all'ultimo main.
  for spec in "sam3|https://github.com/facebookresearch/sam3.git|$SAM3_COMMIT" \
              "Depth-Anything-3|https://github.com/ByteDance-Seed/Depth-Anything-3.git|$DA3_COMMIT"; do
    IFS='|' read -r name url commit <<< "$spec"
    dir="$WORKSPACE/$name"
    [ -d "$dir/.git" ] || git clone "$url" "$dir"
    git -C "$dir" fetch --quiet origin
    git -C "$dir" checkout --quiet "$commit"
    "$VENVS/ann/bin/pip" install -e "$dir"
    echo "$name @ $(git -C "$dir" rev-parse --short HEAD)"
  done
fi

# --------------------------------------------------------------------------
if want opensora; then
  say "opensora: repo, dipendenze e checkpoint"
  # Open-Sora non e' un repo HuggingFace da scaricare e basta: vuole quattro
  # cose dentro ckpts/ e un ambiente con colossalai. Ha il suo venv perche' le
  # sue dipendenze non sono quelle degli altri generatori.
  dir="$WORKSPACE/Open-Sora"
  [ -d "$dir/.git" ] || git clone https://github.com/hpcaitech/Open-Sora "$dir"
  [ -d "$VENVS/opensora" ] || python3 -m venv "$VENVS/opensora"
  "$VENVS/opensora/bin/pip" install --upgrade pip -q
  echo ">>> installazione secondo il README di Open-Sora (colossalai, flash-attn):"
  echo "    $VENVS/opensora/bin/pip install -v ."
  echo "    eseguirla da $dir, e' lunga e vale la pena guardarla"
  echo ">>> poi popolare $dir/ckpts/ con:"
  echo "    Open_Sora_v2.safetensors, hunyuan_vae.safetensors,"
  echo "    google/t5-v1_1-xxl, openai/clip-vit-large-patch14"
  echo ">>> infine: export OPENSORA_DIR=$dir"
fi

# --------------------------------------------------------------------------
if want check; then
  say "check: cosa c'e' davvero"
  echo "HF_HOME=$HF_HOME  ($(du -sh "$HF_HOME" 2>/dev/null | cut -f1) di checkpoint)"
  for v in gen a14b ann opensora; do
    if [ -x "$VENVS/$v/bin/python" ]; then
      printf '%-9s %s\n' "$v" "$("$VENVS/$v/bin/python" -V 2>&1)"
    else
      printf '%-9s assente\n' "$v"
    fi
  done

  # Che il token ci sia si scopre adesso, non dopo aver acceso l'H100.
  "$VENVS/gen/bin/python" - <<'PY' || true
from huggingface_hub import HfApi
try:
    print("HuggingFace: autenticato come %s" % HfApi().whoami().get("name"))
except Exception as e:
    print("HuggingFace NON autenticato (%s). Lanciare: hf auth login" % e)
PY

  echo
  echo "prossimo passo, una clip per modello:"
  echo "  cd $CLOUD"
  echo "  $VENVS/gen/bin/python build_jobs.py --models wan22_5b --set bakeoff --out $WORKSPACE/jobs.jsonl"
  echo "  $VENVS/gen/bin/python runner.py --jobs $WORKSPACE/jobs.jsonl --out-root $WORKSPACE/out --limit 1"
fi
