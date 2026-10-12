#!/usr/bin/env bash
# setup_pod.sh — prepara un pod RunPod da zero.
#
#   bash setup_pod.sh              tutto
#   bash setup_pod.sh gen ann      solo questi blocchi
#   bash setup_pod.sh --list       cosa fa e basta
#
# Blocchi: base, gen, a14b, ann, opensora, cosmos, fun, check
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
#   - Cosmos-Transfer2.5 (blocco cosmos): nvidia/Cosmos-Transfer2.5-2B,
#     nvidia/Cosmos-Guardrail1, nvidia/Cosmos-Reason1-7B, nvidia/Cosmos-Predict2.5-2B.
#     Il blocco check prova l'accesso a tutti e quattro.
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
# Ramo Sim2Real, commit letti il 12/10/2026 (piano Fase 3): sono quelli su cui sono
# stati scritti gli adapter cosmos_t25 e wan22_fun5b_ctrl.
COSMOS_COMMIT="2ff49d0717af02057ae79bc75c00fbff9da1b4e7"
VIDEOX_COMMIT="4b7b6402a1e0f0406bd6801fb66c0a00bd922621"
COSMOS_GATED="nvidia/Cosmos-Transfer2.5-2B nvidia/Cosmos-Guardrail1 nvidia/Cosmos-Reason1-7B nvidia/Cosmos-Predict2.5-2B"

# Com'era la variabile PRIMA che la sovrascrivessimo: serve al blocco check
# per accorgersi che la shell interattiva punta altrove.
HF_HOME_EREDITATO="${HF_HOME:-(non impostata)}"
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
    --list) sed -n '2,32p' "$0"; exit 0 ;;
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
if want cosmos; then
  say "cosmos: Cosmos-Transfer2.5 (repo NVIDIA, ambiente uv proprio)"
  # Ha un ambiente suo, creato da uv dal repo, e l'adapter cosmos_t25 lo chiama in
  # un sottoprocesso. uv, la sua cache e i python che scarica vanno sul volume: coi
  # default finirebbero in /root, sul container disk, e il .venv punterebbe a un
  # interprete che dopo il Terminate non esiste piu'. TMPDIR sul volume per la
  # trappola "Invalid cross-device link" gia' vista con Open-Sora.
  dir="$WORKSPACE/cosmos-transfer2.5"
  export UV_CACHE_DIR="$WORKSPACE/uv-cache" UV_PYTHON_INSTALL_DIR="$WORKSPACE/uv-python"
  export TMPDIR="$WORKSPACE/tmp"
  mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$TMPDIR"
  # docs/setup.md del repo chiede ffmpeg di sistema; sta sul container disk, quindi
  # va rimesso a ogni pod nuovo.
  command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg; }
  if [ ! -x "$VENVS/uv/bin/uv" ]; then
    python3 -m venv "$VENVS/uv"
    "$VENVS/uv/bin/pip" install -q uv
  fi
  [ -d "$dir/.git" ] || git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5.git "$dir"
  git -C "$dir" fetch --quiet origin
  git -C "$dir" checkout --quiet "$COSMOS_COMMIT"
  (cd "$dir" && "$VENVS/uv/bin/uv" sync --extra=cu128)
  # HF_HUB_ENABLE_HF_TRANSFER e' nell'immagine di RunPod: senza il pacchetto il
  # primo download dei checkpoint fallisce.
  "$VENVS/uv/bin/uv" pip install --python "$dir/.venv/bin/python" hf_transfer
  "$dir/.venv/bin/python" -c "import cosmos_transfer2; print('cosmos_transfer2 importabile')"
  echo "cosmos-transfer2.5 @ $(git -C "$dir" rev-parse --short HEAD)"
  echo ">>> i checkpoint si scaricano alla prima clip in \$HF_HOME: servono le licenze"
  echo "    di $COSMOS_GATED (vedi check)"
fi

# --------------------------------------------------------------------------
if want fun; then
  say "fun: Wan2.2-Fun-5B-Control (VideoX-Fun)"
  # VideoX-Fun non e' un pacchetto pip: l'adapter mette il clone nel sys.path e
  # gira dentro runner.py, lanciato col python di questo venv.
  dir="$WORKSPACE/VideoX-Fun"
  export TMPDIR="$WORKSPACE/tmp"
  mkdir -p "$TMPDIR" "$WORKSPACE/models"
  [ -d "$dir/.git" ] || git clone https://github.com/aigc-apps/VideoX-Fun.git "$dir"
  git -C "$dir" fetch --quiet origin
  git -C "$dir" checkout --quiet "$VIDEOX_COMMIT"
  [ -d "$VENVS/fun" ] || python3 -m venv "$VENVS/fun"
  pip="$VENVS/fun/bin/pip"
  "$pip" install --upgrade pip -q
  # torch pinnato come nel venv gen: requirements.txt di VideoX-Fun chiede solo
  # torch>=2.1.2, e lascerebbe scegliere a pip una build qualsiasi.
  printf 'torch==2.11.0+cu128\ntorchvision==0.26.0+cu128\n' > "$TMPDIR/fun-constraints.txt"
  "$pip" install -r "$dir/requirements.txt" -c "$TMPDIR/fun-constraints.txt" \
    --extra-index-url "$TORCH_INDEX"
  # opencv-python vuole libGL, che sul pod non c'e' (stessa scelta del venv gen).
  # Si tolgono entrambe le varianti prima, perche' condividono la cartella cv2.
  "$pip" uninstall -y opencv-python opencv-python-headless
  "$pip" install opencv-python-headless==4.11.0.86 psutil==7.2.2 hf_transfer
  "$pip" freeze > "$VENVS/requirements-fun.lock"
  echo "versioni installate in $VENVS/requirements-fun.lock (da riportare nel repo)"
  # Prima l'import, poi i 24 GB di pesi: diffusers e transformers non sono pinnati
  # da VideoX-Fun, e una rottura va vista prima del download, non dopo.
  "$VENVS/fun/bin/python" -c "import sys; sys.path.insert(0, '$dir');
import videox_fun.pipeline; print('videox_fun importabile')"
  "$VENVS/fun/bin/hf" download alibaba-pai/Wan2.2-Fun-5B-Control \
    --local-dir "$WORKSPACE/models/Wan2.2-Fun-5B-Control"
  echo "VideoX-Fun @ $(git -C "$dir" rev-parse --short HEAD)"
fi

# --------------------------------------------------------------------------
if want check; then
  say "check: cosa c'e' davvero"
  echo "HF_HOME=$HF_HOME  ($(du -sh "$HF_HOME" 2>/dev/null | cut -f1) di checkpoint)"

  # QUESTO CONTROLLO ESISTE PERCHE' E' GIA' COSTATO. Il 17/09/2026, su un pod
  # nuovo, la shell aveva HF_HOME=/workspace/.cache/huggingface: il token salvato
  # in /workspace/hf non veniva trovato, LTX-2.5 (gated) rispondeva 401 e i
  # checkpoint gia' scaricati venivano ripresi da capo in una seconda cache.
  # La ragione e' che l'export vive in ~/.bashrc, cioe' in /root, cioe' SUL
  # CONTAINER DISK, che il Terminate cancella: il token sopravvive sul volume, la
  # variabile che dice dove cercarlo no. Va riesportata a ogni pod nuovo.
  if [ "$HF_HOME_EREDITATO" != "$HF_HOME" ]; then
    echo
    echo "!!! la shell che ti ha lanciato ha HF_HOME=$HF_HOME_EREDITATO"
    echo "!!! ma il token e i checkpoint stanno in $HF_HOME"
    echo "!!! prima di lanciare runner.py:  export HF_HOME=$HF_HOME"
    echo
  fi
  for v in gen a14b ann opensora fun; do
    if [ -x "$VENVS/$v/bin/python" ]; then
      printf '%-9s %s\n' "$v" "$("$VENVS/$v/bin/python" -V 2>&1)"
    else
      printf '%-9s assente\n' "$v"
    fi
  done
  if [ -x "$WORKSPACE/cosmos-transfer2.5/.venv/bin/python" ]; then
    printf '%-9s %s\n' cosmos "$("$WORKSPACE/cosmos-transfer2.5/.venv/bin/python" -V 2>&1)"
  else
    printf '%-9s assente\n' cosmos
  fi

  # Cosmos vuole driver >= 570.124.06 e glibc >= 2.35 (docs/setup.md del repo):
  # su una macchina nuova si guarda prima di installare.
  echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" \
       "(Cosmos vuole >= 570.124.06)"
  echo "glibc:  $(ldd --version | head -1) (Cosmos vuole >= 2.35)"

  # Che il token ci sia, e che apra i repo gated di Cosmos, si scopre adesso, non
  # dopo aver acceso la GPU: si scarica il solo README di ciascuno.
  COSMOS_GATED="$COSMOS_GATED" "$VENVS/gen/bin/python" - <<'PY' || true
import os
from huggingface_hub import HfApi, hf_hub_download
try:
    print("HuggingFace: autenticato come %s" % HfApi().whoami().get("name"))
except Exception as e:
    print("HuggingFace NON autenticato (%s). Lanciare: hf auth login" % e)
for repo in os.environ["COSMOS_GATED"].split():
    try:
        hf_hub_download(repo, "README.md")
        print("  accesso ok:      %s" % repo)
    except Exception as e:
        print("  accesso NEGATO:  %s (%s): accettare la licenza sulla pagina del modello"
              % (repo, type(e).__name__))
PY

  echo
  echo "prossimo passo, una clip per modello:"
  echo "  cd $CLOUD"
  echo "  $VENVS/gen/bin/python build_jobs.py --models wan22_5b --set bakeoff --out $WORKSPACE/jobs.jsonl"
  echo "  $VENVS/gen/bin/python runner.py --jobs $WORKSPACE/jobs.jsonl --out-root $WORKSPACE/out --limit 1"
  echo "ramo Sim2Real (pacchetti in $WORKSPACE/s2r_packs):"
  echo "  $VENVS/gen/bin/python build_jobs.py --models cosmos_t25,wan22_fun5b_ctrl --set sim2real --packs $WORKSPACE/s2r_packs --out $WORKSPACE/jobs_s2r.jsonl"
  echo "  $VENVS/gen/bin/python runner.py --jobs $WORKSPACE/jobs_s2r.jsonl --out-root $WORKSPACE/out_s2r --models cosmos_t25 --limit 1"
  echo "  $VENVS/fun/bin/python runner.py --jobs $WORKSPACE/jobs_s2r.jsonl --out-root $WORKSPACE/out_s2r --models wan22_fun5b_ctrl --limit 1"
fi
