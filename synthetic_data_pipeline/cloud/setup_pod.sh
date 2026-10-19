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
# Open-Sora non e' piu' mantenuto (verificato il 19/10/2026): d0cd5ac del 27/03/2025
# e' l'ultimo commit di codice, dopo c'e' solo il README. Lo stesso SHA sta in
# SPEC["repo_commit"] dell'adapter, e test_opensora2.py controlla che coincidano.
OPENSORA_COMMIT="d0cd5ac50da79e9a9d2285a952d4dcd806e6c8fc"
# TensorNVMe: import non dichiarato di opensora/utils/ckpt.py, da sorgente perche'
# la 0.1.0 su PyPI non ha async_file_io. Ultimo commit del main (27/11/2024).
TENSORNVME_COMMIT="6403388f7c8929f43e407c8f7f39c6453d78c5d4"

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

# Il pip freeze dei venv che NON sono di Open-Sora, uno dopo l'altro. Il blocco
# opensora lo prende prima e dopo: se cambia, ha toccato un ambiente non suo.
# --all perche' senza pip non elenca pip, setuptools e wheel. Il confronto non vede
# una reinstallazione alla stessa versione: e' un controllo sulle versioni, non sui file.
freeze_altri() {
  local v
  for v in gen a14b ann fun; do
    echo "## $v"
    if [ -x "$VENVS/$v/bin/pip" ]; then
      "$VENVS/$v/bin/pip" freeze --all || { echo "!!! pip freeze fallito nel venv $v" >&2; return 1; }
    else
      echo "(assente)"
    fi
  done
}

# La riga di ldd che risolve libaio per il .so di tensornvme ("" se non compare).
# Solo libaio: le librerie di torch (libc10, libtorch) senza rpath risultano
# "not found" a ldd anche quando l'import funziona.
libaio_di() {
  ldd "$1" | grep 'libaio' || true
}

# Il .so compilato di tensornvme, se c'e'. Vive nel venv opensora (sul volume), ma
# la libaio di sistema a cui e' collegato sta sul container disk.
tensornvme_so() {
  "$VENVS/opensora/bin/python" - <<'PY' 2>/dev/null
import glob, os, importlib.util
spec = importlib.util.find_spec("tensornvme")
if spec and spec.submodule_search_locations:
    trovati = glob.glob(os.path.join(list(spec.submodule_search_locations)[0], "_C*.so"))
    print(trovati[0] if trovati else "")
PY
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
  say "opensora: venv proprio, TensorNVMe da sorgente, checkpoint"
  # Open-Sora non e' un repo HuggingFace da scaricare e basta: vuole quattro
  # cose dentro ckpts/ e un ambiente con colossalai e torch 2.4. Ha il suo venv
  # perche' le sue dipendenze non sono quelle degli altri generatori, e OGNI pip
  # di questo blocco passa dal path di quel venv: niente python di sistema, niente
  # altri venv. L'unica cosa di sistema sono i pacchetti apt (libaio, cmake).
  dir="$WORKSPACE/Open-Sora"
  tn="$WORKSPACE/TensorNVMe"
  py="$VENVS/opensora/bin/python"
  pip="$VENVS/opensora/bin/pip"
  # pip muore con "Invalid cross-device link" fra /tmp e il volume MooseFS.
  export TMPDIR="$WORKSPACE/tmp"
  mkdir -p "$TMPDIR"
  freeze_altri > "$TMPDIR/freeze_altri_prima.txt"

  # libaio-dev: tensornvme si collega alla libaio di sistema. Sta sul container
  # disk, quindi dopo un Terminate sparisce: il blocco check lo vede con ldd.
  # cmake: lo chiama setup.py di tensornvme. Righe separate e ricontrollo dopo:
  # in "update && install" un update fallito non ferma lo script, e senza cmake o
  # libaio.h il setup.py di tensornvme ripiega in silenzio su una libaio compilata
  # da pagure.io dentro il volume.
  if ! command -v cmake >/dev/null || [ ! -e /usr/include/libaio.h ]; then
    apt-get update -qq
    apt-get install -y -qq libaio-dev cmake
  fi
  if ! command -v cmake >/dev/null || [ ! -e /usr/include/libaio.h ]; then
    echo "!!! cmake o libaio.h ancora assenti dopo apt: mi fermo"
    exit 1
  fi

  [ -d "$dir/.git" ] || git clone https://github.com/hpcaitech/Open-Sora "$dir"
  git -C "$dir" fetch --quiet origin
  git -C "$dir" checkout --quiet "$OPENSORA_COMMIT"

  [ -d "$VENVS/opensora" ] || python3 -m venv "$VENVS/opensora"
  "$pip" install --upgrade pip -q
  # Sul volume il venv c'e' gia' dal 18/09 (flash-attn, xformers): l'installazione
  # completa del README parte solo se manca qualcosa. Da $TMPDIR e non dal clone:
  # da li' "import opensora" riuscirebbe dai sorgenti anche senza pacchetto.
  if ! (cd "$TMPDIR" && "$py" -c "import torch, colossalai, flash_attn, opensora"); then
    echo ">>> ambiente Open-Sora incompleto (errore sopra): installazione da README"
    (cd "$dir" && "$pip" install -v .)
    "$pip" install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
    # Il setup.py di flash-attn importa torch: senza --no-build-isolation non lo vede.
    "$pip" install flash-attn --no-build-isolation
    echo "$OPENSORA_COMMIT" > "$VENVS/opensora/.opensora_commit"
  fi
  # "pip install ." non e' editabile: torchrun usa la copia in site-packages, non il
  # clone, e il checkout sopra da solo non la cambia. Se la copia non viene dal
  # commit fissato si reinstalla il solo pacchetto opensora, senza dipendenze:
  # torch e flash-attn restano dove sono.
  if [ "$(cat "$VENVS/opensora/.opensora_commit" 2>/dev/null)" != "$OPENSORA_COMMIT" ]; then
    echo ">>> pacchetto opensora non registrato a ${OPENSORA_COMMIT:0:7}: lo reinstallo senza dipendenze"
    (cd "$dir" && "$pip" install --no-deps --force-reinstall .)
    echo "$OPENSORA_COMMIT" > "$VENVS/opensora/.opensora_commit"
  fi
  # hf_transfer: HF_HUB_ENABLE_HF_TRANSFER e' nell'immagine di RunPod.
  # packaging e click: li vuole il setup.py di tensornvme, che qui gira senza deps.
  # imageio, imageio-ffmpeg, psutil: runner.py gira dentro questo venv (torchrun va
  # preso da qui) e scrive l'mp4 con imageio dove diffusers non c'e'. Stesse
  # versioni del venv gen.
  "$pip" install -q hf_transfer packaging click \
    ImageIO==2.37.4 imageio-ffmpeg==0.6.0 psutil==7.2.2

  [ -d "$tn/.git" ] || git clone https://github.com/hpcaitech/TensorNVMe.git "$tn"
  git -C "$tn" fetch --quiet origin
  git -C "$tn" checkout --quiet "$TENSORNVME_COMMIT"
  if ! (cd "$TMPDIR" && "$py" -c "import tensornvme.async_file_io"); then
    echo ">>> tensornvme assente o rotto (errore sopra): lo compilo"
    # Tre trappole del setup.py di tensornvme:
    # - scrive in ~/.tensornvme e appende a ~/.bashrc: con HOME sul volume non
    #   tocca /root;
    # - i requirements dichiarano torch: --no-deps, altrimenti pip puo' toccarlo;
    # - ignora l'esito di cmake e make: l'import e ldd qui sotto sono la verifica.
    # Si cancellano i resti di una compilazione precedente: una libaio lasciata in
    # ~/.tensornvme/lib verrebbe presa dal linker prima di quella di sistema.
    # DISABLE_URING: liburing andrebbe scaricata e compilata, e a noi non serve.
    rm -rf "$WORKSPACE/tensornvme-home/.tensornvme" "$tn/build" "$tn/cmake-build"
    mkdir -p "$WORKSPACE/tensornvme-home"
    (cd "$tn" && HOME="$WORKSPACE/tensornvme-home" DISABLE_URING=1 \
      "$pip" install -v --no-cache-dir --no-build-isolation --no-deps .)
  fi
  so="$(tensornvme_so || true)"
  aio=""
  if [ -n "$so" ]; then aio="$(libaio_di "$so")"; fi
  case "$aio" in
    *"not found"*|"")
      echo "!!! tensornvme: libaio non risolta (${so:-.so non trovato}): $aio"; exit 1 ;;
    *"$WORKSPACE"*)
      echo "!!! tensornvme collegato a una libaio sul volume, non a quella di sistema: $aio"
      exit 1 ;;
    *) echo "tensornvme collegato a: $(echo "$aio" | xargs)" ;;
  esac
  # Da una cartella neutra, non da $dir: torchrun lancia scripts/diffusion/
  # inference.py, quindi opensora arriva dal venv e non dal clone, e la verifica
  # deve vedere la stessa cosa.
  (cd "$TMPDIR" && "$py" -c "import tensornvme.async_file_io, opensora.utils.ckpt as c;
print('tensornvme e opensora.utils.ckpt importabili, opensora da', c.__file__)")

  "$pip" freeze --all > "$VENVS/requirements-opensora.lock"
  echo "versioni installate in $VENVS/requirements-opensora.lock (da riportare nel repo)"

  freeze_altri > "$TMPDIR/freeze_altri_dopo.txt"
  if diff -u "$TMPDIR/freeze_altri_prima.txt" "$TMPDIR/freeze_altri_dopo.txt"; then
    echo "venv gen, a14b, ann, fun invariati"
  else
    echo "!!! il blocco opensora ha cambiato un venv che non e' suo (diff sopra)"
    exit 1
  fi

  for c in Open_Sora_v2.safetensors hunyuan_vae.safetensors google/t5-v1_1-xxl \
           openai/clip-vit-large-patch14; do
    if [ -e "$dir/ckpts/$c" ]; then echo "ckpts/$c presente"; else echo "!!! ckpts/$c MANCA"; fi
  done
  echo "Open-Sora @ $(git -C "$dir" rev-parse --short HEAD), TensorNVMe @ $(git -C "$tn" rev-parse --short HEAD)"
  echo ">>> prima di runner.py: export OPENSORA_DIR=$dir"
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

  # tensornvme sta nel venv opensora, sul volume, ma la libaio di sistema a cui e'
  # collegato sta sul container disk: dopo un Terminate l'import di Open-Sora
  # fallisce anche se il venv e' intatto. Si guarda con ldd e si rimette da apt.
  so="$(tensornvme_so || true)"
  if [ -n "$so" ]; then
    case "$(libaio_di "$so")" in
      *"not found"*)
        echo "tensornvme: libaio mancante, reinstallo libaio-dev"
        apt-get update -qq
        apt-get install -y -qq libaio-dev
        case "$(libaio_di "$so")" in
          *"not found"*) echo "!!! tensornvme: libaio ancora mancante dopo apt" ;;
        esac ;;
    esac
    if (cd /tmp && "$VENVS/opensora/bin/python" -c "import tensornvme.async_file_io"); then
      echo "tensornvme importabile ($(libaio_di "$so" | xargs))"
    else
      echo "!!! tensornvme NON importabile (errore sopra): rilanciare il blocco opensora"
    fi
  elif [ -x "$VENVS/opensora/bin/python" ]; then
    echo "tensornvme assente nel venv opensora: lanciare il blocco opensora"
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
  echo "Open-Sora (venv attivato: torchrun lo prende dal PATH):"
  echo "  source $VENVS/opensora/bin/activate && export OPENSORA_DIR=$WORKSPACE/Open-Sora"
  echo "  python build_jobs.py --models opensora2 --set bakeoff --out $WORKSPACE/jobs_os.jsonl"
  echo "  python runner.py --jobs $WORKSPACE/jobs_os.jsonl --out-root $WORKSPACE/out_os --limit 1"
  echo "ramo Sim2Real (pacchetti in $WORKSPACE/s2r_packs):"
  echo "  $VENVS/gen/bin/python build_jobs.py --models cosmos_t25,wan22_fun5b_ctrl --set sim2real --packs $WORKSPACE/s2r_packs --out $WORKSPACE/jobs_s2r.jsonl"
  echo "  $VENVS/gen/bin/python runner.py --jobs $WORKSPACE/jobs_s2r.jsonl --out-root $WORKSPACE/out_s2r --models cosmos_t25 --limit 1"
  echo "  $VENVS/fun/bin/python runner.py --jobs $WORKSPACE/jobs_s2r.jsonl --out-root $WORKSPACE/out_s2r --models wan22_fun5b_ctrl --limit 1"
fi
