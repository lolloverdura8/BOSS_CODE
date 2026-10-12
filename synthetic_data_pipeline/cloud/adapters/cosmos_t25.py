# cosmos_t25.py — Cosmos-Transfer2.5-2B (NVIDIA), ramo Sim2Real
#
# Video-to-video controllato: ridipinge la scena di CARLA seguendo i video di
# controllo del pacchetto (depth, edge, seg) di CARLA/prepare_controls.py. Le due
# configurazioni A e B, coi pesi, sono in sim2real.CONFIGS.
#
# Come Open-Sora, NON gira dentro il processo del runner: ha un ambiente proprio
# (uv, dal repo nvidia-cosmos/cosmos-transfer2.5) e si pilota da riga di comando
# con uno spec JSON. L'adapter scrive lo spec, lancia examples/inference.py in un
# sottoprocesso e rilegge l'mp4. Per questo:
#   - i pesi si ricaricano a ogni clip: costo da dichiarare nel report, misurato
#     sulla prima clip (scelta confermata il 12/10, un processo per clip);
#   - i pixel passano per l'H.264 scritto da Cosmos, mentre Wan Fun restituisce il
#     tensore: e' l'asimmetria gia' dichiarata per Open-Sora (pixel_source).
#
# GUARDRAIL ATTIVI, NON SI SPENGONO. La model card dice che aggirarli fa decadere i
# diritti della licenza. Sul video applicano anche la sfocatura dei volti
# (RetinaFace): i pedoni delle clip di Cosmos avranno il volto sfocato, quelli di
# Wan no. Va scritto nel report e tenuto presente nella Fase 4.
#
# Fonti verificate il 12/10/2026 al commit 2ff49d07 del repo: docs/inference.md,
# cosmos_transfer2/config.py (campi dello spec), cosmos_transfer2/inference.py
# (uscita in <output_dir>/<name>.mp4, guardrail), _src/transfer2/inference/
# inference_pipeline.py (video_path usato come partenza solo con sigma_max).
#
# SETUP SUL POD: setup_pod.sh cosmos.
import glob
import json
import os
import shutil
import subprocess
import tempfile

# Come wan22_5b con gpu: si importa da cloud/, che runner.py mette nel path.
import prompts
import sim2real
from adapters.opensora2 import _coda, _read_video

REPO_DIR = os.environ.get("COSMOS_DIR", "/workspace/cosmos-transfer2.5")
PYTHON = os.environ.get("COSMOS_PYTHON", os.path.join(REPO_DIR, ".venv", "bin", "python"))

SPEC = {
    "model_id": "nvidia/Cosmos-Transfer2.5-2B",
    "repo": "nvidia-cosmos/cosmos-transfer2.5",
    "commit": "2ff49d0717af02057ae79bc75c00fbff9da1b4e7",
    "width": 1280,
    "height": 720,
    "num_frames": 93,
    "fps": 16,
    # Default del repo (config.py): 35 passi, guidance 3.
    "steps": 35,
    "guidance": 3,
    # Nessun sigma_max: la clip parte dal rumore, l'RGB di CARLA non e' il punto
    # di partenza del denoising (vedi sim2real.cosmos_spec).
    "sigma_max": None,
    "configs": sim2real.CONFIGS["cosmos_t25"],
    "guardrails": "attivi (licenza): blocklist e Qwen3Guard sul testo, filtro e"
                  " sfocatura dei volti sul video",
    "negative": prompts.SIM2REAL_NEGATIVE_COSMOS,
    "dtype": "bf16",
    "pixel_source": "mp4 (H.264) scritto da Cosmos",
    # Da riportare nel confronto A/B: con un controllo solo Cosmos usa il checkpoint
    # dedicato (depth), con piu' controlli il modello multibranch, dove vis ha peso 0
    # e i pesi vengono normalizzati. A contro B cambia anche il modello, non solo i
    # controlli.
    "checkpoint_note": "A: checkpoint depth dedicato; B: multibranch, pesi normalizzati",
    "weights_reload": "a ogni clip (un sottoprocesso per clip)",
}


# Una clip e' stimata 15-30 minuti sull'A100, ricaricamento dei pesi compreso; al
# primo lancio si aggiungono i download dei checkpoint. Oltre le 2 ore e' appesa.
TIMEOUT_S = 2 * 3600


def _env():
    """L'ambiente del sottoprocesso, senza COSMOS_SMOKE.

    Con COSMOS_SMOKE impostato il repo spegne i guardrail (disable_guardrails ha
    default True in modalita' smoke) e accorcia la generazione: le clip non
    sarebbero quelle dichiarate nello SPEC, e la licenza chiede i guardrail attivi.
    """
    return {k: v for k, v in os.environ.items() if k != "COSMOS_SMOKE"}


def load(prompts_list):
    """Non carica niente: il modello vive nel sottoprocesso. Controlla solo che ci sia."""
    for path, what in ((REPO_DIR, "repo"), (PYTHON, "python del venv uv")):
        if not os.path.exists(path):
            raise SystemExit(f"Cosmos: {what} non trovato in {path}. Lanciare setup_pod.sh"
                             " cosmos, o impostare COSMOS_DIR / COSMOS_PYTHON.")
    print(f"Cosmos-Transfer2.5: repo {REPO_DIR}, python {PYTHON}")
    print("NOTA: i pesi si ricaricano a ogni clip; guardrail attivi (volti sfocati).")
    return {"repo": REPO_DIR, "python": PYTHON}


def _diagnosi(r):
    return (f"examples/inference.py uscito con {r.returncode}.\n\n--- stderr ---\n"
            f"{_coda(r.stderr)}\n\n--- stdout ---\n{_coda(r.stdout)}")


def _output_video(out_dir, name):
    """L'mp4 della clip, non quelli dei controlli (<name>_control_<k>.mp4)."""
    path = os.path.join(out_dir, name + ".mp4")
    if os.path.exists(path):
        return path
    found = sorted(glob.glob(os.path.join(out_dir, "**", name + ".mp4"), recursive=True))
    return found[0] if found else None


def generate(handle, prompt, seed, control):
    name = "clip"
    work = tempfile.mkdtemp(prefix="cosmos_")
    try:
        spec = sim2real.cosmos_spec(control, prompt, SPEC["negative"], seed, name,
                                    SPEC["steps"], SPEC["guidance"])
        spec_path = os.path.join(work, "spec.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
        out_dir = os.path.join(work, "out")
        cmd = [handle["python"], "examples/inference.py", "-i", spec_path, "-o", out_dir]
        # check=False: il codice d'uscita si legge sotto, per riportare stdout e stderr.
        # Il timeout ferma un download o un caricamento appeso, che sul pod si paga.
        r = subprocess.run(cmd, cwd=handle["repo"], capture_output=True, text=True, check=False,
                           env=_env(), timeout=TIMEOUT_S)
        if r.returncode != 0:
            raise RuntimeError(_diagnosi(r))
        mp4 = _output_video(out_dir, name)
        if mp4 is None:
            # keep_going e' attivo per default: un blocco dei guardrail o un errore
            # interno possono finire con codice 0 e nessun video.
            raise RuntimeError("nessun video prodotto (guardrail o errore interno).\n"
                               + _diagnosi(r))
        return _read_video(mp4)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def unload(handle):
    # Niente da liberare: il sottoprocesso e' gia' uscito e con lui la sua VRAM.
    pass
