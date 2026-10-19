# opensora2.py — Open-Sora 2.0 (hpc-ai tech)
#
# L'unico dei sei che NON passa da diffusers. Ha un repo proprio con
# torchrun/colossalai, e si pilota da riga di comando con un file di
# configurazione mmengine. L'adapter quindi lancia un sottoprocesso e rilegge
# cio' che ha prodotto.
#
# E' IL MODELLO CHE IMPONE LA SCHEDA DA 80 GB A TUTTO IL BAKE-OFF. La model card
# dichiara, su GPU singola, un picco di 52,5 GB a 256px e 60,3 GB a 768px: da
# solo esclude la A6000 da 48 GB. Se cade a S2, il bake-off scende su A6000 a
# $0,33/h invece che su H100 a $1,99/h.
#
# E' anche il piu' vecchio: 12 marzo 2025, e non ho trovato una versione piu'
# recente cercando sul GitHub hpcaitech/Open-Sora e sull'organizzazione
# HuggingFace hpcai-tech il 15/09/2026. Sta nel bake-off perche' il §2.4 del
# deliverable OR4.1 lo nomina esplicitamente accanto a Wan2.2 e CogVideoX.
#
# NON PIU' MANTENUTO (verificato il 19/10/2026). L'ultimo commit di codice e'
# d0cd5ac del 27/03/2025; dopo c'e' solo un banner nel README (09/04/2026). Le
# issue le chiude un bot di stale senza risposte, le PR esterne restano aperte.
# Il repo si fissa a quel commit (OPENSORA_COMMIT in setup_pod.sh) e i bug che
# troviamo vanno corretti qui: a monte non arriveranno fix.
#
# SEED 0. inference.py passa al campionatore "seed if seed else None": con seed 0
# il rumore latente torna casuale e la clip non si riproduce. Con lo schema di
# build_jobs (seed = indice globale del prompt) succede solo a monopattino_00.
# Si dichiara nello SPEC, quindi nel manifest, e il runner lo stampa.
#
# ASIMMETRIA DA DICHIARARE NEL REPORT, NON DA NASCONDERE. Gli altri cinque
# adapter restituiscono l'array di pixel della pipeline, e i PNG per SAM e DA3
# escono da li', senza compressione. Qui i pixel si possono rileggere solo
# dall'mp4 che lo script ha scritto, quindi sono passati per H.264. E' un
# handicap reale e misurabile su un modello soltanto: SPEC["pixel_source"] lo
# porta nel manifest e da li' nel report.
#
# SETUP SUL POD: "bash setup_pod.sh opensora", non qui. Clona il repo al commit
# fissato, installa nel solo venv opensora (colossalai, flash-attn) e TensorNVMe da
# sorgente, e controlla che ckpts/ abbia Open_Sora_v2.safetensors,
# hunyuan_vae.safetensors, google/t5-v1_1-xxl e openai/clip-vit-large-patch14:
# non e' un solo repo HuggingFace da scaricare, sono quattro cose in una cartella.
import glob
import os
import shutil
import subprocess
import tempfile

import numpy as np

# Radice del clone, sovrascrivibile: sul pod sta sul network volume.
REPO_DIR = os.environ.get("OPENSORA_DIR", "/workspace/Open-Sora")

SPEC = {
    "model_id": "hpcai-tech/Open-Sora-v2",
    "repo": "hpcaitech/Open-Sora",
    # 768px con aspect ratio 16:9. La risoluzione esatta in pixel la decide il
    # modello dalla coppia (resolution, aspect_ratio), non si passa in H x W come
    # negli altri: e' per questo che qui height e width sono None e il valore
    # vero lo registra il runner da cio' che esce.
    "config": "configs/diffusion/inference/768px.py",
    "resolution": "768px",
    "aspect_ratio": "16:9",
    "height": None,
    "width": None,
    # Il vincolo del modello e' num_frames = 4k+1 e < 129. 121 = 4x30 + 1.
    "num_frames": 121,
    "steps": 50,
    # Default del config 256px.py, da cui 768px.py eredita.
    "guidance": 7.5,
    "fps": 24,
    "dtype": "bf16",
    "negative": None,
    # Vedi l'asimmetria in testa al file.
    "pixel_source": "mp4 (H.264)",
    # Stesso valore di OPENSORA_COMMIT in setup_pod.sh: lo controlla il test.
    "repo_commit": "d0cd5ac50da79e9a9d2285a952d4dcd806e6c8fc",
    "manutenzione": "non piu' mantenuto dal 03/2025 (ultimo commit di codice d0cd5ac, 27/03/2025)",
    "seed_rumore": "sampling_option.seed = seed; seed 0 diventa None in inference.py, "
                   "quindi rumore casuale e clip non riproducibile",
}


def load(prompts):
    """Non carica niente: il modello vive dentro il sottoprocesso.

    Il costo e' che i pesi si ricaricano a ogni clip, il che gonfia i secondi
    per clip di M7. Va detto nel report: non e' che il modello sia piu' lento a
    generare, e' che questo adapter paga un caricamento per clip che gli altri
    pagano una volta sola.
    """
    if not os.path.isdir(REPO_DIR):
        raise SystemExit(f"Open-Sora non trovato in {REPO_DIR}. Impostare OPENSORA_DIR o"
                         " clonare il repo (vedi setup_pod.sh).")
    cfg = os.path.join(REPO_DIR, SPEC["config"])
    if not os.path.exists(cfg):
        raise SystemExit(f"config non trovato: {cfg}")
    print(f"Open-Sora: repo {REPO_DIR}, config {SPEC['config']}")
    print("NOTA: i pesi si ricaricano a ogni clip, M7 ne risente.")
    return {"repo": REPO_DIR}


def _read_video(path):
    import imageio.v2 as imageio
    reader = imageio.get_reader(path)
    frames = [np.asarray(f) for f in reader]
    reader.close()
    if not frames:
        raise RuntimeError(f"nessun frame leggibile in {path}")
    # uint8 0..255 -> float 0..1, il contratto degli adapter. check_frames del
    # runner lo farebbe comunque, ma meglio che il contratto lo rispetti chi lo deve.
    return np.stack(frames).astype(np.float32) / 255.0


# Quante righe di coda tenere dei due flussi quando il sottoprocesso fallisce.
# 25 non bastavano: torchrun incapsula il traceback del figlio dentro un
# ChildFailedError e stampa una ventina di righe di impalcatura sua, quindi una
# coda corta mostra solo quella e nasconde la causa. Successo il 18/09/2026 al
# primo tentativo di Open-Sora sul pod: l'errore utile stava piu' su.
CODA_RIGHE = 80


def _coda(testo, n=CODA_RIGHE):
    righe = (testo or "").strip().splitlines()
    return "\n".join(righe[-n:]) if righe else "(vuoto)"


def _diagnosi(r):
    """Il messaggio d'errore con entrambi i flussi.

    Serve stdout oltre a stderr perche' gli script di Open-Sora stampano li'
    sia la configurazione risolta sia buona parte degli errori di caricamento:
    col solo stderr si legge il fallimento del launcher, non cio' che l'ha
    causato.
    """
    return (f"torchrun uscito con {r.returncode}.\n\n"
            f"--- stderr (ultime {CODA_RIGHE} righe) ---\n{_coda(r.stderr)}\n\n"
            f"--- stdout (ultime {CODA_RIGHE} righe) ---\n{_coda(r.stdout)}")


def comando(prompt, seed, out_dir):
    """La riga di torchrun per una clip.

    merge_args di Open-Sora legge gli argomenti dopo il config a coppie
    "--chiave valore": ogni flag qui deve avere il suo valore.
    """
    return [
        "torchrun", "--nproc_per_node", "1", "--standalone",
        "scripts/diffusion/inference.py", SPEC["config"],
        "--save-dir", out_dir,
        "--prompt", prompt,
        "--num_frames", str(SPEC["num_frames"]),
        "--aspect_ratio", SPEC["aspect_ratio"],
        # --seed imposta solo set_seed globale. Il rumore latente lo decide
        # sampling_option.seed, None per default: senza questo Open-Sora era
        # l'unico modello del bake-off senza seed controllato.
        "--seed", str(seed),
        "--sampling_option.seed", str(seed),
    ]


def generate(handle, prompt, seed):
    if seed == 0:
        print("  attenzione: seed 0, Open-Sora lo tratta come None e il rumore e' casuale"
              " (vedi SPEC['seed_rumore'])")
    # Una cartella vuota per clip: il nome del file lo decide get_save_path_name
    # dentro il repo e non e' prevedibile da qui, quindi invece di indovinarlo si
    # prende l'unico mp4 che compare.
    out_dir = tempfile.mkdtemp(prefix="opensora_")
    try:
        cmd = comando(prompt, seed, out_dir)
        r = subprocess.run(cmd, cwd=handle["repo"], capture_output=True, text=True, check=False)
        if r.returncode != 0:
            raise RuntimeError(_diagnosi(r))
        mp4 = sorted(glob.glob(os.path.join(out_dir, "**", "*.mp4"), recursive=True))
        if not mp4:
            raise RuntimeError(f"nessun mp4 prodotto in {out_dir}, ma torchrun e' uscito"
                               f" con 0.\n{_diagnosi(r)}")
        if len(mp4) > 1:
            print(f"  attenzione: {len(mp4)} mp4 prodotti, prendo il primo")
        return _read_video(mp4[0])
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def unload(handle):
    # Niente da liberare: il sottoprocesso e' gia' uscito e con lui la sua VRAM.
    pass
