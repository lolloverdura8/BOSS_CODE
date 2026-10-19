# test_opensora2.py
#
# Casi per l'adapter di Open-Sora 2.0 che non toccano la GPU (piano del 19/10,
# passo 1): il comando passato a torchrun porta anche il seed del rumore, lo SPEC
# dichiara stato del progetto e caso del seed 0, e il commit nello SPEC e' lo
# stesso che setup_pod.sh fissa sul pod.
#
# pytest non e' installato: si esegue con  python cloud/test_opensora2.py
import contextlib
import io
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from adapters import opensora2


def valore_dopo(cmd, flag):
    """Il valore che segue flag nella lista cmd, una volta sola."""
    assert cmd.count(flag) == 1, f"{flag} compare {cmd.count(flag)} volte"
    return cmd[cmd.index(flag) + 1]


def test_il_seed_del_rumore_arriva_a_torchrun():
    cmd = opensora2.comando("un prompt", 1005, "/tmp/out")
    # --seed da solo imposta set_seed globale; il rumore latente lo decide
    # sampling_option.seed, None per default nel config 256px.py.
    assert valore_dopo(cmd, "--seed") == "1005"
    assert valore_dopo(cmd, "--sampling_option.seed") == "1005"


def test_il_comando_porta_prompt_e_cartella():
    cmd = opensora2.comando("ramo basso", 3, "/tmp/out")
    assert valore_dopo(cmd, "--prompt") == "ramo basso"
    assert valore_dopo(cmd, "--save-dir") == "/tmp/out"
    assert valore_dopo(cmd, "--num_frames") == str(opensora2.SPEC["num_frames"])


def test_coppie_flag_valore_per_merge_args():
    # merge_args di Open-Sora legge gli argomenti a coppie (zip(args[::2],
    # args[1::2])) e vuole che ogni chiave inizi con "--": dopo il config, un
    # flag senza valore sposterebbe tutte le coppie successive.
    cmd = opensora2.comando("p", 7, "/tmp/out")
    dopo_config = cmd[cmd.index(opensora2.SPEC["config"]) + 1:]
    assert len(dopo_config) % 2 == 0, dopo_config
    assert all(k.startswith("--") for k in dopo_config[::2]), dopo_config


def test_lo_spec_dichiara_stato_e_seed_zero():
    spec = opensora2.SPEC
    assert "03/2025" in spec["manutenzione"]
    assert "seed 0" in spec["seed_rumore"]


def test_commit_dello_spec_uguale_a_setup_pod():
    with open(os.path.join(HERE, "setup_pod.sh"), encoding="utf-8") as f:
        testo = f.read()
    m = re.search(r'^OPENSORA_COMMIT="([0-9a-f]{40})"', testo, flags=re.MULTILINE)
    assert m, "OPENSORA_COMMIT non trovato in setup_pod.sh"
    assert opensora2.SPEC["repo_commit"] == m.group(1)


def genera_con_torchrun_finto(seed, codice, stdout="riga out", stderr="riga err"):
    """generate() con subprocess.run sostituito: torchrun "esce" con codice senza
    scrivere mp4. Restituisce (eccezione o None, chiamate registrate, stdout)."""
    chiamate = []

    def finto(cmd, **kw):
        chiamate.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, codice, stdout=stdout, stderr=stderr)

    originale = subprocess.run
    subprocess.run = finto
    uscita = io.StringIO()
    try:
        with contextlib.redirect_stdout(uscita):
            opensora2.generate({"repo": HERE}, "p", seed)
    except RuntimeError as exc:
        return exc, chiamate, uscita.getvalue()
    finally:
        subprocess.run = originale
    return None, chiamate, uscita.getvalue()


def test_generate_passa_a_torchrun_il_seed_del_rumore():
    _, chiamate, _ = genera_con_torchrun_finto(42, 3)
    assert len(chiamate) == 1, chiamate
    cmd, kw = chiamate[0]
    assert valore_dopo(cmd, "--sampling_option.seed") == "42"
    assert kw["cwd"] == HERE and kw["check"] is False, kw


def test_fallimento_di_torchrun_solleva_con_diagnosi():
    exc, _, _ = genera_con_torchrun_finto(1, 3)
    assert exc is not None, "generate non ha sollevato"
    assert "uscito con 3" in str(exc) and "riga err" in str(exc), exc


def test_uscita_zero_senza_mp4_solleva():
    exc, _, _ = genera_con_torchrun_finto(1, 0)
    assert exc is not None, "generate non ha sollevato"
    assert "nessun mp4" in str(exc), exc


def test_seed_zero_avvisa():
    _, _, stampato = genera_con_torchrun_finto(0, 3)
    assert "seed 0" in stampato, stampato
    _, _, stampato = genera_con_torchrun_finto(5, 3)
    assert "seed 0" not in stampato, stampato


LOG_OPENSORA = (
    "[\x1b[34m2026-10-20 10:00:00\x1b[0m] CUDA max memory max memory allocated at build model: 30.2 GB\n"
    "[\x1b[34m2026-10-20 10:00:00\x1b[0m] CUDA max memory max memory reserved at build model: 31.0 GB\n"
    "[\x1b[34m2026-10-20 10:09:00\x1b[0m] CUDA max memory max memory allocated at inference: 52.5 GB\n"
    "[\x1b[34m2026-10-20 10:09:00\x1b[0m] CUDA max memory max memory reserved at inference: 60.3 GB\n"
)


def test_picco_vram_letto_dal_log_di_inference():
    # Il runner misura la VRAM nel proprio processo: per Open-Sora, che genera in
    # torchrun, darebbe zero. Il picco vero lo stampa Open-Sora a fine inferenza.
    m = opensora2.memoria_da_log(LOG_OPENSORA)
    assert m == {"vram_alloc_peak_gb_torchrun": 52.5, "vram_reserved_peak_gb_torchrun": 60.3}, m


def test_picco_vram_assente_dal_log_resta_none():
    m = opensora2.memoria_da_log("nessuna riga utile")
    assert m == {"vram_alloc_peak_gb_torchrun": None, "vram_reserved_peak_gb_torchrun": None}, m


def test_metriche_della_clip_dopo_generate():
    # torchrun fallisce, ma il log c'e': le metriche della clip devono riflettere
    # l'ultima chiamata, non restare quelle di una clip precedente.
    handle = {"repo": HERE}
    originale = subprocess.run
    subprocess.run = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr=LOG_OPENSORA)
    try:
        try:
            opensora2.generate(handle, "p", 1)
        except RuntimeError:
            pass
    finally:
        subprocess.run = originale
    assert opensora2.metriche_clip(handle)["vram_alloc_peak_gb_torchrun"] == 52.5


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok     {name}")
        except Exception as exc:  # noqa: BLE001 -- ogni errore di un test va contato, non deve fermare gli altri
            failed += 1
            print(f"FALLITO {name} {type(exc).__name__}: {exc}")
    print(f"\n{len(tests)} test, {failed} falliti")
    sys.exit(1 if failed else 0)
