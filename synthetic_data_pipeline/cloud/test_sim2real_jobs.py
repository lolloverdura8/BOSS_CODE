# test_sim2real_jobs.py
#
# Casi per la parte del ramo Sim2Real che non tocca la GPU (Fase 3, piano del
# 12/10): i job costruiti dai pacchetti di CARLA/prepare_controls.py, gli spec di
# Cosmos, il control_video di Wan Fun, il controllo che l'uscita sia allineata alla
# GT, e le due modifiche al runner (controllo passato all'adapter, tutti i frame su
# disco). I pacchetti veri, se ci sono sul PC, danno i 20 job attesi; quelli
# sintetici provano gli errori.
#
# pytest non e' installato: si esegue con  python cloud/test_sim2real_jobs.py
# (Python 3.11 del PC, che ha torch e psutil, richiesti da runner.py).
import hashlib
import json
import os
import shutil
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import prompts
import runner
import sim2real as S

REAL_PACKS = os.path.join(HERE, "..", "CARLA", "outputs", "s2r_packs")
SCENES = sorted(prompts.SIM2REAL_PROMPTS)


def expect_error(exc_type, fn, *args, contains=""):
    try:
        fn(*args)
    except exc_type as exc:
        assert contains in str(exc), exc
        return
    raise AssertionError(f"{fn.__name__} non ha sollevato {exc_type.__name__}")


def make_pack(root, scenario, suffix, n_frames=93, fps=16, size=(1280, 720), files=None):
    """Pacchetto sintetico: controls/*.mp4 finti (byte qualsiasi) e manifest coerente."""
    name = f"{scenario}__{suffix}"
    files = files or {k: f"video {k} {name}".encode() for k in ("rgb", "depth", "edge", "seg")}
    os.makedirs(os.path.join(root, name, "controls"))
    hashes = {}
    for key, data in files.items():
        with open(os.path.join(root, name, "controls", f"{key}.mp4"), "wb") as f:
            f.write(data)
        hashes[f"controls/{key}.mp4"] = hashlib.sha256(data).hexdigest()
    manifest = {"n_frames": n_frames, "fps": fps, "resolution": list(size),
                "crop_rows": {"top": (720 - size[1]) // 2, "bottom": (720 - size[1]) // 2},
                "session": {"scenario": scenario},
                "obstacle": {"parts": [{"boss_class": "ramo_sporgente"}]}, "files": hashes}
    with open(os.path.join(root, name, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    return name


# --- job dai pacchetti -----------------------------------------------------------------

def test_venti_job_dai_pacchetti_veri():
    if not os.path.isdir(REAL_PACKS):
        print("  SKIP: pacchetti veri assenti in", REAL_PACKS)
        return
    jobs = S.build_jobs(["cosmos_t25", "wan22_fun5b_ctrl"], REAL_PACKS, prompts.SIM2REAL_PROMPTS)
    assert len(jobs) == 20, len(jobs)
    ids = {(j["modello"], j["clip_id"]) for j in jobs}
    assert ids == {(m, f"{s}__{c}") for m in ("cosmos_t25", "wan22_fun5b_ctrl")
                   for s in SCENES for c in "AB"}
    for j in jobs:
        c = j["control"]
        cosmos = j["modello"] == "cosmos_t25"
        assert c["pack"].endswith("__cosmos" if cosmos else "__wanfun")
        assert (c["n_frames"], c["fps"], c["size"]) == (
            (93, 16, [1280, 720]) if cosmos else (121, 24, [1280, 704])), c
        assert j["seed"] == 0 and j["save_all_frames"] and j["prompt"] == prompts.SIM2REAL_PROMPTS[c["scenario"]]
        assert c["crop_rows"] == ({"top": 0, "bottom": 0} if cosmos else {"top": 8, "bottom": 8})
        for key in list(c["controls"]) + ["video"]:
            assert os.path.exists(S.control_path(c, key)), (j["clip_id"], key)
        S.verify_control_files(c)
    insegna = [j for j in jobs if j["control"]["scenario"] == "s2r_insegna"]
    assert {j["classe"] for j in insegna} == {"insegna_cartello_basso"}


def test_pacchetto_incompleto_o_alterato_ferma_i_job():
    root = tempfile.mkdtemp(prefix="packs_")
    try:
        make_pack(root, "s2r_ramo", "cosmos")
        assert len(S.build_jobs(["cosmos_t25"], root, prompts.SIM2REAL_PROMPTS)) == 2
        path = os.path.join(root, "s2r_ramo__cosmos", "controls", "seg.mp4")
        with open(path, "ab") as f:
            f.write(b"x")
        expect_error(S.Sim2RealError, S.build_jobs, ["cosmos_t25"], root,
                     prompts.SIM2REAL_PROMPTS, contains="sha")
        os.remove(path)
        expect_error(S.Sim2RealError, S.build_jobs, ["cosmos_t25"], root,
                     prompts.SIM2REAL_PROMPTS, contains="manca")
    finally:
        shutil.rmtree(root)


def test_errori_di_configurazione():
    root = tempfile.mkdtemp(prefix="packs_")
    try:
        expect_error(S.Sim2RealError, S.build_jobs, ["wan22_5b"], root, {}, contains="sim2real")
        expect_error(S.Sim2RealError, S.build_jobs, ["cosmos_t25"], root, {}, contains="nessun pacchetto")
        make_pack(root, "s2r_ignota", "cosmos")
        expect_error(S.Sim2RealError, S.build_jobs, ["cosmos_t25"], root,
                     prompts.SIM2REAL_PROMPTS, contains="prompt")
        expect_error(S.Sim2RealError, S.build_jobs, ["cosmos_t25"], os.path.join(root, "no"), {},
                     contains="inesistente")
    finally:
        shutil.rmtree(root)


# --- spec e controlli -------------------------------------------------------------------

def control_of(model, config, root):
    name = f"s2r_ramo__{S.PACK_SUFFIX[model]}"
    manifest, sha = S.load_pack(root, name)
    return S.job_control(root, name, manifest, sha, model, config)


def test_spec_cosmos_a_e_b():
    root = tempfile.mkdtemp(prefix="packs_")
    try:
        make_pack(root, "s2r_ramo", "cosmos")
        a = S.cosmos_spec(control_of("cosmos_t25", "A", root), "p", "n", 0, "clip", 35, 3)
        b = S.cosmos_spec(control_of("cosmos_t25", "B", root), "p", "n", 0, "clip", 35, 3)
        assert json.loads(json.dumps(a)) == a
        assert "sigma_max" not in a and "sigma_max" not in b
        assert a["seed"] == 0 and a["num_steps"] == 35 and a["guidance"] == 3
        assert {k: v["control_weight"] for k, v in a.items() if isinstance(v, dict)} == {"depth": 1.0}
        assert {k: v["control_weight"] for k, v in b.items() if isinstance(v, dict)} == {
            "edge": 0.5, "seg": 0.9, "depth": 0.5}
        assert a["video_path"].replace("\\", "/").endswith("s2r_ramo__cosmos/controls/rgb.mp4")
        assert b["seg"]["control_path"].endswith("controls/seg.mp4")
        assert os.path.isabs(a["depth"]["control_path"])
    finally:
        shutil.rmtree(root)


def test_control_video_di_wan():
    root = tempfile.mkdtemp(prefix="packs_")
    try:
        make_pack(root, "s2r_ramo", "wanfun", n_frames=121, fps=24, size=(1280, 704))
        assert S.wan_control_video(control_of("wan22_fun5b_ctrl", "A", root)).endswith("depth.mp4")
        assert S.wan_control_video(control_of("wan22_fun5b_ctrl", "B", root)).endswith("edge.mp4")
        doppio = control_of("wan22_fun5b_ctrl", "A", root)
        doppio["controls"]["edge"] = doppio["controls"]["depth"]
        expect_error(S.Sim2RealError, S.wan_control_video, doppio, contains="un controllo solo")
    finally:
        shutil.rmtree(root)


def test_uscita_disallineata_rifiutata():
    control = {"pack": "p", "n_frames": 121, "size": [1280, 704]}
    assert S.check_control_output((121, 704, 1280, 3), control) is None
    assert "121" in S.check_control_output((120, 704, 1280, 3), control)
    assert S.check_control_output((121, 720, 1280, 3), control) is not None
    assert S.check_control_output((121, 1280, 704, 3), control) is not None   # verticale
    assert S.check_control_output((122, 704, 1280, 3), control) is not None
    assert S.check_control_output((121, 704, 1024, 3), control) is not None
    assert S.check_control_output((121, 704, 1280, 4), control) is not None
    assert "p" in S.check_control_output((121, 704, 1280), control)


def test_campo_control_completo_da_pacchetto_sintetico():
    root = tempfile.mkdtemp(prefix="packs_")
    try:
        files = {k: f"contenuto distinto di {k}".encode() * (i + 1)
                 for i, k in enumerate(("rgb", "depth", "edge", "seg"))}
        name = make_pack(root, "s2r_ramo", "wanfun", n_frames=121, fps=24, size=(1280, 704),
                         files=files)
        jobs = S.build_jobs(["wan22_fun5b_ctrl"], os.path.relpath(root), prompts.SIM2REAL_PROMPTS)
        assert [j["clip_id"] for j in jobs] == ["s2r_ramo__A", "s2r_ramo__B"]
        for j, config, key in zip(jobs, "AB", ("depth", "edge")):
            c = j["control"]
            assert os.path.isabs(c["root"]) and c["pack"] == name and c["config"] == config
            assert c["manifest_sha256"] == S.sha256_file(os.path.join(root, name, "manifest.json"))
            assert list(c["controls"]) == [key]
            assert c["controls"][key]["sha256"] == hashlib.sha256(files[key]).hexdigest()
            assert c["video"]["sha256"] == hashlib.sha256(files["rgb"]).hexdigest()
            assert (c["n_frames"], c["fps"], c["size"], c["scenario"]) == (121, 24, [1280, 704], "s2r_ramo")
            assert c["crop_rows"] == {"top": 8, "bottom": 8}
            assert j["classe"] == "ramo_sporgente" and j["seed"] == 0
        assert S.build_jobs(["wan22_fun5b_ctrl"], root, prompts.SIM2REAL_PROMPTS, seed=7)[0]["seed"] == 7
        # un manifest senza controls/ (pacchetto --no-encode) si ferma
        with open(os.path.join(root, name, "manifest.json")) as f:
            manifest = json.load(f)
        manifest["files"] = {}
        with open(os.path.join(root, name, "manifest.json"), "w") as f:
            json.dump(manifest, f)
        expect_error(S.Sim2RealError, S.build_jobs, ["wan22_fun5b_ctrl"], root,
                     prompts.SIM2REAL_PROMPTS, contains="no-encode")
    finally:
        shutil.rmtree(root)


def test_pacchetti_in_ordine_stabile():
    root = tempfile.mkdtemp(prefix="packs_")
    try:
        for s in ("s2r_ramo", "s2r_insegna", "s2r_monopattino"):
            make_pack(root, s, "cosmos")
        os.makedirs(os.path.join(root, "s2r_ramo__wanfun_vecchio"))
        assert S.pack_names(root, "cosmos") == sorted(f"{s}__cosmos" for s in
                                                      ("s2r_ramo", "s2r_insegna", "s2r_monopattino"))
    finally:
        shutil.rmtree(root)


# --- runner --------------------------------------------------------------------------------

def test_runner_passa_il_controllo_solo_a_chi_lo_ha():
    calls = []
    t2v = types.SimpleNamespace(generate=lambda h, p, s: calls.append((p, s)) or "t2v")
    v2v = types.SimpleNamespace(generate=lambda h, p, s, c: calls.append((p, s, c)) or "v2v")
    assert runner.call_generate(t2v, None, {"prompt": "a", "seed": 1}) == "t2v"
    assert runner.call_generate(v2v, None, {"prompt": "b", "seed": 0, "control": {"k": 1}}) == "v2v"
    assert calls == [("a", 1), ("b", 0, {"k": 1})]


def test_formato_del_pacchetto_contro_lo_spec():
    spec = {"num_frames": 121, "fps": 24, "width": 1280, "height": 704}
    control = {"pack": "p", "n_frames": 121, "fps": 24.0, "size": [1280, 704]}
    assert S.check_against_spec(control, spec) is None
    for campo, valore in (("n_frames", 93), ("fps", 16), ("size", [1280, 720]), ("size", [704, 1280])):
        assert S.check_against_spec(dict(control, **{campo: valore}), spec) is not None, campo


def test_ripresa_solo_per_lo_stesso_job():
    job = {"clip_id": "s2r_ramo__A", "prompt": "p", "seed": 0, "control": {"manifest_sha256": "a"}}
    ok = dict(job, esito="ok")
    assert S.resume_mismatch(job, [ok]) is None
    assert "nessuna riga" in S.resume_mismatch(job, [])
    assert "nessuna riga" in S.resume_mismatch(job, [dict(ok, esito="errore")])
    assert "nessuna riga" in S.resume_mismatch(job, [dict(ok, clip_id="s2r_ramo__B")])
    assert "seed" in S.resume_mismatch(job, [dict(ok, seed=1)])
    assert "prompt" in S.resume_mismatch(job, [dict(ok, prompt="q")])
    assert "control" in S.resume_mismatch(job, [dict(ok, control={"manifest_sha256": "b"})])
    # conta l'ultima riga ok: una rigenerazione col job giusto sana una vecchia
    assert S.resume_mismatch(job, [dict(ok, seed=1), ok]) is None


def test_check_frames_rifiuta_nan():
    frames = np.zeros((2, 4, 4, 3), np.float32)
    frames[1, 2, 2, 0] = np.nan
    expect_error(ValueError, runner.check_frames, frames, "m", contains="NaN")
    expect_error(ValueError, runner.check_frames, np.full((1, 2, 2, 3), np.inf), "m", contains="NaN")
    assert runner.check_frames(np.zeros((1, 2, 2, 3), np.uint8), "m").dtype == np.float32


class FakeRunnerEnv:
    """generate_model senza GPU: torch.cuda, memoria, scrittura e gate sostituiti."""

    def __init__(self, shape_out):
        self.shape_out, self.loads = shape_out, 0
        cuda = types.SimpleNamespace(reset_peak_memory_stats=lambda: None,
                                     empty_cache=lambda: None,
                                     OutOfMemoryError=type("OOM", (Exception,), {}))
        self.patches = {
            "torch": types.SimpleNamespace(cuda=cuda),
            "mem_snapshot": lambda: {"vram_alloc_peak_gb": 0.0, "vram_reserved_peak_gb": 0.0,
                                     "host_avail_gb": 0.0},
            "write_video": lambda frames, path, fps: open(path, "wb").close() or "finto",
            "run_gate": lambda path: {},
        }

    def adapter(self, spec):
        env = self

        def load(prompts_list):
            env.loads += 1
            return {}

        return types.SimpleNamespace(
            SPEC=spec, load=load, unload=lambda h: None,
            generate=lambda h, p, s, c: np.full(env.shape_out, 0.5, np.float32))

    def __enter__(self):
        self.saved = {k: getattr(runner, k) for k in self.patches}
        for k, v in self.patches.items():
            setattr(runner, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(runner, k, v)
        sys.modules.pop("adapters.finto_s2r", None)


def run_model(env, spec, jobs, out_root):
    import adapters  # noqa: F401  il pacchetto deve esserci per import_module("adapters.x")
    sys.modules["adapters.finto_s2r"] = env.adapter(spec)
    costs = runner.Costs(1.0, len(jobs), 100)
    return runner.generate_model("finto_s2r", jobs, out_root, costs, dry_run=False)


def records_of(out_root):
    return runner.read_records(os.path.join(out_root, "finto_s2r", "manifest.jsonl"))


def test_runner_aggiunge_le_metriche_clip_dell_adapter():
    # Open-Sora genera in un sottoprocesso: la VRAM del runner non e' la sua, e
    # l'adapter passa il picco vero con metriche_clip(handle).
    spec = {"num_frames": 6, "fps": 24, "width": 8, "height": 4}
    root = tempfile.mkdtemp(prefix="s2r_metriche_")
    try:
        packs, out = os.path.join(root, "packs"), os.path.join(root, "out")
        os.makedirs(packs)
        make_pack(packs, "s2r_ramo", "wanfun", n_frames=6, fps=24, size=(8, 4))
        jobs = [dict(j, modello="finto_s2r") for j in
                S.build_jobs(["wan22_fun5b_ctrl"], packs, prompts.SIM2REAL_PROMPTS)][:1]
        with FakeRunnerEnv((6, 4, 8, 3)) as env:
            originale = env.adapter

            def con_metriche(spec_):
                a = originale(spec_)
                a.metriche_clip = lambda h: {"vram_alloc_peak_gb_torchrun": 52.5}
                return a

            env.adapter = con_metriche
            esiti = run_model(env, spec, jobs, out)
        assert esiti["ok"] == 1, esiti
        assert records_of(out)[0]["vram_alloc_peak_gb_torchrun"] == 52.5
    finally:
        shutil.rmtree(root)


def test_runner_sim2real_end_to_end():
    spec = {"num_frames": 6, "fps": 24, "width": 8, "height": 4}
    root = tempfile.mkdtemp(prefix="s2r_runner_")
    try:
        packs, out = os.path.join(root, "packs"), os.path.join(root, "out")
        os.makedirs(packs)
        make_pack(packs, "s2r_ramo", "wanfun", n_frames=6, fps=24, size=(8, 4))
        jobs = [dict(j, modello="finto_s2r") for j in
                S.build_jobs(["wan22_fun5b_ctrl"], packs, prompts.SIM2REAL_PROMPTS)]
        mp4_a, frames_a, _ = runner.clip_paths(out, jobs[0])

        # 1. un frame in meno: errore, e niente mp4 ne' PNG che alla ripresa
        #    passerebbero per una clip completata
        with FakeRunnerEnv((5, 4, 8, 3)) as env:
            esiti = run_model(env, spec, jobs, out)
        assert esiti["errore"] == 2 and esiti["ok"] == 0, esiti
        assert not os.path.exists(mp4_a) and not os.path.exists(frames_a)
        assert all("GT" in r["errore"] for r in records_of(out))

        # 2. forma giusta: ok, tutti i 6 frame su disco
        with FakeRunnerEnv((6, 4, 8, 3)) as env:
            esiti = run_model(env, spec, jobs, out)
        assert esiti["ok"] == 2, esiti
        assert os.path.exists(mp4_a) and len(os.listdir(frames_a)) == 6

        # 3. rilancio con lo stesso job: saltate, il modello non si carica nemmeno
        with FakeRunnerEnv((6, 4, 8, 3)) as env:
            esiti = run_model(env, spec, jobs, out)
            assert esiti["saltate"] == 2 and env.loads == 0, esiti

        # 4. rilancio con un seed diverso: si ferma invece di tenere la clip vecchia
        cambiati = [dict(j, seed=7) for j in jobs]
        with FakeRunnerEnv((6, 4, 8, 3)) as env:
            expect_error(SystemExit, run_model, env, spec, cambiati, out, contains="seed")
    finally:
        shutil.rmtree(root)


def test_runner_sim2real_controlli_prima_della_gpu():
    spec = {"num_frames": 6, "fps": 24, "width": 8, "height": 4}
    root = tempfile.mkdtemp(prefix="s2r_runner_")
    try:
        packs, out = os.path.join(root, "packs"), os.path.join(root, "out")
        os.makedirs(packs)
        make_pack(packs, "s2r_ramo", "wanfun", n_frames=6, fps=24, size=(8, 4))
        jobs = [dict(j, modello="finto_s2r") for j in
                S.build_jobs(["wan22_fun5b_ctrl"], packs, prompts.SIM2REAL_PROMPTS)]

        # formato del pacchetto diverso dallo spec: fallisce senza caricare il modello
        with FakeRunnerEnv((6, 4, 8, 3)) as env:
            esiti = run_model(env, dict(spec, fps=16), jobs, out)
            assert esiti["errore"] == 2 and env.loads == 0, esiti
        assert all("pacchetti" in r["errore"] for r in records_of(out))

        # pacchetti in un'altra radice (jobs.jsonl fatto sul PC): fallisce prima del load
        altrove = [dict(j, control=dict(j["control"], root=os.path.join(root, "pc"))) for j in jobs]
        with FakeRunnerEnv((6, 4, 8, 3)) as env:
            esiti = run_model(env, spec, altrove, os.path.join(root, "out_pc"))
            assert esiti["errore"] == 2 and env.loads == 0, esiti

        # video del pacchetto cambiato dopo build_jobs: errore, non una clip coi video nuovi
        with open(os.path.join(packs, "s2r_ramo__wanfun", "controls", "depth.mp4"), "ab") as f:
            f.write(b"x")
        with FakeRunnerEnv((6, 4, 8, 3)) as env:
            esiti = run_model(env, spec, jobs, os.path.join(root, "out2"))
        assert esiti == {"ok": 1, "oom": 0, "errore": 1, "saltate": 0}, esiti
        errori = [r for r in records_of(os.path.join(root, "out2")) if r["esito"] == "errore"]
        assert errori[0]["clip_id"] == "s2r_ramo__A" and "cambiato" in errori[0]["errore"]
    finally:
        shutil.rmtree(root)


def test_build_jobs_rifiuta_adapter_sim2real_fuori_dal_suo_insieme():
    import subprocess
    out = os.path.join(tempfile.mkdtemp(prefix="bj_"), "jobs.jsonl")
    r = subprocess.run([sys.executable, os.path.join(HERE, "build_jobs.py"), "--models",
                        "cosmos_t25", "--set", "bakeoff", "--out", out],
                       capture_output=True, text=True, check=False)
    assert r.returncode != 0 and "--set sim2real" in r.stderr and not os.path.exists(out), r.stderr
    shutil.rmtree(os.path.dirname(out))


def test_runner_salva_tutti_i_frame_se_richiesto():
    frames = np.random.default_rng(0).random((25, 8, 12, 3)).astype(np.float32)
    root = tempfile.mkdtemp(prefix="frames_")
    try:
        idx = runner.save_frames(frames, os.path.join(root, "tutti"), all_frames=True)
        assert idx == list(range(25)) and len(os.listdir(os.path.join(root, "tutti"))) == 25
        idx = runner.save_frames(frames, os.path.join(root, "eval"))
        assert len(idx) == runner.EVAL_FRAMES == len(os.listdir(os.path.join(root, "eval")))
    finally:
        shutil.rmtree(root)


# --- adapter --------------------------------------------------------------------------------

def test_spec_degli_adapter():
    from adapters import cosmos_t25, wan22_fun5b_ctrl
    c, w = cosmos_t25.SPEC, wan22_fun5b_ctrl.SPEC
    assert c["sigma_max"] is None and c["configs"] == S.CONFIGS["cosmos_t25"]
    assert (c["width"], c["height"], c["num_frames"], c["fps"]) == (1280, 720, 93, 16)
    assert c["negative"] == prompts.SIM2REAL_NEGATIVE_COSMOS
    assert w["sample_size"] == [704, 1280] and (w["height"], w["width"]) == (704, 1280)
    assert w["teacache"] is False and w["gpu_memory_mode"] == "model_full_load"
    assert (w["num_frames"], w["fps"], w["steps"], w["guidance"]) == (121, 24, 40, 6.0)
    assert w["negative"] == prompts.NEGATIVE_PROMPT


FAKE_INFERENCE = r'''
import json, os, sys
args = sys.argv[1:]
spec = json.load(open(args[args.index("-i") + 1]))
json.dump(spec, open(os.environ["FAKE_COSMOS_LOG"], "w"))
out = args[args.index("-o") + 1]
os.makedirs(out, exist_ok=True)
open(os.path.join(out, spec["name"] + "_control_depth.mp4"), "wb").write(b"controllo")
sys.exit(int(os.environ.get("FAKE_COSMOS_EXIT", "0")))
'''


def test_cosmos_scrive_lo_spec_e_segnala_l_assenza_del_video():
    from adapters import cosmos_t25
    root = tempfile.mkdtemp(prefix="cosmos_")
    try:
        repo = os.path.join(root, "repo")
        os.makedirs(os.path.join(repo, "examples"))
        with open(os.path.join(repo, "examples", "inference.py"), "w") as f:
            f.write(FAKE_INFERENCE)
        make_pack(root, "s2r_ramo", "cosmos")
        control = control_of("cosmos_t25", "B", root)
        os.environ["FAKE_COSMOS_LOG"] = os.path.join(root, "spec.json")
        handle = {"repo": repo, "python": sys.executable}
        os.environ["FAKE_COSMOS_EXIT"] = "0"
        # uscita 0 ma solo il video del controllo: non va scambiato per la clip
        expect_error(RuntimeError, cosmos_t25.generate, handle, "p", 0, control,
                     contains="nessun video")
        with open(os.environ["FAKE_COSMOS_LOG"]) as f:
            spec = json.load(f)
        assert spec["seed"] == 0 and "sigma_max" not in spec and spec["edge"]["control_weight"] == 0.5
        assert spec["negative_prompt"] == prompts.SIM2REAL_NEGATIVE_COSMOS
        os.environ["FAKE_COSMOS_EXIT"] = "3"
        expect_error(RuntimeError, cosmos_t25.generate, handle, "p", 0, control,
                     contains="uscito con 3")
    finally:
        os.environ.pop("FAKE_COSMOS_EXIT", None)
        shutil.rmtree(root)


def test_cosmos_successo_seed_e_comando():
    from adapters import cosmos_t25
    root = tempfile.mkdtemp(prefix="cosmos_")
    letti, saved = [], cosmos_t25._read_video
    try:
        repo = os.path.join(root, "repo")
        os.makedirs(os.path.join(repo, "examples"))
        with open(os.path.join(repo, "examples", "inference.py"), "w") as f:
            extra = ('open(os.path.join(out, spec["name"] + ".mp4"), "wb").write(b"clip")\n'
                     'open(os.environ["FAKE_COSMOS_ARGS"], "w").write(json.dumps(sys.argv[1:]))\n')
            f.write(FAKE_INFERENCE.replace("sys.exit(", extra + "sys.exit("))
        make_pack(root, "s2r_ramo", "cosmos")
        control = control_of("cosmos_t25", "A", root)
        os.environ["FAKE_COSMOS_LOG"] = os.path.join(root, "spec.json")
        os.environ["FAKE_COSMOS_ARGS"] = os.path.join(root, "args.json")
        os.environ["FAKE_COSMOS_EXIT"] = "0"

        def fake_read(path):
            with open(path, "rb") as f:
                letti.append((path, f.read()))
            return "frame"

        cosmos_t25._read_video = fake_read
        handle = {"repo": repo, "python": sys.executable}
        assert cosmos_t25.generate(handle, "p", 7, control) == "frame"
        assert letti[0][1] == b"clip" and os.path.basename(letti[0][0]) == "clip.mp4"
        # la cartella di lavoro si cancella dopo la clip
        assert not os.path.exists(os.path.dirname(os.path.dirname(letti[0][0])))
        with open(os.environ["FAKE_COSMOS_LOG"]) as f:
            spec = json.load(f)
        assert (spec["seed"], spec["num_steps"], spec["guidance"]) == (7, 35, 3)
        # COSMOS_SMOKE spegnerebbe i guardrail: non deve arrivare al sottoprocesso
        os.environ["COSMOS_SMOKE"] = "1"
        assert "COSMOS_SMOKE" not in cosmos_t25._env()
        assert cosmos_t25._env()["FAKE_COSMOS_LOG"] == os.environ["FAKE_COSMOS_LOG"]
        del os.environ["COSMOS_SMOKE"]
        with open(os.environ["FAKE_COSMOS_ARGS"]) as f:
            args = json.load(f)
        assert os.path.dirname(letti[0][0]) == args[args.index("-o") + 1]
        # uscita non zero CON l'mp4 presente: resta un errore, non una clip
        os.environ["FAKE_COSMOS_EXIT"] = "3"
        expect_error(RuntimeError, cosmos_t25.generate, handle, "p", 7, control, contains="uscito con 3")
        try:
            cosmos_t25.generate(handle, "p", 7, control)
        except RuntimeError as exc:
            assert "nessun video" not in str(exc)
    finally:
        cosmos_t25._read_video = saved
        for k in ("FAKE_COSMOS_EXIT", "FAKE_COSMOS_ARGS"):
            os.environ.pop(k, None)
        shutil.rmtree(root)


def test_wan_argomenti_della_pipeline():
    import torch

    from adapters import wan22_fun5b_ctrl as W
    kw = W.pipe_kwargs("gen", "vid", "mask", "ctrl", 0.875)
    assert (kw["height"], kw["width"], kw["num_frames"]) == (704, 1280, 121)
    assert (kw["guidance_scale"], kw["num_inference_steps"], kw["shift"]) == (6.0, 40, 5)
    assert kw["negative_prompt"] == prompts.NEGATIVE_PROMPT and kw["generator"] == "gen"
    assert (kw["video"], kw["mask_video"], kw["control_video"], kw["boundary"]) == (
        "vid", "mask", "ctrl", 0.875)
    sample = torch.arange(1 * 3 * 4 * 5 * 6, dtype=torch.float32).reshape(1, 3, 4, 5, 6)
    out = W.to_frames(sample)
    assert out.shape == (4, 5, 6, 3) and out.dtype == np.float32
    assert out[2, 3, 4, 1] == sample[0, 1, 2, 3, 4].item()


def test_build_jobs_da_riga_di_comando():
    import subprocess
    root = tempfile.mkdtemp(prefix="bj_")
    try:
        packs = os.path.join(root, "packs")
        os.makedirs(packs)
        make_pack(packs, "s2r_ramo", "cosmos")

        def run(*extra):
            out = os.path.join(root, "jobs.jsonl")
            if os.path.exists(out):
                os.remove(out)
            r = subprocess.run([sys.executable, os.path.join(HERE, "build_jobs.py"), "--models",
                                "cosmos_t25", "--set", "sim2real", "--packs", packs, "--out", out,
                                *extra], capture_output=True, text=True, check=False)
            jobs = []
            if os.path.exists(out):
                with open(out, encoding="utf-8") as f:
                    jobs = [json.loads(line) for line in f]
            return r, jobs

        r, jobs = run()
        assert r.returncode == 0 and [j["clip_id"] for j in jobs] == ["s2r_ramo__A", "s2r_ramo__B"], r.stderr
        assert all(j["seed"] == 0 and j["save_all_frames"] for j in jobs)
        r, jobs = run("--limit", "1")
        assert len(jobs) == 1
        r, jobs = run("--seeds", "2")
        assert r.returncode != 0 and not jobs
        os.remove(os.path.join(packs, "s2r_ramo__cosmos", "controls", "edge.mp4"))
        r, jobs = run()
        assert r.returncode != 0 and "ERRORE" in r.stderr and not jobs, r.stderr
    finally:
        shutil.rmtree(root)


def test_cosmos_sceglie_la_clip_e_non_i_controlli():
    from adapters import cosmos_t25
    root = tempfile.mkdtemp(prefix="out_")
    try:
        for name in ("clip_control_depth.mp4", "clip_control_seg.mp4"):
            open(os.path.join(root, name), "wb").close()
        assert cosmos_t25._output_video(root, "clip") is None
        open(os.path.join(root, "clip.mp4"), "wb").close()
        assert cosmos_t25._output_video(root, "clip") == os.path.join(root, "clip.mp4")
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok     {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FALLITO {name} {exc}")
    print(f"\n{len(tests)} test, {failed} falliti")
    sys.exit(1 if failed else 0)
