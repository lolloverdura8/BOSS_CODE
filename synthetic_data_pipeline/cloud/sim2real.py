# sim2real.py
#
# La parte del ramo Sim2Real (Fase 3) che non tocca la GPU: quali controlli usa
# ciascuna configurazione, come i pacchetti di CARLA/prepare_controls.py diventano
# job, come un job diventa lo spec di Cosmos, e il controllo che l'uscita di un
# modello sia allineata alla GT del pacchetto.
#
# Sta in un modulo a parte, con la sola standard library, perche' lo usano in
# quattro: build_jobs.py (sul PC e sul pod), runner.py nel venv gen e nel venv fun,
# e i due adapter. Cosi' si prova tutto in locale, senza torch ne' modelli.
#
# I pacchetti sul pod stanno in una radice (default /workspace/s2r_packs) con dentro
# <scenario>__<cosmos|wanfun>/manifest.json e controls/*.mp4: e' cio' che si carica
# dal PC, la GT resta li'.
import hashlib
import json
import os

# Quale pacchetto legge ciascun adapter: il formato (fps, frame, ritaglio) e' gia'
# deciso da prepare_controls.py.
PACK_SUFFIX = {"cosmos_t25": "cosmos", "wan22_fun5b_ctrl": "wanfun"}

# Le due configurazioni per modello (piano Fase 3 del 12/10). Per Cosmos i pesi dei
# controlli; Wan Fun accetta un solo control_video, quindi il peso non c'e'.
#   Cosmos A: solo depth, peso 1,0 (come gli esempi del repo)
#   Cosmos B: edge 0,5 + seg 0,9 dalla ricetta CARLA (rain.toml), depth 0,5
#   Wan Fun A: depth; Wan Fun B: edge, lo stesso Canny mascherato di Cosmos
CONFIGS = {
    "cosmos_t25": {"A": {"depth": 1.0}, "B": {"edge": 0.5, "seg": 0.9, "depth": 0.5}},
    "wan22_fun5b_ctrl": {"A": {"depth": None}, "B": {"edge": None}},
}

# Cosmos vuole sempre video_path: senza sigma_max non lo usa come punto di partenza
# (inference_pipeline.py, primo chunk senza frame condizionanti), ne prende solo
# dimensioni e fps. Gli si da' l'RGB di CARLA del pacchetto.
COSMOS_VIDEO = "rgb"


class Sim2RealError(RuntimeError):
    """Pacchetto incompleto o uscita disallineata: proseguire darebbe dati sbagliati."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pack(root, name):
    """(manifest, sha del manifest) di un pacchetto, dopo aver controllato i video.

    Ogni controls/*.mp4 deve esserci e avere lo sha scritto nel manifest: un upload
    a meta' si scopre qui, prima di accendere la GPU, e non dopo una clip sbagliata.
    """
    path = os.path.join(root, name, "manifest.json")
    if not os.path.exists(path):
        raise Sim2RealError(f"manifest assente: {path}")
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    controls = {k: v for k, v in manifest["files"].items() if k.startswith("controls/")}
    if not controls:
        raise Sim2RealError(f"{name}: il manifest non elenca controls/*.mp4 (pacchetto --no-encode?)")
    for rel, digest in controls.items():
        full = os.path.join(root, name, rel)
        if not os.path.exists(full):
            raise Sim2RealError(f"{name}: manca {rel}")
        if sha256_file(full) != digest:
            raise Sim2RealError(f"{name}: {rel} non coincide con lo sha del manifest")
    return manifest, sha256_file(path)


def pack_names(root, suffix):
    """Le cartelle <scenario>__<suffix> della radice, in ordine stabile."""
    if not os.path.isdir(root):
        raise Sim2RealError(f"cartella dei pacchetti inesistente: {root}")
    return sorted(d for d in os.listdir(root)
                  if d.endswith("__" + suffix) and os.path.isdir(os.path.join(root, d)))


def control_entry(root, name, manifest, key):
    rel = f"controls/{key}.mp4"
    if rel not in manifest["files"]:
        raise Sim2RealError(f"{name}: il manifest non ha {rel}")
    return {"path": f"{name}/{rel}", "sha256": manifest["files"][rel]}


def job_control(root, name, manifest, manifest_sha, model, config):
    """Il campo control di un job: tutto cio' che serve all'adapter e alla Fase 4."""
    weights = CONFIGS[model][config]
    controls = {}
    for key, weight in weights.items():
        controls[key] = dict(control_entry(root, name, manifest, key), weight=weight)
    return {
        "root": os.path.abspath(root),
        "pack": name,
        "config": config,
        "scenario": manifest["session"]["scenario"],
        "manifest_sha256": manifest_sha,
        "controls": controls,
        "video": control_entry(root, name, manifest, COSMOS_VIDEO),
        "n_frames": manifest["n_frames"],
        "fps": manifest["fps"],
        "size": manifest["resolution"],                     # [larghezza, altezza]
        # Righe tolte sopra e sotto ai 720 di CARLA (Wan Fun: 8 + 8). Senza, la Fase 4
        # confronterebbe la clip con la GT spostata di 8 pixel in verticale.
        "crop_rows": manifest["crop_rows"],
    }


def build_jobs(models, root, prompts_by_scenario, seed=0):
    """Un job per pacchetto e configurazione: <scenario>__A, <scenario>__B."""
    jobs = []
    for model in models:
        if model not in PACK_SUFFIX:
            raise Sim2RealError(f"{model} non e' un adapter sim2real ({sorted(PACK_SUFFIX)})")
        names = pack_names(root, PACK_SUFFIX[model])
        if not names:
            raise Sim2RealError(f"nessun pacchetto __{PACK_SUFFIX[model]} in {root}")
        for name in names:
            manifest, manifest_sha = load_pack(root, name)
            scenario = manifest["session"]["scenario"]
            if scenario not in prompts_by_scenario:
                raise Sim2RealError(f"nessun prompt per la scena {scenario}")
            boss_class = manifest["obstacle"]["parts"][0]["boss_class"]
            for config in sorted(CONFIGS[model]):
                jobs.append({
                    "modello": model, "clip_id": f"{scenario}__{config}", "classe": boss_class,
                    "prompt_idx": None, "prompt": prompts_by_scenario[scenario], "seed": seed,
                    "save_all_frames": True,
                    "control": job_control(root, name, manifest, manifest_sha, model, config),
                })
    return jobs


def control_path(control, key):
    """Percorso assoluto di un video di controllo del job (key "video" = l'RGB)."""
    entry = control["video"] if key == "video" else control["controls"][key]
    return os.path.join(control["root"], entry["path"])


def verify_control_files(control):
    """Ricontrolla gli sha dei video del job subito prima di generare.

    build_jobs li ha gia' controllati, ma i job restano scritti in jobs.jsonl: un
    pacchetto ricaricato dopo darebbe una clip fatta coi video nuovi e registrata
    con lo sha dei vecchi. Sono un centinaio di MB, si rileggono in un secondo.
    """
    entries = {"video": control["video"], **control["controls"]}
    for key, entry in entries.items():
        path = control_path(control, key)
        if not os.path.exists(path):
            raise Sim2RealError(f"{control['pack']}: manca {path}")
        if sha256_file(path) != entry["sha256"]:
            raise Sim2RealError(f"{control['pack']}: {entry['path']} e' cambiato dopo build_jobs")


def missing_files(control):
    """I video del job che non ci sono. root e' assoluta, fissata da build_jobs: un
    jobs.jsonl costruito sul PC punterebbe a percorsi che sul pod non esistono."""
    keys = ["video", *control["controls"]]
    return [p for p in (control_path(control, k) for k in keys) if not os.path.exists(p)]


def check_against_spec(control, spec):
    """None se frame, fps e dimensioni del pacchetto sono quelli dell'adapter.

    Va chiamata prima di caricare il modello: un pacchetto del formato sbagliato
    altrimenti si scopre solo dopo una generazione intera, o non si scopre affatto
    se il modello ricampiona in silenzio.
    """
    expected = (spec["num_frames"], float(spec["fps"]), [spec["width"], spec["height"]])
    got = (control["n_frames"], float(control["fps"]), list(control["size"]))
    if got != expected:
        return (f"pacchetto {control['pack']} (frame, fps, [L, A]) = {got}, l'adapter vuole"
                f" {expected}")
    return None


# Cio' che deve coincidere perche' una clip gia' su disco valga per il job di adesso.
RESUME_KEYS = ("prompt", "seed", "control")


def resume_mismatch(job, records):
    """None se la clip su disco e' stata fatta per questo stesso job, altrimenti il motivo.

    Il runner salta una clip se l'mp4 c'e', guardando solo clip_id. Per il Sim2Real
    clip_id e' <scenario>__A: con pacchetti, pesi o prompt cambiati la clip vecchia
    passerebbe per nuova. records sono le righe del manifest.jsonl del modello.
    """
    ok = [r for r in records if r.get("clip_id") == job["clip_id"] and r.get("esito") == "ok"]
    if not ok:
        return f"{job['clip_id']}: c'e' l'mp4 ma nessuna riga \"ok\" nel manifest"
    diverse = [k for k in RESUME_KEYS if ok[-1].get(k) != job.get(k)]
    if diverse:
        return f"{job['clip_id']}: la clip su disco e' stata fatta con {', '.join(diverse)} diversi"
    return None


def cosmos_spec(control, prompt, negative, seed, name, num_steps, guidance):
    """Lo spec JSON di examples/inference.py di Cosmos-Transfer2.5.

    sigma_max assente di proposito: con sigma_max l'RGB di CARLA diventerebbe il
    punto di partenza del denoising e la clip ne erediterebbe l'aspetto da videogioco.
    """
    spec = {
        "name": name,
        "prompt": prompt,
        "negative_prompt": negative,
        "seed": seed,
        "guidance": guidance,
        "num_steps": num_steps,
        "video_path": control_path(control, "video"),
    }
    for key, entry in control["controls"].items():
        spec[key] = {"control_path": control_path(control, key), "control_weight": entry["weight"]}
    return spec


def wan_control_video(control):
    """Il solo control_video di Wan Fun: la configurazione ne ha esattamente uno."""
    if len(control["controls"]) != 1:
        raise Sim2RealError(f"Wan Fun accetta un controllo solo, ricevuti {sorted(control['controls'])}")
    return control_path(control, next(iter(control["controls"])))


def check_control_output(shape, control):
    """None se (T, H, W, 3) coincide col pacchetto, altrimenti il motivo.

    Una clip con un frame in piu' o in meno, o con un'altezza diversa, ha la GT
    sfasata: la Fase 4 confronterebbe frame che non si corrispondono.
    """
    width, height = control["size"]
    expected = (control["n_frames"], height, width, 3)
    if tuple(shape) != expected:
        return (f"uscita {tuple(shape)} diversa dal pacchetto {control['pack']} {expected}:"
                " la GT non sarebbe allineata")
    return None
