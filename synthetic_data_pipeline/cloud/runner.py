# runner.py
#
# Generatore a lotti, indipendente dal modello: legge jobs.jsonl, carica
# l'adapter richiesto, genera, scrive. Gira sul pod.
#
#   python runner.py --jobs jobs.jsonl --out-root /workspace/out --usd-per-hour 1.99
#   python runner.py --jobs jobs.jsonl --dry-run
#   python runner.py --jobs jobs.jsonl --models wan22_5b --limit 1
#
# Uscite, in /workspace/out/<modello>/:
#   {clip_id}.mp4              il video, per guardarlo
#   frames/{clip_id}/NNN.png   EVAL_FRAMES frame a passo costante, PNG lossless
#                              presi dall'array dell'adapter e NON estratti
#                              dall'mp4: gli annotatori devono vedere i pixel
#                              generati, non quelli ricompressi in H.264
#   manifest.jsonl             una riga per tentativo di clip
#
# I cinque requisiti che il PIANO_ATTACCO_OR4.1 pone al runner sul pod:
# riprendibilita', scrittura atomica, cattura del SIGTERM, gate in linea,
# contatore dei costi. I primi due venivano gia' da generate_test_clips.py.
import argparse
import gc
import importlib
import json
import os
import signal
import sys
import time

import numpy as np
import psutil
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# NORMALIZZAZIONE CHE RENDE CONFRONTABILI I MODELLI. Ogni generatore ha il suo
# conteggio nativo di frame (73, 81, 121...), ma il giudizio di SAM e DA3 deve
# poggiare sullo stesso numero di campioni per clip, altrimenti un modello che
# ne produce di piu' porta piu' peso nelle medie.
#
# 19 e' il numero che veniva dal passo di 4 frame del reference set CARLA (B.1):
# frame consecutivi sono quasi identici e per un detector valgono uno solo.
# A 73 frame questo campionamento coincide ESATTAMENTE con lo stride 4 storico
# (0, 4, 8, ... 72), quindi non cambia nulla per le clip gia' prodotte.
EVAL_FRAMES = 19

PROC = psutil.Process()

# Alzato dal gestore di SIGTERM: RunPod lo manda prima di revocare un'istanza.
# Non si puo' interrompere un pipe() a meta', quindi la clip in corso finisce e
# ci si ferma prima della successiva. Una clip persa a meta' non e' un problema:
# la scrittura atomica fa si' che non lasci un mp4 che sembra completo.
stop_requested = False


def _on_stop(signum, frame):
    global stop_requested
    stop_requested = True
    print("\n[%s ricevuto] finisco la clip in corso e mi fermo. Il rilancio"
          " riprende da qui." % signal.Signals(signum).name, flush=True)


def mem_snapshot():
    """Le cinque metriche di vram_benchmark.py, stessi nomi e stesse unita' (GiB)."""
    free, total = torch.cuda.mem_get_info()
    return {
        "vram_alloc_peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "vram_reserved_peak_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "vram_device_used_gb": round((total - free) / 1024**3, 3),
        "host_rss_gb": round(PROC.memory_info().rss / 1024**3, 3),
        "host_avail_gb": round(psutil.virtual_memory().available / 1024**3, 3),
    }


def eval_indices(n_frames, k=EVAL_FRAMES):
    """k indici a passo costante fra 0 e n_frames-1, estremi inclusi."""
    if n_frames <= k:
        return list(range(n_frames))
    return sorted(set(int(round(x)) for x in np.linspace(0, n_frames - 1, k)))


def save_frames(frames, frame_dir):
    from PIL import Image
    os.makedirs(frame_dir, exist_ok=True)
    idx = eval_indices(len(frames))
    for i in idx:
        img = (np.clip(frames[i], 0.0, 1.0) * 255.0).round().astype(np.uint8)
        Image.fromarray(img).save(os.path.join(frame_dir, "%03d.png" % i))
    return idx


def write_video(frames, path, fps):
    """Scrive l'mp4. export_to_video di diffusers e' cio' che ha prodotto le clip
    esistenti, quindi resta la prima scelta; ma gli adapter di Open-Sora e LTX
    girano in ambienti dove diffusers non c'e', e li' si scrive con imageio.
    Sotto il cofano export_to_video usa comunque imageio, quindi il codec e' lo
    stesso (H.264): il brand nell'header distingue i due casi, isomiso2avc1mp41
    per H.264 contro isomiso2mp41 del ramo OpenCV deprecato."""
    try:
        from diffusers.utils import export_to_video
    except ImportError:
        import imageio
        arr = [(np.clip(f, 0.0, 1.0) * 255.0).round().astype(np.uint8) for f in frames]
        imageio.mimsave(path, arr, fps=fps, codec="libx264")
        return "imageio"
    export_to_video(frames, path, fps=fps)
    return "diffusers"


def run_gate(path):
    """Nitidezza e SSIM sulla clip appena scritta. Gira su CPU mentre la GPU fa
    la prossima: in pratica e' gratis.

    I VALORI SI REGISTRANO, NON SI FILTRA. Le soglie di quality_gate.py sono
    calibrate a 720x480 e non valgono alle risoluzioni native degli altri
    modelli: usarle come pass/fail qui scarterebbe clip buone. La
    ricalibrazione per risoluzione e' un lavoro a parte.
    """
    try:
        import quality_gate
    except ImportError as e:
        return {"gate_sharpness": None, "gate_ssim": None, "gate_note": "cv2 assente: %s" % e}
    try:
        r = quality_gate.evaluate_clip(path)
        return {"gate_sharpness": r.get("sharpness_median"),
                "gate_ssim": r.get("ssim_median"),
                "gate_note": r.get("reason") or ""}
    except Exception as e:  # il gate non deve mai fermare il lotto
        return {"gate_sharpness": None, "gate_ssim": None, "gate_note": "errore gate: %s" % e}


def load_jobs(path, models, limit):
    jobs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                jobs.append(json.loads(line))
    if models:
        jobs = [j for j in jobs if j["modello"] in models]
    if limit:
        # per modello, non in totale: --limit 1 deve dare una clip per ciascuno
        seen, out = {}, []
        for j in jobs:
            n = seen.get(j["modello"], 0)
            if n < limit:
                out.append(j)
                seen[j["modello"]] = n + 1
        jobs = out
    return jobs


def clip_paths(out_root, job):
    d = os.path.join(out_root, job["modello"])
    return (os.path.join(d, job["clip_id"] + ".mp4"),
            os.path.join(d, "frames", job["clip_id"]),
            os.path.join(d, "manifest.jsonl"))


class Costs:
    """Contatore dei costi. Su RunPod si paga il pod acceso, non la GPU usata:
    sapere a meta' lotto che si sta andando al doppio del previsto permette di
    fermarsi, invece di scoprirlo dalla fattura."""

    def __init__(self, usd_per_hour, total_jobs, every):
        self.rate = usd_per_hour
        self.total = total_jobs
        self.every = every
        self.t0 = time.time()
        self.done = 0

    def tick(self, esito):
        self.done += 1
        if self.done % self.every and self.done != self.total:
            return
        el = time.time() - self.t0
        per = el / self.done
        left = (self.total - self.done) * per
        print("  [costi] %d/%d clip, %.1f s/clip, trascorse %.2f h, spesi $%.2f;"
              " restano %.2f h, totale previsto $%.2f"
              % (self.done, self.total, per, el / 3600, el / 3600 * self.rate,
                 left / 3600, (el + left) / 3600 * self.rate), flush=True)


def generate_model(name, jobs, out_root, costs, dry_run):
    adapter = importlib.import_module("adapters.%s" % name)
    spec = dict(adapter.SPEC)
    print("\n=== %s ===" % name)
    print("spec: %s" % json.dumps(spec, ensure_ascii=False, default=str))

    mp4_dir = os.path.join(out_root, name)
    os.makedirs(os.path.join(mp4_dir, "frames"), exist_ok=True)
    manifest = os.path.join(mp4_dir, "manifest.jsonl")

    todo = [j for j in jobs if not os.path.exists(clip_paths(out_root, j)[0])]
    print("clip richieste: %d, gia' presenti: %d, da generare: %d"
          % (len(jobs), len(jobs) - len(todo), len(todo)))
    if not todo or dry_run:
        return {"ok": 0, "oom": 0, "errore": 0, "saltate": len(jobs) - len(todo)}

    # load() riceve i prompt perche' gli adapter con un text encoder pesante
    # (UMT5-XXL di Wan pesa 11,36 GB, quasi meta' del modello) li codificano
    # tutti in blocco e poi lo rilasciano: e' l'ottimizzazione che fa entrare
    # Wan2.2-5B in 16 GB. Chi non ne ha bisogno ignora l'argomento.
    handle = adapter.load([j["prompt"] for j in todo])
    esiti = {"ok": 0, "oom": 0, "errore": 0, "saltate": len(jobs) - len(todo)}
    try:
        for n, job in enumerate(todo):
            print("\n[%d/%d] %s / %s (seed %d)"
                  % (n + 1, len(todo), name, job["clip_id"], job["seed"]), flush=True)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            frames, esito, errore = None, "ok", ""
            try:
                frames = adapter.generate(handle, job["prompt"], job["seed"])
            except torch.cuda.OutOfMemoryError:
                esito = "oom"
                # un'eccezione a meta' pipe() salta maybe_free_model_hooks: senza
                # questa chiamata il modulo che era in GPU ci resta anche dopo
                if hasattr(adapter, "on_oom"):
                    adapter.on_oom(handle)
            except Exception as e:
                esito, errore = "errore", "%s: %s" % (type(e).__name__, e)
                print("  ERRORE: %s" % errore, flush=True)
            elapsed = time.time() - t0
            snap = mem_snapshot()

            record = dict(job, esito=esito, tempo_s=round(elapsed, 1),
                          eval_frames=EVAL_FRAMES, errore=errore,
                          adapter_spec=spec,
                          usd_stimati=round(elapsed / 3600 * costs.rate, 4), **snap)

            if frames is not None:
                mp4, frame_dir, _ = clip_paths(out_root, job)
                idx = save_frames(frames, frame_dir)
                # prima un file temporaneo, poi il nome definitivo: l'mp4 e' il
                # marcatore di clip completata, e un file troncato da
                # un'interruzione non deve sembrarlo
                tmp = mp4 + ".tmp"
                record["video_backend"] = write_video(frames, tmp, spec["fps"])
                os.replace(tmp, mp4)
                record["num_frames_generati"] = len(frames)
                record["frame_indices"] = idx
                record.update(run_gate(mp4))

            with open(manifest, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

            esiti[esito] += 1
            print("  esito %s, %.1f s, picco allocato %.2f GB, prenotato %.2f GB, RAM host"
                  " libera %.2f GB" % (esito, elapsed, snap["vram_alloc_peak_gb"],
                                       snap["vram_reserved_peak_gb"], snap["host_avail_gb"]),
                  flush=True)
            costs.tick(esito)

            del frames
            gc.collect()
            torch.cuda.empty_cache()

            if stop_requested:
                break
    finally:
        adapter.unload(handle)
        gc.collect()
        torch.cuda.empty_cache()
    return esiti


def main():
    parser = argparse.ArgumentParser(description="Genera le clip di jobs.jsonl, un modello alla volta.")
    parser.add_argument("--jobs", default="jobs.jsonl")
    parser.add_argument("--out-root", default="outputs",
                        help="una sottocartella per modello; sul pod /workspace/out")
    parser.add_argument("--models", default=None,
                        help="solo questi adapter, separati da virgola")
    parser.add_argument("--limit", type=int, default=None,
                        help="solo le prime N clip PER MODELLO, per una prova")
    parser.add_argument("--usd-per-hour", type=float, default=0.0,
                        help="tariffa del pod, per il contatore dei costi")
    parser.add_argument("--cost-every", type=int, default=10,
                        help="ogni quante clip stampare il contatore")
    parser.add_argument("--dry-run", action="store_true",
                        help="elenca cosa farebbe senza caricare nessun modello")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")] if args.models else None
    jobs = load_jobs(args.jobs, models, args.limit)
    if not jobs:
        sys.exit("nessun job da %s" % args.jobs)

    ordine = []
    for j in jobs:
        if j["modello"] not in ordine:
            ordine.append(j["modello"])

    print("job: %d su %d modelli (%s)" % (len(jobs), len(ordine), ", ".join(ordine)))
    print("uscite: %s/<modello>/" % args.out_root)
    print("frame valutati per clip: %d" % EVAL_FRAMES)

    if args.dry_run:
        print("\n--- dry run: nessun modello caricato ---")
        for j in jobs:
            mp4 = clip_paths(args.out_root, j)[0]
            print("  %-14s %-24s seed %-5d %s %s"
                  % (j["modello"], j["clip_id"], j["seed"],
                     "GIA' PRESENTE" if os.path.exists(mp4) else "da generare", mp4))
        return

    signal.signal(signal.SIGTERM, _on_stop)
    signal.signal(signal.SIGINT, _on_stop)

    costs = Costs(args.usd_per_hour, len(jobs), max(1, args.cost_every))
    totali = {"ok": 0, "oom": 0, "errore": 0, "saltate": 0}
    for name in ordine:
        e = generate_model(name, [j for j in jobs if j["modello"] == name],
                           args.out_root, costs, args.dry_run)
        for k in totali:
            totali[k] += e.get(k, 0)
        if stop_requested:
            print("\nfermato su richiesta: i modelli rimanenti non sono stati caricati.")
            break

    el = time.time() - costs.t0
    print("\n--- sintesi ---")
    print("clip:   ok %d, oom %d, errore %d, gia' presenti %d"
          % (totali["ok"], totali["oom"], totali["errore"], totali["saltate"]))
    print("tempo:  %.2f h" % (el / 3600))
    if costs.rate:
        print("costo:  $%.2f a $%.2f/h" % (el / 3600 * costs.rate, costs.rate))
    print("uscite: %s" % args.out_root)


if __name__ == "__main__":
    main()
