# night_run.py
#
# Watchdog per il run notturno di generate_test_clips.py (passo 0.2 esteso).
#
# Lancia generate_test_clips.py come processo figlio e ogni INTERVAL_S secondi
# legge GPU (nvidia-smi), RAM, disco e avanzamento. Se una soglia viene superata
# UCCIDE il figlio da solo, senza aspettare nessuno: la decisione di fermarsi non
# deve dipendere da chi legge lo stato. Chi legge (Claude, al mattino, chiunque)
# trova in outputs/night_run/<ts>/:
#   status.json    stato corrente, riscritto a ogni campione in modo atomico
#   watchdog.csv   un campione per riga
#   generate.log   stdout+stderr del figlio
#
# Non importa torch: la GPU si legge da nvidia-smi, cosi' il watchdog non occupa
# VRAM e non entra nel budget del figlio.
#
# Exit code: 0 figlio finito bene, 1 figlio crashato, 2 fermato dal watchdog,
# 3 pre-volo fallito.
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import psutil

INTERVAL_S = 20

# --- soglie di stop (S1-S9 del piano) -----------------------------------------
# Massimo pubblicato da NVIDIA per la 5070 Ti: 88 C. Il run di riferimento ha
# misurato 68 C. 80 lascia 8 C sotto la specifica e 12 sopra il normale.
GPU_TEMP_MAX_C = 80
# Bit di clocks_event_reasons.active che indicano stress termico o elettrico.
# SW_POWER_CAP (0x4) NON e' fra questi: e' il normale tetto di potenza, visto
# anche oggi a 56 C durante SAM, e fermarsi li' vorrebbe dire fermarsi sempre.
THROTTLE_STOP_MASK = 0x8 | 0x20 | 0x40 | 0x80   # HW_SLOWDOWN | SW_THERMAL | HW_THERMAL | HW_POWER_BRAKE
SW_POWER_CAP_BIT = 0x4
FAN_MAX_PCT = 95
POWER_OVER_LIMIT_FACTOR = 1.05
# Campioni consecutivi richiesti per ventola e potenza: un picco isolato di 20 s
# non e' un problema, un minuto si'.
CONSECUTIVE_SAMPLES = 3
# Su Windows non c'e' OOM killer: sotto questa RAM il sistema spilla sul pagefile
# (vram_benchmark.py, HOST_RAM_MIN_GB). Il figlio ha misurato 11 GB liberi al picco.
RAM_MIN_GB = 4.0
DISK_MIN_GB = 20.0
# Clip attesa ~10 min (569 s misurati). La prima include il caricamento del modello.
STALL_MIN = 25
FIRST_STALL_MIN = 35
MAX_HOURS = 5.0
NVSMI_FAIL_MAX = 3

# --- pre-volo del watchdog (il figlio ha il suo, righe 168-183) ---------------
PREFLIGHT_VRAM_USED_MAX_GIB = 1.7   # 15.92 - 14.2 richiesti dal figlio
PREFLIGHT_RAM_MIN_GB = 16.0
PREFLIGHT_DISK_MIN_GB = 50.0
PREFLIGHT_GPU_TEMP_MAX_C = 55       # a riposo: se e' piu' calda, qualcosa gira

KILL_GRACE_S = 10

# 4 classi x 5 prompt in generate_test_clips.PROMPTS. Non si importa quel modulo
# perche' trascina torch e diffusers.
CLIPS_TOTAL = 20
CLIPS_DIR = os.path.join("outputs", "test_clips")
OUT_ROOT = os.path.join("outputs", "night_run")

NVSMI_FIELDS = ["temperature.gpu", "fan.speed", "power.draw", "power.limit",
                "clocks_event_reasons.active", "memory.used", "utilization.gpu",
                "clocks.current.sm"]

CSV_FIELDS = ["ts", "elapsed_s", "gpu_temp_c", "fan_pct", "power_w", "power_limit_w",
              "throttle_mask", "sw_power_cap", "vram_used_mib", "gpu_util_pct",
              "sm_clock_mhz", "cpu_pct", "ram_avail_gb", "disk_free_gb", "clips_done",
              "child_alive"]


def _num(s, cast=float):
    """nvidia-smi scrive '[N/A]' per i sensori assenti: diventa None, non errore."""
    try:
        return cast(s)
    except (TypeError, ValueError):
        return None


def gpu_query():
    """Un campione da nvidia-smi. None se nvidia-smi non risponde."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=" + ",".join(NVSMI_FIELDS),
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=10).strip().splitlines()[0]
    except Exception:
        return None
    p = [v.strip() for v in out.split(",")]
    if len(p) != len(NVSMI_FIELDS):
        return None
    try:
        mask = int(p[4], 16)
    except ValueError:
        mask = None
    return {"gpu_temp_c": _num(p[0], int), "fan_pct": _num(p[1], int),
            "power_w": _num(p[2]), "power_limit_w": _num(p[3]), "throttle_mask": mask,
            "vram_used_mib": _num(p[5], int), "gpu_util_pct": _num(p[6], int),
            "sm_clock_mhz": _num(p[7], int)}


def gpu_compute_apps():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True, timeout=10).strip()
    except Exception:
        return []
    return [l.strip() for l in out.splitlines() if l.strip()]


def clips_done():
    return len([p for p in glob.glob(os.path.join(CLIPS_DIR, "*.mp4"))
                if not p.endswith(".tmp.mp4")])


def manifest_mtime():
    path = os.path.join(CLIPS_DIR, "manifest.jsonl")
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


def write_status(path, payload):
    """tmp + os.replace: chi legge non trova mai un JSON a meta'."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def kill_tree(proc):
    """Figlio e discendenti: terminate, KILL_GRACE_S di attesa, poi kill."""
    try:
        parent = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        return
    procs = parent.children(recursive=True) + [parent]
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=KILL_GRACE_S)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


def preflight():
    problems = []
    # Su Windows query-compute-apps elenca OGNI processo con un contesto sulla
    # scheda (explorer, browser, VS Code, sfondi animati...), non solo CUDA.
    # Il pericolo vero e' un altro python (Wan, SAM, DA3) gia' in corso: si
    # filtra su quello; il resto pesa solo sulla VRAM occupata, controllata sotto.
    apps = [a for a in gpu_compute_apps() if "python" in a.lower()]
    if apps:
        problems.append("altri python sulla GPU: %s" % "; ".join(apps))
    g = gpu_query()
    if g is None:
        problems.append("nvidia-smi non risponde")
    else:
        if g["vram_used_mib"] is not None and g["vram_used_mib"] / 1024 > PREFLIGHT_VRAM_USED_MAX_GIB:
            problems.append("VRAM occupata %.2f GiB > %.1f: chiudere browser e app con"
                            " accelerazione GPU" % (g["vram_used_mib"] / 1024, PREFLIGHT_VRAM_USED_MAX_GIB))
        if g["gpu_temp_c"] is not None and g["gpu_temp_c"] > PREFLIGHT_GPU_TEMP_MAX_C:
            problems.append("GPU a %d C a riposo (> %d): qualcosa sta girando"
                            % (g["gpu_temp_c"], PREFLIGHT_GPU_TEMP_MAX_C))
    ram = psutil.virtual_memory().available / 1024**3
    if ram < PREFLIGHT_RAM_MIN_GB:
        problems.append("RAM disponibile %.1f GB < %.0f" % (ram, PREFLIGHT_RAM_MIN_GB))
    disk = psutil.disk_usage(os.path.abspath(".")).free / 1024**3
    if disk < PREFLIGHT_DISK_MIN_GB:
        problems.append("disco libero %.0f GB < %.0f" % (disk, PREFLIGHT_DISK_MIN_GB))
    print("pre-volo: VRAM occupata %s MiB, GPU %s C, RAM libera %.1f GB, disco %.0f GB, %d"
          " processi GPU" % (g and g["vram_used_mib"], g and g["gpu_temp_c"], ram, disk,
                             len(apps)), flush=True)
    if problems:
        sys.exit("pre-volo fallito:\n  " + "\n  ".join(problems))


def main():
    parser = argparse.ArgumentParser(
        description="Watchdog notturno per generate_test_clips.py: ferma da solo, registra tutto.")
    parser.add_argument("--interval", type=int, default=INTERVAL_S)
    parser.add_argument("--gpu-temp-max", type=int, default=GPU_TEMP_MAX_C)
    parser.add_argument("--ram-min-gb", type=float, default=RAM_MIN_GB)
    parser.add_argument("--disk-min-gb", type=float, default=DISK_MIN_GB)
    parser.add_argument("--stall-min", type=float, default=STALL_MIN)
    parser.add_argument("--first-stall-min", type=float, default=FIRST_STALL_MIN)
    parser.add_argument("--max-hours", type=float, default=MAX_HOURS)
    parser.add_argument("--limit", type=int, default=None,
                        help="inoltrato a generate_test_clips.py")
    parser.add_argument("--rehearsal", action="store_true",
                        help="figlio fittizio (sleep 150 s) al posto di Wan: prova del meccanismo")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, ts)
    os.makedirs(out_dir, exist_ok=True)
    status_path = os.path.join(out_dir, "status.json")
    csv_path = os.path.join(out_dir, "watchdog.csv")
    log_path = os.path.join(out_dir, "generate.log")

    preflight()

    if args.rehearsal:
        cmd = [sys.executable, "-u", "-c", "import time; time.sleep(150)"]
    else:
        cmd = [sys.executable, "-u", "generate_test_clips.py"]
        if args.limit:
            cmd += ["--limit", str(args.limit)]

    started = time.time()
    started_iso = datetime.now().isoformat(timespec="seconds")
    clips_at_start = clips_done()
    last_progress = started
    last_clips, last_manifest = clips_at_start, manifest_mtime()
    fan_hits = power_hits = nvsmi_fails = 0
    peak_temp, peak_power, sw_cap_samples, n_samples = 0, 0.0, 0, 0
    state, reason, returncode = "running", "", None
    last_sample = None

    log = open(log_path, "w", encoding="utf-8")
    child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    print("figlio pid %d: %s" % (child.pid, " ".join(cmd)), flush=True)
    print("stato: %s" % status_path, flush=True)

    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()

    def status():
        # legge state/reason/returncode/last_sample dallo scope di main al momento
        # della chiamata: cosi' anche il finally scrive l'ultimo campione vero
        write_status(status_path, {
            "state": state, "reason": reason, "ts": datetime.now().isoformat(timespec="seconds"),
            "started": started_iso, "elapsed_s": round(time.time() - started),
            "clips_done": clips_done(), "clips_total": CLIPS_TOTAL, "rehearsal": args.rehearsal,
            "last": last_sample, "child_pid": child.pid, "child_returncode": returncode,
            "log": os.path.abspath(log_path), "csv": os.path.abspath(csv_path)})

    status()
    try:
        while True:
            time.sleep(args.interval)
            now = time.time()
            n_samples += 1
            g = gpu_query()
            ram = psutil.virtual_memory().available / 1024**3
            disk = psutil.disk_usage(os.path.abspath(CLIPS_DIR if os.path.isdir(CLIPS_DIR) else ".")).free / 1024**3
            cpu = psutil.cpu_percent(interval=None)
            done = clips_done()
            mm = manifest_mtime()
            alive = child.poll() is None

            if done != last_clips or mm != last_manifest:
                last_progress, last_clips, last_manifest = now, done, mm

            sample = {"ts": datetime.now().isoformat(timespec="seconds"),
                      "elapsed_s": round(now - started), "cpu_pct": cpu,
                      "ram_avail_gb": round(ram, 2), "disk_free_gb": round(disk, 1),
                      "clips_done": done, "child_alive": alive}
            if g is not None:
                nvsmi_fails = 0
                sample.update(g)
                sample["sw_power_cap"] = bool(g["throttle_mask"] and g["throttle_mask"] & SW_POWER_CAP_BIT)
                sw_cap_samples += int(sample["sw_power_cap"])
                if g["gpu_temp_c"] is not None:
                    peak_temp = max(peak_temp, g["gpu_temp_c"])
                if g["power_w"] is not None:
                    peak_power = max(peak_power, g["power_w"])
            else:
                nvsmi_fails += 1
            writer.writerow({k: sample.get(k, "") for k in CSV_FIELDS})
            csv_file.flush()
            last_sample = sample

            # --- il figlio ha finito da solo ---
            if not alive:
                returncode = child.returncode
                state = "finished" if returncode == 0 else "crashed"
                reason = "" if returncode == 0 else "exit code %s, vedi generate.log" % returncode
                status()
                break

            # --- S1-S9 ---
            trip = None
            if g is None:
                if nvsmi_fails >= NVSMI_FAIL_MAX:
                    trip = "S8 nvidia-smi non risponde da %d campioni" % nvsmi_fails
            else:
                t, fan, pw, lim, mask = (g["gpu_temp_c"], g["fan_pct"], g["power_w"],
                                         g["power_limit_w"], g["throttle_mask"])
                if t is not None and t >= args.gpu_temp_max:
                    trip = "S1 GPU %d C >= %d" % (t, args.gpu_temp_max)
                elif mask is not None and mask & THROTTLE_STOP_MASK:
                    trip = "S2 throttling termico/elettrico, mask 0x%x" % mask
                fan_hits = fan_hits + 1 if (fan is not None and fan >= FAN_MAX_PCT) else 0
                power_hits = power_hits + 1 if (pw is not None and lim and pw > lim * POWER_OVER_LIMIT_FACTOR) else 0
                if trip is None and fan_hits >= CONSECUTIVE_SAMPLES:
                    trip = "S3 ventola %d%% per %d campioni" % (fan, fan_hits)
                if trip is None and power_hits >= CONSECUTIVE_SAMPLES:
                    trip = "S4 potenza %.0f W > limite %.0f W per %d campioni" % (pw, lim, power_hits)
            if trip is None and ram < args.ram_min_gb:
                trip = "S5 RAM disponibile %.1f GB < %.1f" % (ram, args.ram_min_gb)
            if trip is None and disk < args.disk_min_gb:
                trip = "S6 disco libero %.0f GB < %.0f" % (disk, args.disk_min_gb)
            stall_limit = args.first_stall_min if done == clips_at_start else args.stall_min
            if trip is None and (now - last_progress) / 60 > stall_limit:
                trip = "S7 nessun progresso da %.0f min (clip %d)" % ((now - last_progress) / 60, done)
            if trip is None and (now - started) / 3600 > args.max_hours:
                trip = "S9 durata %.1f h > %.1f" % ((now - started) / 3600, args.max_hours)

            if trip:
                print("STOP: %s" % trip, file=sys.stderr, flush=True)
                state, reason = "stopped", trip
                status()
                kill_tree(child)
                returncode = child.poll()
                status()
                break

            status()
            if n_samples % 15 == 0:   # ogni 5 min a 20 s
                print("  %s  GPU %s C  %s W  fan %s%%  VRAM %s MiB  RAM %.1f GB  clip %d/%d"
                      % (sample["ts"][11:19], sample.get("gpu_temp_c"), sample.get("power_w"),
                         sample.get("fan_pct"), sample.get("vram_used_mib"), ram, done,
                         CLIPS_TOTAL), flush=True)
    except KeyboardInterrupt:
        state, reason = "stopped", "interrotto da tastiera"
        status()
    finally:
        # R9 di carla_capture.py: il figlio non sopravvive al watchdog, mai.
        if child.poll() is None:
            kill_tree(child)
        returncode = child.poll()
        status()
        csv_file.close()
        log.close()

    print("\n--- sintesi ---")
    print("stato:             %s%s" % (state, (" (%s)" % reason) if reason else ""))
    print("durata:            %.1f min" % ((time.time() - started) / 60))
    print("clip:              %d/%d (%d fatte in questo run)"
          % (clips_done(), CLIPS_TOTAL, clips_done() - clips_at_start))
    print("picco GPU:         %d C, %.0f W" % (peak_temp, peak_power))
    print("campioni:          %d, con SW_POWER_CAP %d" % (n_samples, sw_cap_samples))
    print("stato/csv/log:     %s" % out_dir)
    sys.exit({"finished": 0, "crashed": 1, "stopped": 2}[state])


if __name__ == "__main__":
    main()
