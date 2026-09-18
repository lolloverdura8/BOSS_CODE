# gpu.py
#
# Telemetria nvidia-smi e guardie di memoria, condivise da annotate_sam.py e
# annotate_da3.py. Prima erano copiate dentro ognuno dei due, con le soglie
# cablate sui 16 GB della 5070 Ti: su una scheda diversa fermavano lo script
# sbagliando, o non lo fermavano quando serviva.
#
# Due cose cambiano rispetto alle copie originali:
#
# 1. Il campo di nvidia-smi. "clocks_throttle_reasons.active" e' un alias
#    deprecato in favore di "clocks_event_reasons.active". Verificato il
#    15/09/2026 su questa workstation (driver 610.88, RTX 5070 Ti): il nome
#    vecchio risponde ANCORA, quindi qui non era rotto. Ma se su un pod con un
#    driver diverso non rispondesse, gpu_telemetry() cadrebbe nell'except e
#    restituirebbe None per tutto senza dire niente: i CSV avrebbero le colonne
#    di throttling vuote e nessuno saprebbe perche'. Qui si prova prima il nome
#    moderno, si ripiega sul vecchio, e si stampa quale dei due ha risposto -
#    cosi' il caso "nessuno dei due" diventa una riga di log invece che silenzio.
# 2. Le soglie sono argomenti, non costanti, e i default si ricavano dalla
#    scheda che c'e' davvero.
#
# Dipende solo dalla standard library: torch e' importato dentro le funzioni che
# lo usano, cosi' il modulo resta importabile anche dove torch non c'e'.
import subprocess

# Nome moderno per primo. Il vecchio resta come ripiego per i driver piu' datati.
_THROTTLE_FIELDS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")

# Quale dei due ha risposto: risolto al primo uso e poi riusato.
_field_in_use = None
_field_reported = False

EMPTY = {"gpu_temp_c": None, "throttle_mask": None, "sm_clock_mhz": None}

# Frazione della VRAM totale oltre la quale il run si ferma. 0,80 su 15,92 GiB
# da 12,7 GB, cioe' il valore che i due script usavano cablato (13,0) sulla
# 5070 Ti: la formula riproduce la soglia storica e la trasporta sulle altre schede.
VRAM_BUDGET_FRACTION = 0.80

# VRAM libera minima prima di caricare. E' una soglia assoluta e non una frazione:
# risponde a "c'e' qualcun altro sulla scheda?", non a "quanto e' grande". SAM ha
# un picco misurato di 4,35 GB e DA3 di 1,49 GB (B.1-B.2), 6 GB lasciano margine.
VRAM_FREE_MIN_GB = 6.0

TEMP_LIMIT_C = 83


def _query(field):
    out = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=temperature.gpu,%s,clocks.current.sm" % field,
         "--format=csv,noheader,nounits"],
        stderr=subprocess.DEVNULL, text=True, timeout=10)
    temp, throttle, sm = [p.strip() for p in out.strip().splitlines()[0].split(",")]
    return {"gpu_temp_c": int(temp), "throttle_mask": throttle, "sm_clock_mhz": int(sm)}


def gpu_telemetry():
    """Temperatura, maschera di throttling e clock SM. EMPTY se nvidia-smi non risponde."""
    global _field_in_use, _field_reported

    candidates = (_field_in_use,) if _field_in_use else _THROTTLE_FIELDS
    for field in candidates:
        try:
            tel = _query(field)
        except Exception:
            continue
        _field_in_use = field
        if not _field_reported:
            print("telemetria GPU: nvidia-smi risponde a %s" % field)
            _field_reported = True
        return tel

    if not _field_reported:
        # Non e' un dettaglio da lasciare implicito: senza questa riga le colonne
        # di throttling dei CSV sarebbero vuote e nessuno saprebbe perche'.
        print("telemetria GPU non disponibile: nvidia-smi non risponde a nessuno di %s."
              " Le colonne di throttling dei CSV resteranno vuote." % ", ".join(_THROTTLE_FIELDS))
        _field_reported = True
    return dict(EMPTY)


def is_throttling(tel, temp_limit_c=TEMP_LIMIT_C):
    if tel["throttle_mask"] is None:
        return False
    try:
        mask = int(tel["throttle_mask"], 16)
    except ValueError:
        return False
    hot = tel["gpu_temp_c"] is not None and tel["gpu_temp_c"] >= temp_limit_c
    return mask != 0 or hot


def vram_budget_gb(fraction=VRAM_BUDGET_FRACTION):
    """Soglia di stop ricavata dalla scheda presente, in GB decimali come il resto."""
    import torch
    return torch.cuda.mem_get_info()[1] / 1e9 * fraction


def check_vram_free(min_gb=VRAM_FREE_MIN_GB):
    """Stampa la VRAM libera e solleva se qualcun altro occupa ancora la scheda."""
    import torch
    free, total = torch.cuda.mem_get_info()
    print("VRAM libera prima di caricare: %.2f GB su %.2f" % (free / 1e9, total / 1e9))
    if free / 1e9 < min_gb:
        raise SystemExit("VRAM libera %.2f GB < %.1f GB richiesti: la scheda e' ancora"
                         " occupata da un altro processo." % (free / 1e9, min_gb))
    return free / 1e9, total / 1e9


# --------------------------------------------------------------------------
# RAM HOST. Dentro un container psutil legge /proc/meminfo, che e' quello della
# MACCHINA e non del container. Misurato il 16/09/2026 sul primo pod RunPod
# (RTX PRO 4000, EU-RO-1): "free -g" dichiarava 125 GiB totali e "nproc" 48
# core, mentre il cgroup concedeva 30.999.998.464 byte e 10,2 core. Un processo
# che avesse creduto ai 125 GiB sarebbe stato ucciso dal kernel al superamento
# del cgroup: nessuna eccezione Python, nessuna riga di log, solo il processo
# sparito a meta' clip. E' la stessa classe di guasto silenzioso di output_type.
#
# Quindi: si legge il cgroup, e si ripiega su psutil dove non c'e' (Windows, e
# Linux fuori da un container).
#
# Unita': GiB, come gli altri campi host del manifest. La VRAM qui sopra e' in
# GB decimali: sono due convenzioni diverse, ma vengono dai valori gia' misurati
# il 14/09 e cambiarle romperebbe il confronto con quelli.

_CGROUPS = (
    ("cgroup v2", "/sys/fs/cgroup/memory.max",
                  "/sys/fs/cgroup/memory.current",
                  "/sys/fs/cgroup/memory.stat"),
    ("cgroup v1", "/sys/fs/cgroup/memory/memory.limit_in_bytes",
                  "/sys/fs/cgroup/memory/memory.usage_in_bytes",
                  "/sys/fs/cgroup/memory/memory.stat"),
)

# Senza tetto, v1 scrive 9223372036854771712 e v2 la parola "max": oltre questa
# soglia "limite" vuol dire "nessun limite", e la risposta giusta e' psutil.
_NO_LIMIT = 1 << 62

_ram_source_reported = False


def _read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _read_stat(path):
    out = {}
    try:
        with open(path) as f:
            for line in f:
                chiave, _, valore = line.partition(" ")
                try:
                    out[chiave] = int(valore)
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def _cgroup_ram():
    for nome, maxp, curp, statp in _CGROUPS:
        limite = _read_int(maxp)
        if limite is None or limite >= _NO_LIMIT:
            continue
        stat = _read_stat(statp)
        # La page cache inattiva risulta occupata ma il kernel la cede sotto
        # pressione. Contarla come spesa farebbe scattare la guardia subito dopo
        # il download dei checkpoint, quando 240 GB di letture hanno riempito la
        # cache fino al tetto: si rifiuterebbe di caricare avendo la RAM libera.
        cedibile = stat.get("inactive_file", 0) + stat.get("slab_reclaimable", 0)
        usata = max(0, (_read_int(curp) or 0) - cedibile)
        return limite / 1024**3, (limite - usata) / 1024**3, nome
    return None


def host_ram():
    """(totale, disponibile, sorgente) in GiB, del container quando ce n'e' uno.

    La sorgente e' parte della risposta, non un dettaglio: un numero di RAM
    senza provenienza e' esattamente cio' che ha reso invisibile questo bug.
    """
    global _ram_source_reported
    misura = _cgroup_ram()
    if misura is None:
        import psutil
        m = psutil.virtual_memory()
        misura = (m.total / 1024**3, m.available / 1024**3, "psutil")
    if not _ram_source_reported:
        print("RAM host: tetto %.1f GiB, disponibili %.1f GiB, letti da %s"
              % (misura[0], misura[1], misura[2]))
        _ram_source_reported = True
    return misura


# --------------------------------------------------------------------------
# NOME DELLA SCHEDA. Il bake-off confronta $/istanza_utile, e quel numero
# dipende dai secondi per clip: una proprieta' della GPU, non del modello.
# Il 17-18/09/2026 le misure sono arrivate da schede diverse in sessioni
# diverse (5070 Ti in locale, RTX PRO 4000 e poi A100 sul pod) e il manifest
# non registrava su quale: senza questo campo, due clip dello stesso modello
# generate su schede diverse finiscono nella stessa tabella senza che nulla
# lo segnali, e il confronto e' silenziosamente falsato.
_gpu_name_cache = None


def gpu_name():
    """Il modello della GPU corrente, cosi' com'e' riportato da nvidia-smi.

    Interrogata una volta sola per processo: non cambia durante un run.
    "sconosciuta" (non None) se nvidia-smi non risponde, per restare un
    valore stringa scrivibile in CSV/JSON senza casi speciali a valle.
    """
    global _gpu_name_cache
    if _gpu_name_cache is None:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL, text=True, timeout=10)
            _gpu_name_cache = out.strip().splitlines()[0].strip()
        except Exception:
            _gpu_name_cache = "sconosciuta"
    return _gpu_name_cache
