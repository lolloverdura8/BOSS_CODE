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
