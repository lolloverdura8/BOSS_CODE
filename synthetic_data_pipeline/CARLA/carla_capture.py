# carla_capture.py
#
# Fase A del piano dati sintetici: registra una sessione CARLA con camera indossata
# da un pedone e quattro sensori allineati, producendo la ground truth densa che
# la stereocamera non puo' dare (depth esatta per ogni pixel, segmentazione esatta,
# posa esatta, tutte per costruzione e non per misura).
#
# Lo script ACQUISISCE E BASTA (par. 14.4). Non mappa le classi CARLA su quelle
# BOSS, non calcola box 2D, non calcola distanze per oggetto, non produce video di
# visualizzazione, non scrive il manifest del dataset: tutto questo e' compito di
# carla_export.py, e tenerli separati e' cio' che consente di rivedere i criteri di
# esportazione senza ri-registrare la sessione.
#
# I requisiti R1-R10 dell'appendice sono vincolanti e ognuno corrisponde a un
# errore che NON produce eccezioni ma rende i dati inutilizzabili, spesso scoperto
# solo dopo aver registrato tutto. Sono annotati nel codice punto per punto.
#
# Prerequisito: server CARLA 0.9.16 gia' avviato, tipicamente con
#   CarlaUE4.exe -RenderOffScreen -quality-level=Low
# La qualita' bassa e' legittima (par. 5.0): nel ramo Sim2Real l'aspetto viene
# rigenerato dal modello video, a CARLA si chiede geometria ed etichette esatte.
#
# Dal 05/10 il lavoro e' diviso in tre moduli: qui la sessione (mondo, pedone,
# sensori, scrittura), in scene_actors.py la costruzione degli ostacoli e del mezzo
# in arrivo, in acceptance.py i controlli A1-A9, che non importano carla e si
# rieseguono su una sessione gia' registrata.
import argparse
import hashlib
import json
import math
import os
import queue
import random
import subprocess
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

import carla

from acceptance import print_acceptance, run_acceptance_checks, write_acceptance
from boss_classes import BOSS_CLASS_BY_NAME, CARLA_TAG_NAMES
from scenarios import SCENARIOS
from scene_actors import (ApproachingRider, place_obstacle, resolve_blueprint, spec_label,
                          transform_to_dict, tree_trunks, validate_meshes,
                          walker_origin_above_feet)

HOST, PORT = "localhost", 2000
CLIENT_TIMEOUT_S = 20.0

# Risoluzione allineata agli ANNOTATORI, non ai generatori. La cattura e' la
# sorgente di ground truth, e la ground truth viene confrontata con l'output di
# SAM 3.1 e Depth Anything 3: allinearsi alla risoluzione nativa di un generatore
# video sarebbe una scommessa su una decisione non ancora presa.
#   - 1008 e' la risoluzione nativa esatta di SAM 3.1 (img_size=1008 in
#     sam3/model_builder.py, ricorrente in dieci punti del sorgente);
#   - 1008 e 756 sono entrambi divisibili per 14, il patch_size del backbone
#     DINOv2 condiviso da SAM 3.1 e DA3. Con dimensioni non multiple di 14 i due
#     modelli applicano internamente un resize, che introduce uno slittamento
#     sub-pixel fra l'RGB che il modello vede e la depth GT di CARLA: e' un errore
#     che non appartiene al modello e contaminerebbe AbsRel, RMSE e delta<1,25;
#   - 4:3 e' l'aspect delle recordings reali e della stereocamera di progetto.
# Si puo' sempre sottocampionare verso il generatore a valle; il contrario no.
WIDTH, HEIGHT = 1008, 756

# FOV orizzontale dell'ottica di progetto (Piano di Sviluppo, RGB a 82 gradi).
FOV = 82.0

# Camera all'altezza di torace/testa, non del tetto di un'auto: e' il punto che
# giustifica CARLA rispetto a un dataset pubblico. x leggermente in avanti perche'
# a x=0 la camera sarebbe dentro la mesh del pedone.
CAMERA_X, CAMERA_Z = 0.25, 1.50

# Passo fisso a 16 Hz. E' divisibile per i target a valle plausibili: 16 fps a
# frame pieno, 8 fps prendendo un frame su due. Con i 20 Hz di default nessuno dei
# due sottocampionamenti sarebbe intero.
FIXED_DELTA_SECONDS = 0.0625

# I primi tick dopo lo spawn hanno LOD e meteo non assestati e il pedone e' ancora
# fermo: si scartano invece di registrarli.
WARMUP_TICKS = 30

# Quanti tick di assestamento fare dopo lo spawn del controller AI prima che
# go_to_location abbia effetto. Misurato: senza questi tick il comando viene
# ignorato in silenzio e il pedone resta fermo al punto di spawn.
CONTROLLER_SETTLE_TICKS = 2

# Gli ostacoli dello scenario NON si piazzano piu' con una proiezione statica dalla
# retta spawn->target: due sessioni reali hanno mostrato che il controller AI segue
# la nav mesh (il marciapiede), che curva entro pochi metri, e una singola stima di
# direzione - anche presa camminando davvero, anche a 24 tick - resta valida solo
# vicino al punto in cui e' stata misurata (misurato: A7 falliva ancora con 0/300
# frame, con il pedone che a meta' cammino si trovava a **1,5-2 m** dal prop ma
# fuori dal suo cono di ripresa, perche' nel frattempo aveva girato).
#
# Si piazza invece OGNI prop appena il pedone raggiunge, camminando (non in linea
# d'aria), la distanza "forward" configurata: SPAWN_LEAD_M e' quanto in anticipo,
# usando la direzione ATTUALE del pedone in quel preciso istante (non quella di
# 9-15 m prima). L'errore di estrapolazione si riduce cosi' da "l'intera distanza
# del prop" a "questi pochi metri", indipendentemente da quanto curva il percorso.
#
# Il valore non e' libero: con la camera a CAMERA_Z=1,5 m (circa 1,66 m nel mondo,
# vedi origin_above_feet) e mira in piano (pitch 0), un ostacolo a terra esce dal
# cono VERTICALE (mezza apertura 33,1 gradi) sotto circa 1,5/tan(33,1) = 2,3 m di
# distanza, qualunque sia il piazzamento orizzontale. Misurato: con
# SPAWN_LEAD_M=3,0 la finestra utile (dal piazzamento a quel limite) e' di soli
# ~8 frame, sotto la soglia A7_MIN_FRAMES=10. 5,0 m allunga la finestra utile a
# ~30 frame stimati, con un rischio di curvatura ancora contenuto.
SPAWN_LEAD_M = 5.0

WALKER_SPEED = 1.4          # m/s, passo di un pedone
NAV_POOL_SIZE = 64          # locazioni estratte dalla nav mesh, indicizzate dallo scenario
QUEUE_TIMEOUT_S = 10.0      # attesa massima di un frame da un sensore
FRAME_SYNC_RETRIES = 10     # quanti frame arretrati si scartano prima di dichiarare R2 rotto

# La depth viene salvata in millimetri su 16 bit: risoluzione 1 mm fino a 65,535 m,
# ampiamente sufficiente per ostacoli urbani. Il fattore di scala e' dichiarato in
# session.json perche' senza di esso il PNG non e' interpretabile in metri.
DEPTH_SCALE = 1000.0
DEPTH_MAX_U16 = 65535

PROGRESS_EVERY = 25

# Durata della finestra di posizioni recenti con cui si stima la direzione di
# marcia (vedi recent_locs in capture()): 8 tick a 16 Hz. Espressa in secondi
# perche' gli scenari Sim2Real girano a 48 Hz, dove 8 tick sarebbero 0,17 s.
DIRECTION_WINDOW_S = 0.5

# Sotto questo spostamento nella finestra di direzione la stima e' rumore di posa,
# non moto: si tiene la stima precedente invece di aggiornarla.
HEADING_MIN_MOTION_M = 0.05

# Chiavi ammesse negli scenari e nelle loro dichiarazioni. Si controllano tutte,
# anche le facoltative: un refuso in "fps" o in "min_window_s" non solleverebbe
# nulla, ma registrerebbe a 1008x756 e 16 Hz o spegnerebbe A8 (revisione 05/10).
SCENARIO_KEYS = {"map", "weather", "wind_intensity", "n_frames", "seed", "walker_bp_index",
                 "walker_spawn_index", "walker_target_index", "traffic_vehicles",
                 "traffic_walkers", "width", "height", "fps", "spawn_lead_m", "min_window_s",
                 "place_at_start", "actors", "movers"}
ACTOR_KEYS = {"blueprint", "mesh", "scale", "forward", "lateral", "z", "yaw", "pitch", "roll",
              "boss_class", "physics", "anchor", "support"}
ACTOR_REQUIRED = ("forward", "lateral", "z", "yaw", "boss_class", "physics")
SUPPORT_KEYS = {"blueprint", "mesh", "scale", "forward", "lateral", "z", "yaw"}
SUPPORT_REQUIRED = ("forward", "lateral", "z")
MOVER_KEYS = {"kind", "group", "boss_class", "rider", "forward", "lateral", "speed_kmh",
              "pass_lateral", "swerve_from_m", "swerve_to_m", "deck_top_m", "parts"}
PART_KEYS = {"part", "mesh", "scale", "dx", "dy", "z", "pitch", "yaw", "roll"}
PART_REQUIRED = ("part", "mesh", "dx", "dy", "z")
# La geometria Sim2Real si dichiara tutta insieme o per niente.
S2R_KEYS = ("width", "height", "fps", "min_window_s")


# --- utilita' di basso livello ----------------------------------------------

def to_bgra(image):
    """Vista (H, W, 4) sul buffer grezzo del sensore, in ordine BGRA."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    return arr.reshape((image.height, image.width, 4))


def decode_depth_mm(image):
    """Depth in millimetri su uint16, decodificata dal raw a 24 bit (R3 + R4).

    Non si usa MAI carla.ColorConverter.Depth: produce una visualizzazione
    logaritmica plausibile all'occhio e completamente falsa nei valori. Attenzione
    all'ordine dei canali: il buffer e' BGRA, quindi il rosso e' l'indice 2.
    """
    bgra = to_bgra(image)
    b = bgra[:, :, 0].astype(np.float64)
    g = bgra[:, :, 1].astype(np.float64)
    r = bgra[:, :, 2].astype(np.float64)
    normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 ** 3 - 1.0)
    meters = 1000.0 * normalized
    # Il clip a 65535 mm satura tutto cio' che sta oltre 65,5 m, cielo compreso:
    # e' esattamente cio' che il controllo A4 va a verificare.
    return np.clip(meters * DEPTH_SCALE, 0.0, DEPTH_MAX_U16).astype(np.uint16)


def compute_intrinsics(width, height, fov_deg):
    """Matrice K della camera pinhole di CARLA (pixel quadrati, principale al centro)."""
    f = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return {"fx": f, "fy": f, "cx": width / 2.0, "cy": height / 2.0}


def vertical_fov_deg(width, height, fov_deg):
    """FOV verticale: CARLA fissa l'orizzontale, il verticale segue l'aspect."""
    half_h = math.tan(math.radians(fov_deg) / 2.0)
    return math.degrees(2.0 * math.atan(half_h * height / float(width)))


def capture_geometry(scenario):
    """Risoluzione e passo di una sessione, letti dallo scenario.

    Gli scenari del reference set non dichiarano nulla e restano a 1008x756 e
    16 Hz: e' cio' che tiene riproducibile eval_set.json. Gli scenari Sim2Real
    dichiarano width, height e fps nello scenario stesso, e non da riga di
    comando: cosi' finiscono testualmente in session.json, e una sessione non si
    puo' registrare alla geometria sbagliata dimenticando un flag.

    I tick di warmup e la finestra di direzione sono durate, non conteggi: a
    48 Hz gli stessi 30 tick durerebbero un terzo, e LOD e meteo non farebbero in
    tempo ad assestarsi. Al passo di default i conteggi restano quelli di prima.
    """
    fps = scenario.get("fps")
    fixed_delta = 1.0 / fps if fps else FIXED_DELTA_SECONDS
    return {
        "width": scenario.get("width", WIDTH),
        "height": scenario.get("height", HEIGHT),
        "fixed_delta": fixed_delta,
        "warmup_ticks": round(WARMUP_TICKS * FIXED_DELTA_SECONDS / fixed_delta),
        "direction_window": round(DIRECTION_WINDOW_S / fixed_delta) + 1,
        "spawn_lead_m": scenario.get("spawn_lead_m", SPAWN_LEAD_M),
    }


def check_versions(client):
    """R1: client e server devono essere della stessa identica versione.

    Il disallineamento e' il primo errore che si incontra e da' messaggi poco
    chiari: API che rispondono in modo incoerente invece di un errore netto.
    """
    cv_, sv = client.get_client_version(), client.get_server_version()
    if cv_ != sv:
        sys.exit("R1 FALLITO: client %s != server %s. Installare il pacchetto"
                 " python della stessa versione del server." % (cv_, sv))
    print("CARLA %s (client e server allineati)" % cv_)
    return cv_


def _check_keys(what, spec, allowed, required=()):
    unknown = sorted(set(spec) - allowed)
    if unknown:
        sys.exit("chiavi sconosciute in %s: %s (ammesse: %s)" % (what, unknown, sorted(allowed)))
    missing = [k for k in required if k not in spec]
    if missing:
        sys.exit("chiavi mancanti in %s: %s" % (what, missing))


def _check_class(name):
    if name not in BOSS_CLASS_BY_NAME:
        sys.exit("classe BOSS sconosciuta nello scenario: %s" % name)


def _check_source(what, spec, library):
    """Un attore viene o da un blueprint del catalogo o da una mesh: uno dei due, non entrambi."""
    if ("blueprint" in spec) == ("mesh" in spec):
        sys.exit("%s deve dichiarare esattamente uno fra blueprint e mesh: %s" % (what, spec))
    if "blueprint" in spec:
        resolve_blueprint(library, spec["blueprint"])


def validate_scenario(scenario, library):
    """Verifica chiavi, blueprint e classi BOSS dello scenario prima di toccare il mondo.

    Le mesh si verificano dopo, con validate_meshes(), perche' per farlo servono
    degli spawn di prova.
    """
    _check_keys("scenario", scenario, SCENARIO_KEYS)
    declared = [k for k in S2R_KEYS if k in scenario]
    if declared and len(declared) != len(S2R_KEYS):
        sys.exit("scenario Sim2Real incompleto: dichiara %s ma non %s"
                 % (declared, [k for k in S2R_KEYS if k not in scenario]))
    for spec in scenario["actors"]:
        _check_keys("ostacolo", spec, ACTOR_KEYS, ACTOR_REQUIRED)
        _check_source("ostacolo", spec, library)
        _check_class(spec["boss_class"])
        if spec.get("anchor") not in (None, "tree"):
            sys.exit("ancoraggio sconosciuto: %s (ammesso: tree)" % spec["anchor"])
        if "support" in spec:
            _check_keys("sostegno", spec["support"], SUPPORT_KEYS, SUPPORT_REQUIRED)
            _check_source("sostegno", spec["support"], library)
    movers = scenario.get("movers", [])
    if movers and not scenario.get("place_at_start"):
        sys.exit("i mezzi in movimento si piazzano solo con place_at_start")
    for mover in movers:
        _check_keys("mezzo", mover, MOVER_KEYS, tuple(sorted(MOVER_KEYS - {"group"})))
        if mover["kind"] != "approaching_rider":
            sys.exit("tipo di mezzo sconosciuto: %s" % mover["kind"])
        if mover["swerve_from_m"] <= mover["swerve_to_m"]:
            sys.exit("swerve_from_m deve superare swerve_to_m: %s" % mover)
        resolve_blueprint(library, mover["rider"])
        _check_class(mover["boss_class"])
        for part in mover["parts"]:
            _check_keys("pezzo del mezzo", part, PART_KEYS, PART_REQUIRED)


def code_version():
    """Il commit del codice che registra (git describe), o None fuori da un checkout.

    Senza, una sessione non dice con quale versione di questo script e' nata: le
    sessioni del set di valutazione hanno spawn_lead_m 3,0 mentre il codice ha 5,0,
    e nulla nel file lo spiegava (revisione 05/10).
    """
    try:
        done = subprocess.run(["git", "describe", "--always", "--dirty"],
                              cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None


# --- costruzione della scena -------------------------------------------------

def build_nav_pool(world, size):
    """Pool deterministico di locazioni navigabili, indicizzato dallo scenario.

    Si usa la nav mesh e non map.get_spawn_points(): quelli sono punti stradali,
    pensati per i veicoli, e metterebbero il pedone in mezzo alla carreggiata.
    Il determinismo viene da world.set_pedestrians_seed(), chiamato prima.
    """
    pool = []
    for _ in range(size):
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            pool.append(loc)
    if len(pool) < 2:
        sys.exit("nav mesh non disponibile su questa mappa: nessun punto pedonale estratto")
    return pool


def path_frame(spawn_loc, target_loc):
    """Terna (avanti, destra, yaw) della direzione di marcia iniziale.

    CARLA usa un sistema levogiro con X avanti, Y a destra e yaw crescente verso
    +Y: ruotando la direzione di marcia di +90 gradi si ottiene quindi la destra,
    cioe' (-dy, dx). Sbagliare questo segno specchia lateralmente tutti gli
    ostacoli dello scenario senza che nulla segnali l'errore.
    """
    dx = target_loc.x - spawn_loc.x
    dy = target_loc.y - spawn_loc.y
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        sys.exit("spawn e target del pedone coincidono: correggere gli indici nello scenario")
    fx, fy = dx / norm, dy / norm
    return (fx, fy), (-fy, fx), math.degrees(math.atan2(fy, fx))


def spawn_traffic(world, library, traffic_manager, n_vehicles, n_walkers, excluded_bps):
    """Traffico NPC di contorno: fornisce le istanze delle classi native da tag.

    I blueprint gia' usati dagli attori in override sono esclusi dal pool NPC. Se
    non lo fossero, lo stesso identico modello comparirebbe nella stessa sessione
    una volta come veicolo_elettrico_silenzioso (per override sull'id) e una volta
    come automobile (per tag): una ground truth che contraddice se' stessa nello
    stesso frame, e nessun controllo strutturale se ne accorgerebbe. Vale anche per
    i pedoni: il modello del conducente del monopattino non gira come NPC.
    """
    actors = []
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    vehicle_bps = [b for b in library.filter("vehicle.*") if b.id not in excluded_bps]
    if not vehicle_bps:
        sys.exit("nessun blueprint veicolo disponibile dopo l'esclusione degli override")
    for tf in spawn_points[:n_vehicles]:
        bp = random.choice(vehicle_bps)
        v = world.try_spawn_actor(bp, tf)
        if v is not None:
            v.set_autopilot(True, traffic_manager.get_port())
            actors.append(v)

    walker_bps = [b for b in library.filter("walker.pedestrian.*") if b.id not in excluded_bps]
    controller_bp = library.find("controller.ai.walker")
    walkers, controllers = [], []
    for _ in range(n_walkers):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        w = world.try_spawn_actor(random.choice(walker_bps), carla.Transform(loc))
        if w is not None:
            walkers.append(w)
    world.tick()
    for w in walkers:
        c = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=w)
        if c is not None:
            controllers.append(c)
    world.tick()
    for c in controllers:
        c.start()
        target = world.get_random_location_from_navigation()
        if target is not None:
            c.go_to_location(target)
        c.set_max_speed(WALKER_SPEED)
    return actors + walkers + controllers


def build_sensors(world, library, parent, origin_above_feet, width, height):
    """I quattro sensori, con attributi rigorosamente identici (R5).

    Risoluzione, FOV e trasformazione relativa vengono da un solo dizionario e da
    una sola Transform, riusati per tutti e quattro. Se i flussi divergessero
    anche solo di un grado di FOV, la maschera non coprirebbe l'oggetto che la
    depth misura, e nessuna eccezione lo segnalerebbe.
    Attachment Rigid e non SpringArm: la firma del moto del passo deve arrivare
    intatta, ed e' il motivo per cui si usa CARLA invece di un dataset pubblico.
    """
    camera_z_relative = CAMERA_Z - origin_above_feet

    attrs = {"image_size_x": str(width), "image_size_y": str(height),
             "fov": str(FOV), "sensor_tick": "0.0"}
    tf = carla.Transform(carla.Location(x=CAMERA_X, z=camera_z_relative))
    ids = {
        "rgb": "sensor.camera.rgb",
        "depth": "sensor.camera.depth",
        "semantic": "sensor.camera.semantic_segmentation",
        "instance": "sensor.camera.instance_segmentation",
    }
    sensors, queues = {}, {}
    for name, bp_id in ids.items():
        bp = resolve_blueprint(library, bp_id)
        for k, v in attrs.items():
            if bp.has_attribute(k):
                bp.set_attribute(k, v)
        s = world.spawn_actor(bp, tf, attach_to=parent,
                              attachment_type=carla.AttachmentType.Rigid)
        q = queue.Queue()
        s.listen(q.put)
        sensors[name], queues[name] = s, q
    return sensors, queues, tf


def pull_synced_frame(queues, expected_frame):
    """R2: un frame da ciascuna coda, con il frame id verificato su tutti e quattro.

    Una coda per sensore e non una condivisa: e' l'unico modo per accorgersi che
    un flusso e' rimasto indietro. Un frame arretrato viene scartato, ma un frame
    che resta disallineato dopo i tentativi previsti e' un errore fatale: senza
    questo controllo la depth finirebbe accanto a un'immagine che non descrive, e
    l'intera sessione sarebbe inutilizzabile senza che nulla lo segnali.
    """
    out = {}
    for name, q in queues.items():
        img = q.get(timeout=QUEUE_TIMEOUT_S)
        tries = 0
        while img.frame < expected_frame and tries < FRAME_SYNC_RETRIES:
            img = q.get(timeout=QUEUE_TIMEOUT_S)
            tries += 1
        if img.frame != expected_frame:
            sys.exit("R2 FALLITO: sensore %s al frame %d, atteso %d."
                     % (name, img.frame, expected_frame))
        out[name] = img
    return out


# --- cattura -----------------------------------------------------------------

class SceneLog:
    """Cio' che lo scenario ha messo nel mondo, e cosa ne e' stato.

    Un attore spawnato dopo world.tick() non e' nelle immagini di quel tick: compare
    dal tick successivo. Per questo entra in `fresh` e passa in `tracked` solo al
    giro dopo (promote). Prima entrava subito, e la prima distanza registrata era
    quella dall'origine del mondo, un frame prima del suo primo pixel (misurato su
    8 attori di 3 sessioni, revisione 05/10).
    """

    def __init__(self):
        self.entries = []      # spawned_actors di session.json
        self.fresh = []        # spawnati in questo giro: nei frame dal prossimo tick
        self.tracked = []      # (actor_id, actor) presenti nei frame
        self.failures = []     # spawn_failures di session.json
        self.actors = []       # tutti gli attori dello scenario, da distruggere alla fine
        self.movers = []       # (ApproachingRider, tick dello spawn)

    def add(self, entries, actors):
        self.entries.extend(entries)
        self.fresh.extend(entries)
        self.actors.extend(actors)

    def fail(self, spec, reason):
        self.failures.append({"boss_class": spec["boss_class"],
                              "declared": spec.get("rider") or spec_label(spec),
                              "reason": reason})

    def promote(self):
        for entry in self.fresh:
            actor = entry["_actor"]
            # La posa vera dopo il primo tick, non quella chiesta allo spawn.
            entry["transform"] = transform_to_dict(actor.get_transform())
            self.tracked.append((entry["actor_id"], actor))
        self.fresh = []


def heading_estimate(recent_locs, current):
    """Direzione di marcia dalla traiettoria recente, oppure la stima precedente.

    Non si assume che rotation.yaw del pedone segua il moto (comportamento interno
    non documentato): si usa lo spostamento effettivo nella finestra.
    """
    ref, cur = recent_locs[0], recent_locs[-1]
    dx, dy = cur.x - ref.x, cur.y - ref.y
    norm = math.hypot(dx, dy)
    if norm <= HEADING_MIN_MOTION_M:
        return current
    fx, fy = dx / norm, dy / norm
    return (fx, fy), (-fy, fx), math.degrees(math.atan2(fy, fx))


def placement_frame(cur_loc, origin_above_feet, heading):
    """Il sistema in cui si piazzano gli ostacoli: posizione, suolo e direzione del pedone."""
    fwd, _, yaw = heading
    return {"origin": (cur_loc.x, cur_loc.y), "ground_z": cur_loc.z - origin_above_feet,
            "fwd": fwd, "yaw": yaw}


def place_one(world, library, spec, frame, ahead, trunks, log, fatal):
    """Piazza un ostacolo dichiarato e lo registra; un fallimento e' fatale se fatal."""
    entry, created, reason = place_obstacle(world, library, spec, frame, ahead, trunks)
    if entry is None:
        log.fail(spec, reason)
        if fatal:
            sys.exit("spawn fallito (%s): %s. In uno scenario Sim2Real un ostacolo mancante"
                     " rende inutile la sessione: ci si ferma qui." % (spec["boss_class"], reason))
        print("ATTENZIONE: spawn fallito (%s): %s. La classe restera' scoperta in questa"
              " sessione, e A7 la segnalera'." % (spec["boss_class"], reason), file=sys.stderr)
        return
    log.add([entry], created)
    loc = entry["transform"]["location"]
    where = ("ancorato al tronco (%.2f, %.2f)" % (entry["anchor"]["trunk"]["x"], entry["anchor"]["trunk"]["y"])
             if "anchor" in entry else "a %.1f m avanti" % ahead)
    print("ostacolo %s: %s a (%.2f, %.2f, %.2f), %s"
          % (spec["boss_class"], spec_label(spec), loc["x"], loc["y"], loc["z"], where))


def place_at_start(world, library, scenario, frame, trunks, log, fixed_delta, tick_index):
    """Scenari Sim2Real: tutti gli ostacoli e i mezzi a "forward" metri dal pedone, subito.

    Si fa all'ultimo tick di warmup: gli attori compaiono dal frame 0, senza il
    pop-in che A8 deve escludere, e "forward" e' la distanza al frame 0.
    """
    for spec in scenario["actors"]:
        place_one(world, library, spec, frame, spec["forward"], trunks, log, fatal=True)
    for i, spec in enumerate(scenario.get("movers", [])):
        group = spec.get("group") or "%s_%d" % (spec["boss_class"], i)
        mover = ApproachingRider(spec, frame, fixed_delta, group)
        reason = mover.spawn(world, library, frame["origin"])
        log.actors.extend(mover.actors())
        if reason is not None:
            log.fail(spec, reason)
            sys.exit("spawn fallito (%s): %s" % (spec["boss_class"], reason))
        log.add(mover.entries, [])
        log.movers.append((mover, tick_index))
        print("mezzo %s: %d attori, parte a %.0f m a %.0f km/h"
              % (group, len(mover.entries), spec["forward"], spec["speed_kmh"]))


def write_png(path, image, digest=None):
    """Codifica PNG, aggiorna l'hash e scrive: un errore di scrittura non passa in silenzio."""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        sys.exit("codifica PNG fallita: %s" % path)
    data = buf.tobytes()
    if digest is not None:
        digest.update(data)
    with open(path, "wb") as f:
        f.write(data)


def record_frame(out_dir, idx, images, hashes):
    stem = "%06d.png" % idx
    write_png(os.path.join(out_dir, "rgb", stem), to_bgra(images["rgb"])[:, :, :3], hashes["rgb"])
    write_png(os.path.join(out_dir, "depth", stem), decode_depth_mm(images["depth"]), hashes["depth"])
    # Semantic e instance si salvano come BGR grezzo, senza convertitore di colore:
    # il tag sta nel canale rosso e l'identita' in G e B, e applicare la
    # CityScapesPalette li distruggerebbe entrambi.
    write_png(os.path.join(out_dir, "semantic", stem), to_bgra(images["semantic"])[:, :, :3])
    write_png(os.path.join(out_dir, "instance", stem), to_bgra(images["instance"])[:, :, :3],
              hashes["instance"])


def pose_record(idx, frame_id, images, cam_tf, tracked):
    """La riga di poses.jsonl di un frame."""
    # Distanza camera-attore dall'aritmetica delle trasformazioni. Non viola R10: non
    # e' una distanza per box ricavata dalla depth, ma la seconda strada esatta del
    # par. 5.5, che serve come controprova indipendente dell'export. E' euclidea
    # fino all'origine dell'attore, non la profondita' z dell'export.
    distances, locations = {}, {}
    for aid, act in tracked:
        loc = act.get_transform().location
        distances[str(aid)] = cam_tf.location.distance(loc)
        locations[str(aid)] = [loc.x, loc.y, loc.z]
    return {
        "index": idx,
        "frame": frame_id,
        "timestamp": images["rgb"].timestamp,
        "frames": dict((k, v.frame) for k, v in images.items()),
        "camera": transform_to_dict(cam_tf),
        "tracked_distances_m": distances,
        # Posizione per frame di ogni attore tracciato, dallo snapshot dello stesso
        # tick dei sensori: per il mezzo in movimento e' la GT della traiettoria.
        "tracked_locations": locations,
    }


def capture(scenario_name, out_dir):
    scenario = SCENARIOS[scenario_name]
    geom = capture_geometry(scenario)
    warmup_ticks = geom["warmup_ticks"]
    spawn_lead_m = geom["spawn_lead_m"]
    at_start = bool(scenario.get("place_at_start"))

    client = carla.Client(HOST, PORT)
    client.set_timeout(CLIENT_TIMEOUT_S)
    carla_version = check_versions(client)

    world = client.load_world(scenario["map"])
    library = world.get_blueprint_library()
    validate_scenario(scenario, library)
    validate_meshes(world, library, scenario)
    anchored = any(s.get("anchor") == "tree" for s in scenario["actors"])
    trunks = tree_trunks(world) if anchored else []

    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager()
    sensors, extra_actors = {}, []
    log = SceneLog()

    try:
        # R8: determinismo. I semi vanno fissati PRIMA di qualunque spawn, e quello
        # dei pedoni prima che la nav mesh venga interrogata, altrimenti il pool di
        # locazioni cambia a ogni esecuzione e il confronto A6 non ha senso.
        seed = scenario["seed"]
        random.seed(seed)
        np.random.seed(seed)
        traffic_manager.set_random_device_seed(seed)
        world.set_pedestrians_seed(seed)

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = geom["fixed_delta"]
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)

        nav_pool = build_nav_pool(world, NAV_POOL_SIZE)
        spawn_loc = nav_pool[scenario["walker_spawn_index"] % len(nav_pool)]
        # Il target NON e' l'indice configurato in walker_target_index: il pool e'
        # generato da un seed diverso per ogni scenario (R8), quindi un indice
        # fisso puo' dare una coppia spawn/target vicinissima con un altro seed.
        # Misurato: silent_vehicles (seed=1) percorreva solo 11 m contro i 30 m
        # richiesti dal prop piu' lontano, perche' il pedone raggiungeva il target
        # e si fermava li'. Si sceglie invece il punto del pool piu' lontano dallo
        # spawn: massimizza la distanza percorribile qualunque sia il seed, e
        # walker_target_index resta nello scenario solo come documentazione
        # dell'intento originale.
        target_loc = max(nav_pool, key=spawn_loc.distance)
        fwd, right, base_yaw = path_frame(spawn_loc, target_loc)

        world.set_weather(getattr(carla.WeatherParameters, scenario["weather"]))
        if "wind_intensity" in scenario:
            # Il vento muove le foglie con una fase che dipende dal tempo del mondo, non
            # riproducibile: due catture identiche differivano sul 2,5-6 % della depth,
            # quasi tutto vegetazione (revisione 05/10). Per i rami, a vento zero la
            # sessione si ripete e la chioma non trema fra un frame e l'altro.
            weather = world.get_weather()
            weather.wind_intensity = float(scenario["wind_intensity"])
            world.set_weather(weather)

        walker_bps = library.filter("walker.pedestrian.*")
        walker_bp = walker_bps[scenario["walker_bp_index"] % len(walker_bps)]
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "true")
        walker = world.try_spawn_actor(
            walker_bp, carla.Transform(spawn_loc, carla.Rotation(yaw=base_yaw)))
        if walker is None:
            sys.exit("spawn del pedone fallito: cambiare walker_spawn_index nello scenario")
        extra_actors.append(walker)

        # Serve sia per la camera (build_sensors) sia per piazzare i prop a terra
        # (walker.get_transform().location.z NON e' la quota del suolo): un solo
        # valore, per evitare che le due correzioni possano mai divergere.
        origin_above_feet = walker_origin_above_feet(walker)

        controller = world.spawn_actor(library.find("controller.ai.walker"),
                                       carla.Transform(), attach_to=walker)
        extra_actors.append(controller)

        # Il controller AI ha bisogno di un paio di tick di assestamento dopo lo
        # spawn prima che go_to_location abbia effetto: stesso pattern gia' usato
        # per i pedoni NPC in spawn_traffic(). Misurato: senza questi tick il
        # comando viene ignorato in silenzio e il pedone resta fermo al punto di
        # spawn.
        for _ in range(CONTROLLER_SETTLE_TICKS):
            world.tick()

        controller.start()
        controller.go_to_location(target_loc)
        controller.set_max_speed(WALKER_SPEED)

        excluded = set(s["blueprint"] for s in scenario["actors"] if "blueprint" in s)
        excluded |= set(m["rider"] for m in scenario.get("movers", []))
        extra_actors.extend(spawn_traffic(world, library, traffic_manager,
                                          scenario["traffic_vehicles"],
                                          scenario["traffic_walkers"], excluded))

        sensors, queues, camera_tf = build_sensors(world, library, walker, origin_above_feet,
                                                   geom["width"], geom["height"])

        for sub in ("rgb", "depth", "semantic", "instance"):
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

        camera = sensors["rgb"]

        # Due modi di piazzare gli ostacoli. Negli scenari Sim2Real (place_at_start)
        # tutti insieme all'ultimo tick di warmup, a "forward" metri dal pedone. Negli
        # altri UNO ALLA VOLTA durante il cammino: il controller AI segue la nav mesh,
        # che curva entro pochi metri, e qualunque stima statica di direzione presa
        # all'inizio resta valida solo vicino a dove e' stata misurata (misurato: a
        # meta' cammino il pedone passava a 1,5-2 m dal prop ma girato altrove).
        # Ogni ostacolo si piazza quando la distanza REALMENTE percorsa raggiunge
        # spec["forward"] - SPAWN_LEAD_M, con la direzione ATTUALE del pedone.
        pending = [] if at_start else list(scenario["actors"])
        placed = not at_start
        path_distance = 0.0
        last_loc = walker.get_transform().location
        # Finestra di posizioni recenti per stimare la direzione dalla traiettoria
        # effettiva: 0,5 s (DIRECTION_WINDOW_S, 8 tick a 16 Hz, 24 a 48 Hz) bastano a
        # mediare il rumore tick-per-tick restando comunque locali nel tempo.
        recent_locs = deque(maxlen=geom["direction_window"])
        recent_locs.append(last_loc)
        # Stima di riserva finche' la finestra non e' stabile: la direzione naive
        # spawn->target e' comunque meglio di niente.
        heading = (fwd, right, base_yaw)
        cur_loc = last_loc

        # A6: gli hash si accumulano sui byte PNG mentre vengono scritti, cosi'
        # session.json resta scritto una volta sola a fine sessione (R7).
        hashes = {"rgb": hashlib.sha256(), "depth": hashlib.sha256(), "instance": hashlib.sha256()}

        n_frames = scenario["n_frames"]
        poses_path = os.path.join(out_dir, "poses.jsonl")
        t0 = time.time()
        written = 0

        with open(poses_path, "w") as poses:
            for tick_index in range(warmup_ticks + n_frames):
                expected = world.tick()
                log.promote()

                cur_loc = walker.get_transform().location
                path_distance += last_loc.distance(cur_loc)
                last_loc = cur_loc
                recent_locs.append(cur_loc)
                window_full = len(recent_locs) == recent_locs.maxlen

                if not placed and tick_index == warmup_ticks - 1:
                    if not window_full:
                        sys.exit("warmup piu' corto della finestra di direzione: impossibile"
                                 " stimare la marcia prima di piazzare gli ostacoli")
                    heading = heading_estimate(recent_locs, heading)
                    frame = placement_frame(cur_loc, origin_above_feet, heading)
                    place_at_start(world, library, scenario, frame, trunks, log,
                                   geom["fixed_delta"], tick_index)
                    placed = True
                # Si valuta lo spawn dei prop solo a warmup concluso e con la finestra
                # piena. Misurato (silent_vehicles, prop a forward=5,0 m): subito dopo
                # start()/go_to_location() il controller AI puo' ruotare sul posto
                # prima di camminare; un trigger in quella fase prende la direzione da
                # uno spostamento quasi casuale (errore osservato: 70 gradi).
                elif pending and tick_index >= warmup_ticks and window_full:
                    heading = heading_estimate(recent_locs, heading)
                    frame = placement_frame(cur_loc, origin_above_feet, heading)
                    still_pending = []
                    for spec in pending:
                        if path_distance < spec["forward"] - spawn_lead_m:
                            still_pending.append(spec)
                            continue
                        place_one(world, library, spec, frame, spawn_lead_m, trunks, log, fatal=False)
                    pending = still_pending

                # La posa comandata ora viene resa al prossimo tick: il frame k+1
                # mostra il mezzo dopo k tick di moto, alla velocita' dichiarata.
                for mover, spawn_tick in log.movers:
                    mover.update(tick_index - spawn_tick, (cur_loc.x, cur_loc.y))

                images = pull_synced_frame(queues, expected)
                if tick_index < warmup_ticks:
                    continue

                idx = tick_index - warmup_ticks
                record_frame(out_dir, idx, images, hashes)
                poses.write(json.dumps(pose_record(idx, expected, images, camera.get_transform(),
                                                   log.tracked)) + "\n")
                written += 1

                if idx % PROGRESS_EVERY == 0:
                    print("frame %d/%d" % (idx, n_frames), flush=True)

        # Ostacoli mai raggiunti (il pedone non ha percorso abbastanza strada entro
        # fine sessione): si piazzano comunque, con un avviso, invece di sparire in
        # silenzio da session.json. Non compaiono in nessun frame, e A7 lo dice.
        if pending:
            heading = heading_estimate(recent_locs, heading)
            frame = placement_frame(cur_loc, origin_above_feet, heading)
            for spec in pending:
                print("ATTENZIONE: %s non raggiunto entro fine sessione (percorsi %.1f m,"
                      " ne servivano %.1f): piazzato alla posizione finale del pedone."
                      % (spec["boss_class"], path_distance, spec["forward"]), file=sys.stderr)
                place_one(world, library, spec, frame, spawn_lead_m, trunks, log, fatal=False)

        elapsed = time.time() - t0

        # R7: metadati di sessione scritti UNA VOLTA, completi. Senza fattore di
        # scala e intrinseci i PNG non sono interpretabili in metri; senza seme e
        # versione la sessione non e' riproducibile; senza l'elenco degli attori
        # spawnati con la classe assegnata, carla_export.py non ha l'override e le
        # classi coperte solo da prop restano invisibili.
        session = {
            "scenario": scenario_name,
            "scenario_config": dict((k, v) for k, v in scenario.items()),
            "carla_version": carla_version,
            "code_version": code_version(),
            "map": scenario["map"],
            "weather": scenario["weather"],
            "wind_intensity": scenario.get("wind_intensity"),
            "seed": scenario["seed"],
            "fixed_delta_seconds": geom["fixed_delta"],
            "warmup_ticks": warmup_ticks,
            "width": geom["width"],
            "height": geom["height"],
            "fov": FOV,
            # CARLA fissa il FOV orizzontale: a 16:9 il verticale scende a ~52
            # gradi contro i ~66 del 4:3 della stereocamera, e un ostacolo a quota
            # testa esce di campo piu' presto. Dichiararlo evita di scoprirlo
            # confrontando sessioni con aspect diverso.
            "vertical_fov": vertical_fov_deg(geom["width"], geom["height"], FOV),
            "intrinsics": compute_intrinsics(geom["width"], geom["height"], FOV),
            "depth_scale": DEPTH_SCALE,
            "depth_unit": "mm",
            "depth_max_u16": DEPTH_MAX_U16,
            "camera_transform_relative": transform_to_dict(camera_tf),
            "walker_speed_mps": WALKER_SPEED,
            "walker_spawn": {"x": spawn_loc.x, "y": spawn_loc.y, "z": spawn_loc.z},
            "walker_target": {"x": target_loc.x, "y": target_loc.y, "z": target_loc.z},
            # Distanza REALMENTE percorsa dal pedone (lungo la nav mesh, non in
            # linea d'aria): e' la metrica con cui e' stato deciso quando piazzare
            # ciascun ostacolo negli scenari a piazzamento progressivo.
            "path_distance_walked_m": path_distance,
            "place_at_start": at_start,
            "spawn_lead_m": None if at_start else spawn_lead_m,
            "spawned_actors": [dict((k, v) for k, v in a.items() if k != "_actor")
                               for a in log.entries],
            "spawn_failures": log.failures,
            "movers": [m.describe() for m, _ in log.movers],
            "n_frames_written": written,
            "rgb_sha256": hashes["rgb"].hexdigest(),
            "depth_sha256": hashes["depth"].hexdigest(),
            "instance_sha256": hashes["instance"].hexdigest(),
            "capture_elapsed_s": elapsed,
            "captured_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(os.path.join(out_dir, "session.json"), "w") as f:
            json.dump(session, f, indent=2)

        print("\n--- sintesi ---")
        print("sessione:        %s" % out_dir)
        print("frame scritti:   %d" % written)
        print("tempo:           %.1f s (%.2f frame/s)"
              % (elapsed, written / elapsed if elapsed > 0 else 0.0))
        print("sha256 RGB:      %s" % session["rgb_sha256"])
        return out_dir

    finally:
        cleanup(client, world, traffic_manager, sensors, extra_actors + log.actors, original_settings)


def cleanup(client, world, traffic_manager, sensors, actors, original_settings):
    """R9: lasciare il server come lo si e' trovato, anche se la sessione e' fallita.

    La trappola piu' frequente: se lo script termina male senza questo blocco, il
    server resta in modalita' sincrona con nessuno che faccia tick e la sessione
    successiva si blocca all'avvio senza spiegazione.

    Il primo giro reale (urban_day, 300 frame) ha mostrato un crash nativo del client
    proprio qui: i dati erano gia' scritti su disco, ma l'uscita non era pulita. E' un
    problema noto della community CARLA: distruggere gli attori uno a uno con RPC
    separate, mentre il traffic manager li referenzia ancora, senza un tick che faccia
    processare le rimozioni al server prima di disattivare la modalita' sincrona. Si
    adotta lo stesso pattern degli script ufficiali (generate_traffic.py): fermare
    prima i listener (sensori, controller AI), distruggere tutto in un solo
    client.apply_batch, poi un tick MENTRE si e' ancora sincroni, e solo alla fine
    disattivare il sync.
    """
    for s in sensors.values():
        try:
            s.stop()
        except RuntimeError:
            pass
    for a in actors:
        try:
            if a.type_id.startswith("controller."):
                a.stop()
        except RuntimeError:
            pass

    to_destroy = list(sensors.values()) + list(reversed(actors))
    if to_destroy:
        try:
            client.apply_batch([carla.command.DestroyActor(a) for a in to_destroy])
        except RuntimeError as exc:
            print("ATTENZIONE: distruzione degli attori non riuscita: %s" % exc, file=sys.stderr)

    try:
        world.tick()
    except RuntimeError:
        pass

    try:
        traffic_manager.set_synchronous_mode(False)
    except RuntimeError:
        pass
    world.apply_settings(original_settings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Registra una sessione CARLA con ground truth densa (Fase A, OR 4.1).")
    parser.add_argument("--scenario", required=True, choices=sorted(SCENARIOS.keys()),
                        help="scenario da registrare, definito in scenarios.py")
    parser.add_argument("--out", default=None,
                        help="cartella di destinazione (default:"
                             " outputs/<scenario>_<timestamp>)")
    args = parser.parse_args()

    out = args.out or os.path.join(
        "outputs", "%s_%s" % (args.scenario, datetime.now().strftime("%Y%m%d_%H%M%S")))
    os.makedirs(out, exist_ok=True)

    capture(args.scenario, out)
    results, info = run_acceptance_checks(out)
    write_acceptance(out, results, info)
    sys.exit(1 if print_acceptance(results) else 0)
