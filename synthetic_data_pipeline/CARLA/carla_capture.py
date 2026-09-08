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
import argparse
import hashlib
import json
import math
import os
import queue
import random
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

import carla

from boss_classes import BOSS_CLASS_BY_NAME, CARLA_TAG_NAMES, TAG_SKY
from scenarios import SCENARIOS

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

# Controlli di accettazione: parametri.
A3_SAMPLE_FRAMES = 20       # frame campionati per le verifiche statistiche
A4_SKY_TOLERANCE = 0.99     # quota minima di pixel di cielo che deve stare al massimo
A7_MIN_FRAMES = 10          # in quanti frame almeno un attore tracciato deve essere visibile


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


def transform_to_dict(tf):
    return {
        "location": {"x": tf.location.x, "y": tf.location.y, "z": tf.location.z},
        "rotation": {"pitch": tf.rotation.pitch, "yaw": tf.rotation.yaw, "roll": tf.rotation.roll},
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


def resolve_blueprint(library, bp_id):
    """Trova un blueprint, oppure si ferma stampando i candidati simili.

    L'inventario degli asset cambia fra versioni e fra installazioni (par. 5.2):
    invece di assumere che gli id dello scenario esistano, li si verifica tutti
    prima di registrare. Senza questo controllo un id sbagliato non produce una
    scena mancante ma una scena diversa, e ce ne si accorge a sessione finita.
    """
    try:
        return library.find(bp_id)
    except (IndexError, RuntimeError):
        pass
    family = bp_id.rsplit(".", 1)[0]
    candidates = sorted(b.id for b in library.filter(family + ".*"))
    print("blueprint non trovato: %s" % bp_id, file=sys.stderr)
    if candidates:
        print("candidati nella famiglia %s.* su questa installazione:" % family, file=sys.stderr)
        for c in candidates:
            print("   %s" % c, file=sys.stderr)
    else:
        print("nessun blueprint nella famiglia %s.*" % family, file=sys.stderr)
    sys.exit("scenario non eseguibile: correggere scenarios.py")


def validate_scenario(scenario, library):
    """Verifica blueprint e classi BOSS dello scenario prima di toccare il mondo."""
    for spec in scenario["actors"]:
        resolve_blueprint(library, spec["blueprint"])
        if spec["boss_class"] not in BOSS_CLASS_BY_NAME:
            sys.exit("classe BOSS sconosciuta nello scenario: %s" % spec["boss_class"])


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


def spawn_one_actor(world, library, spec, loc, yaw):
    """Spawna un singolo ostacolo dello scenario nella posizione e nell'orientamento dati.

    Ogni attore riceve la classe BOSS dichiarata nello scenario. E' questo, non il
    tag semantico, cio' che rende utilizzabili gli ostacoli personalizzati
    (par. 5.4): un prop spawnato porta il tag della sua mesh - tipicamente Static
    (20) o Other (22) - e mai una classe BOSS.

    Restituisce None se lo spawn fallisce (collisione con la geometria della
    mappa): il chiamante decide se e' fatale o se la classe resta scoperta.
    """
    bp = resolve_blueprint(library, spec["blueprint"])
    tf = carla.Transform(loc, carla.Rotation(yaw=yaw + spec["yaw"]))
    actor = world.try_spawn_actor(bp, tf)
    if actor is None:
        print("ATTENZIONE: spawn fallito per %s (%s): collisione con la geometria"
              " della mappa. La classe %s restera' scoperta in questa sessione."
              % (spec["blueprint"], spec["boss_class"], spec["boss_class"]),
              file=sys.stderr)
        return None
    if not spec["physics"]:
        # R6: senza questo gli ostacoli sospesi cadono al primo tick.
        actor.set_simulate_physics(False)
    return {
        "actor_id": actor.id,
        # L'instance segmentation codifica l'identita' su 16 bit fra i canali G e B
        # dell'immagine (R e' il tag semantico). La doc ufficiale descrive la
        # ricomposizione come (G<<8)|B, ma su questa installazione (0.9.16) la
        # verifica empirica su due prop spawnati (confronto actor.id noto contro i
        # pixel effettivamente renderizzati) mostra il contrario: (B<<8)|G, canale
        # B byte alto. Questo e' il valore, non actor.id, che si legge
        # dall'immagine - vedi run_acceptance_checks() piu' sotto e la stessa
        # formula in carla_export.py, che deve restare identica a questa.
        "instance_id": actor.id & 0xFFFF,
        "blueprint": spec["blueprint"],
        "boss_class": spec["boss_class"],
        "boss_class_id": BOSS_CLASS_BY_NAME[spec["boss_class"]]["id"],
        "transform": transform_to_dict(tf),
        "_actor": actor,
    }


def spawn_traffic(world, library, traffic_manager, n_vehicles, n_walkers, excluded_bps):
    """Traffico NPC di contorno: fornisce le istanze delle classi native da tag.

    I blueprint gia' usati dagli attori in override sono esclusi dal pool NPC. Se
    non lo fossero, lo stesso identico modello comparirebbe nella stessa sessione
    una volta come veicolo_elettrico_silenzioso (per override sull'id) e una volta
    come automobile (per tag): una ground truth che contraddice se' stessa nello
    stesso frame, e nessun controllo strutturale se ne accorgerebbe.
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

    walker_bps = library.filter("walker.pedestrian.*")
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


def walker_origin_above_feet(walker):
    """Quanto l'origine dell'attore pedone sta sopra i suoi stessi piedi, in metri.

    L'origine non e' ai piedi ma al centro della capsula fisica (misurato: con un
    offset relativo fisso la camera finiva a ~2,6 m dal suolo invece dei 1,5 m di
    torace/testa richiesti dal par. 5.6, e un prop piazzato a walker.z + 0 finiva
    a ~1,1 m di altezza invece che a terra). Si ricava dalla bounding box
    dell'attore, generica per qualunque blueprint pedone: la quota dei piedi
    rispetto all'origine e' bb.location.z - bb.extent.z, quindi l'origine sta
    sopra i piedi di bb.extent.z - bb.location.z. Va sottratta ovunque nel codice
    si usi walker.get_transform().location.z come se fosse la quota del suolo.
    """
    bb = walker.bounding_box
    return bb.extent.z - bb.location.z


def build_sensors(world, library, parent, origin_above_feet):
    """I quattro sensori, con attributi rigorosamente identici (R5).

    Risoluzione, FOV e trasformazione relativa vengono da un solo dizionario e da
    una sola Transform, riusati per tutti e quattro. Se i flussi divergessero
    anche solo di un grado di FOV, la maschera non coprirebbe l'oggetto che la
    depth misura, e nessuna eccezione lo segnalerebbe.
    Attachment Rigid e non SpringArm: la firma del moto del passo deve arrivare
    intatta, ed e' il motivo per cui si usa CARLA invece di un dataset pubblico.
    """
    camera_z_relative = CAMERA_Z - origin_above_feet

    attrs = {"image_size_x": str(WIDTH), "image_size_y": str(HEIGHT),
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

def capture(scenario_name, out_dir):
    scenario = SCENARIOS[scenario_name]

    client = carla.Client(HOST, PORT)
    client.set_timeout(CLIENT_TIMEOUT_S)
    carla_version = check_versions(client)

    world = client.load_world(scenario["map"])
    library = world.get_blueprint_library()
    validate_scenario(scenario, library)

    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager()
    sensors, extra_actors, scenario_actors = {}, [], []

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
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
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

        extra_actors.extend(spawn_traffic(world, library, traffic_manager,
                                          scenario["traffic_vehicles"],
                                          scenario["traffic_walkers"],
                                          set(s["blueprint"] for s in scenario["actors"])))

        sensors, queues, camera_tf = build_sensors(world, library, walker, origin_above_feet)

        for sub in ("rgb", "depth", "semantic", "instance"):
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

        camera = sensors["rgb"]

        # Gli ostacoli dello scenario si spawnano UNO ALLA VOLTA durante il cammino,
        # non tutti prima: due sessioni reali hanno mostrato che il controller AI
        # segue la nav mesh (il marciapiede), che curva entro pochi metri, e
        # qualunque stima statica di direzione presa all'inizio resta valida solo
        # vicino a dove e' stata misurata (misurato: a meta' cammino il pedone
        # passava a 1,5-2 m dal prop ma girato altrove, fuori dal cono di ripresa).
        #
        # pending resta la lista degli ostacoli non ancora spawnati; quando la
        # distanza REALMENTE percorsa (path_distance, non la linea d'aria dal
        # punto di spawn) raggiunge spec["forward"] - SPAWN_LEAD_M, l'ostacolo
        # viene piazzato usando la direzione ATTUALE del pedone: l'errore di
        # estrapolazione si riduce cosi' a pochi metri invece che all'intera
        # distanza del prop, indipendentemente da quanto il percorso curva.
        pending = list(scenario["actors"])
        scenario_actors = []
        tracked = []
        path_distance = 0.0
        last_loc = walker.get_transform().location
        # Finestra di posizioni recenti per stimare la direzione dalla traiettoria
        # effettiva, senza assumere che rotation.yaw del pedone segua il moto
        # (comportamento interno non documentato): 8 tick (0,5 s) bastano a
        # mediare il rumore tick-per-tick restando comunque locali nel tempo.
        recent_locs = deque(maxlen=9)
        recent_locs.append(last_loc)
        # Stima di riserva finche' la finestra non e' stabile, o se un ostacolo
        # resta ancora da piazzare a fine sessione (fallback dopo il ciclo, dove
        # cur_loc resta quello dell'ultimo tick eseguito): la direzione naive
        # spawn->target e' comunque meglio di niente.
        cur_fwd, cur_right, cur_yaw = fwd, right, base_yaw
        cur_loc = last_loc

        # A6: l'hash si accumula sui byte PNG mentre vengono scritti, cosi'
        # session.json resta scritto una volta sola a fine sessione (R7) e non
        # serve rileggere l'intera cartella per ottenerlo.
        rgb_hash = hashlib.sha256()

        n_frames = scenario["n_frames"]
        poses_path = os.path.join(out_dir, "poses.jsonl")
        t0 = time.time()
        written = 0

        with open(poses_path, "w") as poses:
            for tick_index in range(WARMUP_TICKS + n_frames):
                expected = world.tick()

                cur_loc = walker.get_transform().location
                path_distance += last_loc.distance(cur_loc)
                last_loc = cur_loc
                recent_locs.append(cur_loc)

                # Si valuta lo spawn dei prop solo a warmup concluso e con la
                # finestra di posizioni recenti piena. Misurato (silent_vehicles,
                # prop a forward=5,0 m): subito dopo start()/go_to_location() il
                # controller AI puo' ruotare sul posto verso il target prima di
                # avviare la locomozione vera; se il trigger cade in quella fase la
                # stima di direzione e' presa su uno spostamento piccolo e quasi
                # casuale (errore osservato: 70 gradi), e il prop finisce fuori dal
                # cono di ripresa per l'intera sessione. Il WARMUP_TICKS esiste gia'
                # per LOD/meteo (par. 14): qui in piu' garantisce che il pedone
                # cammini per davvero prima che qualunque prop venga piazzato.
                if pending and tick_index >= WARMUP_TICKS and len(recent_locs) == recent_locs.maxlen:
                    ref_loc = recent_locs[0]
                    dx, dy = cur_loc.x - ref_loc.x, cur_loc.y - ref_loc.y
                    norm = math.hypot(dx, dy)
                    # Sotto pochi centimetri la direzione e' rumore di posa, non
                    # moto: si tiene la stima precedente invece di aggiornarla.
                    if norm > 0.05:
                        fx, fy = dx / norm, dy / norm
                        cur_fwd, cur_right = (fx, fy), (-fy, fx)
                        cur_yaw = math.degrees(math.atan2(fy, fx))

                    still_pending = []
                    for spec in pending:
                        if path_distance < spec["forward"] - SPAWN_LEAD_M:
                            still_pending.append(spec)
                            continue
                        loc = carla.Location(
                            x=cur_loc.x + cur_fwd[0] * SPAWN_LEAD_M + cur_right[0] * spec["lateral"],
                            y=cur_loc.y + cur_fwd[1] * SPAWN_LEAD_M + cur_right[1] * spec["lateral"],
                            z=(cur_loc.z - origin_above_feet) + spec["z"],
                        )
                        result = spawn_one_actor(world, library, spec, loc, cur_yaw)
                        if result is not None:
                            scenario_actors.append(result)
                            extra_actors.append(result["_actor"])
                            tracked.append((result["actor_id"], result["_actor"]))
                    pending = still_pending

                images = pull_synced_frame(queues, expected)
                if tick_index < WARMUP_TICKS:
                    continue

                idx = tick_index - WARMUP_TICKS
                stem = "%06d.png" % idx

                ok, buf = cv2.imencode(".png", to_bgra(images["rgb"])[:, :, :3])
                if not ok:
                    sys.exit("codifica PNG dell'RGB fallita al frame %d" % idx)
                raw_png = buf.tobytes()
                rgb_hash.update(raw_png)
                with open(os.path.join(out_dir, "rgb", stem), "wb") as f:
                    f.write(raw_png)

                cv2.imwrite(os.path.join(out_dir, "depth", stem),
                            decode_depth_mm(images["depth"]))
                # Semantic e instance si salvano come BGR grezzo, senza convertitore
                # di colore: il tag sta nel canale rosso e l'identita' in G e B, e
                # applicare la CityScapesPalette li distruggerebbe entrambi.
                cv2.imwrite(os.path.join(out_dir, "semantic", stem),
                            to_bgra(images["semantic"])[:, :, :3])
                cv2.imwrite(os.path.join(out_dir, "instance", stem),
                            to_bgra(images["instance"])[:, :, :3])

                cam_tf = camera.get_transform()
                # Distanza camera-attore dall'aritmetica delle trasformazioni. Non
                # viola R10: non e' una distanza per box ricavata dalla depth, ma la
                # seconda strada esatta del par. 5.5, che serve come controprova
                # indipendente dell'export. Se le due divergono, il colpevole e' la
                # scala della depth (R4) o l'allineamento dei sensori (R5).
                distances = dict(
                    (str(aid), cam_tf.location.distance(act.get_transform().location))
                    for aid, act in tracked)

                poses.write(json.dumps({
                    "index": idx,
                    "frame": expected,
                    "timestamp": images["rgb"].timestamp,
                    "frames": dict((k, v.frame) for k, v in images.items()),
                    "camera": transform_to_dict(cam_tf),
                    "tracked_distances_m": distances,
                }) + "\n")
                written += 1

                if idx % PROGRESS_EVERY == 0:
                    print("frame %d/%d" % (idx, n_frames), flush=True)

        # Ostacoli mai raggiunti (il pedone non ha percorso abbastanza strada entro
        # fine sessione): si piazzano comunque, con un avviso, invece di sparire in
        # silenzio da session.json. cur_loc/cur_fwd/cur_right/cur_yaw sono quelli
        # dell'ultimo tick registrato nel ciclo sopra.
        for spec in pending:
            print("ATTENZIONE: %s non raggiunto entro fine sessione (percorsi %.1f m,"
                  " ne servivano %.1f): piazzato alla posizione finale del pedone."
                  % (spec["boss_class"], path_distance, spec["forward"]), file=sys.stderr)
            loc = carla.Location(
                x=cur_loc.x + cur_fwd[0] * SPAWN_LEAD_M + cur_right[0] * spec["lateral"],
                y=cur_loc.y + cur_fwd[1] * SPAWN_LEAD_M + cur_right[1] * spec["lateral"],
                z=(cur_loc.z - origin_above_feet) + spec["z"],
            )
            result = spawn_one_actor(world, library, spec, loc, cur_yaw)
            if result is not None:
                scenario_actors.append(result)
                extra_actors.append(result["_actor"])
                tracked.append((result["actor_id"], result["_actor"]))

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
            "map": scenario["map"],
            "weather": scenario["weather"],
            "seed": scenario["seed"],
            "fixed_delta_seconds": FIXED_DELTA_SECONDS,
            "warmup_ticks": WARMUP_TICKS,
            "width": WIDTH,
            "height": HEIGHT,
            "fov": FOV,
            "intrinsics": compute_intrinsics(WIDTH, HEIGHT, FOV),
            "depth_scale": DEPTH_SCALE,
            "depth_unit": "mm",
            "depth_max_u16": DEPTH_MAX_U16,
            "camera_transform_relative": transform_to_dict(camera_tf),
            "walker_speed_mps": WALKER_SPEED,
            "walker_spawn": {"x": spawn_loc.x, "y": spawn_loc.y, "z": spawn_loc.z},
            "walker_target": {"x": target_loc.x, "y": target_loc.y, "z": target_loc.z},
            # Distanza REALMENTE percorsa dal pedone (lungo la nav mesh, non in
            # linea d'aria): e' la metrica con cui e' stato deciso quando piazzare
            # ciascun ostacolo (vedi SPAWN_LEAD_M), non la retta spawn->target, che
            # serve solo da stima iniziale di riserva.
            "path_distance_walked_m": path_distance,
            "spawn_lead_m": SPAWN_LEAD_M,
            "spawned_actors": [dict((k, v) for k, v in a.items() if k != "_actor")
                               for a in scenario_actors],
            "n_frames_written": written,
            "rgb_sha256": rgb_hash.hexdigest(),
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
        # R9: la trappola piu' frequente. Se lo script termina male senza questo
        # blocco, il server resta in modalita' sincrona con nessuno che faccia tick
        # e la sessione successiva si blocca all'avvio senza spiegazione.
        #
        # Il primo giro reale (urban_day, 300 frame) ha mostrato un crash nativo del
        # client proprio in questo blocco: i dati erano gia' scritti su disco (nessuna
        # perdita), ma l'uscita non era pulita. E' un problema noto della community
        # CARLA: distruggere gli attori uno a uno con RPC separate, mentre il traffic
        # manager li referenzia ancora, senza un tick che faccia processare le
        # rimozioni al server prima di disattivare la modalita' sincrona. Si adotta lo
        # stesso pattern degli script ufficiali (PythonAPI/examples/generate_traffic.py):
        # fermare prima i listener (sensori, controller AI: richiedono stop()
        # esplicito), distruggere tutto in un solo client.apply_batch invece di N
        # round-trip separati, poi un tick MENTRE si e' ancora sincroni per far
        # processare i comandi al server, e solo alla fine disattivare il sync.
        for s in sensors.values():
            try:
                s.stop()
            except RuntimeError:
                pass
        for a in extra_actors:
            try:
                if a.type_id.startswith("controller."):
                    a.stop()
            except RuntimeError:
                pass

        to_destroy = list(sensors.values()) + list(reversed(extra_actors))
        if to_destroy:
            try:
                client.apply_batch([carla.command.DestroyActor(a) for a in to_destroy])
            except RuntimeError:
                pass

        try:
            world.tick()
        except RuntimeError:
            pass

        try:
            traffic_manager.set_synchronous_mode(False)
        except RuntimeError:
            pass
        world.apply_settings(original_settings)


# --- controlli di accettazione A1-A7 -----------------------------------------

def run_acceptance_checks(out_dir):
    """I sette controlli del par. 14.3, da superare tutti prima di usare la sessione.

    Sono verifiche di poche righe che intercettano errori i quali non producono
    eccezioni: una depth decodificata male e' plausibile all'occhio, e dei flussi
    disallineati producono file perfettamente validi.
    """
    results = []

    def record(name, ok, detail):
        results.append((name, ok, detail))

    session_path = os.path.join(out_dir, "session.json")
    if not os.path.exists(session_path):
        record("A5", False, "session.json assente")
        return results
    with open(session_path) as f:
        session = json.load(f)

    dirs = ["rgb", "depth", "semantic", "instance"]
    counts = dict((d, sorted(os.listdir(os.path.join(out_dir, d)))) for d in dirs)
    with open(os.path.join(out_dir, "poses.jsonl")) as f:
        poses = [json.loads(l) for l in f if l.strip()]

    n = len(counts["rgb"])
    a1 = all(len(counts[d]) == n for d in dirs) and len(poses) == n
    record("A1", a1, "rgb=%d depth=%d semantic=%d instance=%d poses=%d"
           % (len(counts["rgb"]), len(counts["depth"]), len(counts["semantic"]),
              len(counts["instance"]), len(poses)))

    bad = [p["index"] for p in poses if len(set(p["frames"].values())) != 1]
    record("A2", not bad, "ok su %d frame" % len(poses) if not bad
           else "frame id discordi agli indici %s" % bad[:5])

    step = max(1, n // A3_SAMPLE_FRAMES) if n else 1
    sample = counts["depth"][::step][:A3_SAMPLE_FRAMES]

    lo_all, hi_all, zeros, sat = [], [], 0, 0
    for name in sample:
        d = cv2.imread(os.path.join(out_dir, "depth", name), cv2.IMREAD_UNCHANGED)
        lo_all.append(float(np.percentile(d, 5)))
        hi_all.append(float(np.percentile(d, 95)))
        zeros += int((d == 0).all())
        sat += int((d == DEPTH_MAX_U16).all())
    a3 = bool(sample) and zeros == 0 and sat == 0 and max(hi_all) > min(lo_all)
    record("A3", a3, "p5 min=%.0f mm, p95 max=%.0f mm, frame tutti-zero=%d tutti-saturi=%d"
           % (min(lo_all) if lo_all else 0, max(hi_all) if hi_all else 0, zeros, sat)
           if sample else "nessun frame da campionare")

    # A4, invariante cielo: e' il controllo piu' informativo dei sette. Il cielo in
    # CARLA sta a 1000 m, cioe' ben oltre i 65,5 m rappresentabili: dopo il clip
    # DEVE valere esattamente 65535. Se non vale, o la decodifica R3 e' sbagliata,
    # o i flussi sono disallineati (R2/R5), e in entrambi i casi la depth non
    # descrive l'immagine che le sta accanto.
    sky_total, sky_ok, frames_with_sky = 0, 0, 0
    for name in sample:
        sem = cv2.imread(os.path.join(out_dir, "semantic", name), cv2.IMREAD_UNCHANGED)
        dep = cv2.imread(os.path.join(out_dir, "depth", name), cv2.IMREAD_UNCHANGED)
        mask = sem[:, :, 2] == TAG_SKY          # canale rosso: cv2 legge in BGR
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        frames_with_sky += 1
        sky_total += cnt
        sky_ok += int((dep[mask] == DEPTH_MAX_U16).sum())
    if frames_with_sky == 0:
        record("A4", False, "nessun pixel di cielo nei frame campionati:"
                            " controllo non eseguibile, inquadratura da rivedere")
    else:
        ratio = sky_ok / float(sky_total)
        record("A4", ratio >= A4_SKY_TOLERANCE,
               "%.4f dei pixel di cielo al massimo (soglia %.2f), su %d frame"
               % (ratio, A4_SKY_TOLERANCE, frames_with_sky))

    required = ["intrinsics", "fov", "width", "height", "depth_scale", "carla_version",
                "map", "seed", "fixed_delta_seconds", "camera_transform_relative",
                "spawned_actors", "rgb_sha256"]
    missing = [k for k in required if k not in session or session[k] is None]
    record("A5", not missing, "tutti i campi R7 presenti" if not missing
           else "campi mancanti: %s" % missing)

    # A6 non si chiude in una sola esecuzione: si stampa l'hash e il confronto e'
    # fra due sessioni registrate con lo stesso seme.
    record("A6", None, "sha256 RGB = %s (confrontare con una seconda esecuzione"
                       " dello stesso scenario)" % session["rgb_sha256"])

    tracked = session["spawned_actors"]
    if not tracked:
        record("A7", None, "nessun attore tracciato in questo scenario")
    else:
        seen = dict((a["instance_id"], 0) for a in tracked)
        for name in counts["instance"]:
            inst = cv2.imread(os.path.join(out_dir, "instance", name), cv2.IMREAD_UNCHANGED)
            # (B<<8)|G, non (G<<8)|B come da doc ufficiale: vedi la nota in
            # spawn_one_actor(). BGR: B=indice 0 (byte alto), G=indice 1 (byte basso).
            ids = (inst[:, :, 0].astype(np.uint16) << 8) | inst[:, :, 1]
            present = set(np.unique(ids).tolist())
            for iid in seen:
                if iid in present:
                    seen[iid] += 1
        weak = [(a["boss_class"], a["instance_id"], seen[a["instance_id"]])
                for a in tracked if seen[a["instance_id"]] < A7_MIN_FRAMES]
        record("A7", not weak,
               "tutti i %d attori tracciati visibili in >= %d frame" % (len(tracked), A7_MIN_FRAMES)
               if not weak else "fuori campo o quasi: %s" % weak)

    return results


def print_acceptance(results):
    print("\n--- controlli di accettazione ---")
    failed = 0
    for name, ok, detail in results:
        if ok is None:
            status = "INFO"
        elif ok:
            status = "PASS"
        else:
            status = "FAIL"
            failed += 1
        print("%-4s %-4s %s" % (name, status, detail))
    if failed:
        print("\n%d controlli falliti: la sessione NON e' valida." % failed)
        if any(n == "A4" and ok is False for n, ok, _ in results):
            print("A4 fallito e' il segnale piu' forte: rivedere la decodifica della"
                  " depth (R3) o l'allineamento dei sensori (R2/R5) prima di rigirare.")
    else:
        print("\nTutti i controlli eseguibili superati.")
    return failed


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
    sys.exit(1 if print_acceptance(run_acceptance_checks(out)) else 0)
