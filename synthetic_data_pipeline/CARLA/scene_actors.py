# scene_actors.py
#
# Costruzione degli attori degli scenari: prop del catalogo o mesh del pacchetto
# (static.prop.mesh), sostegni, ancoraggio ad alberi veri della mappa e il mezzo che
# arriva incontro al pedone. carla_capture.py decide QUANDO piazzarli; qui si
# decide COME, e si prepara cio' che finisce in session.json.
#
# Tre fatti di CARLA 0.9.16 su cui poggia il modulo, tutti misurati il 05/10 su
# questa installazione (un'ipotesi sbagliata qui non solleva eccezioni, cambia la
# scena):
#   - static.prop.mesh accetta qualunque mesh del pacchetto (mesh_path) e una
#     scala uniforme. Il tag semantico viene dalla cartella della mesh: Vegetation
#     per un albero, Pole per un palo, TrafficSign per un cartello. Ogni attore ha
#     il proprio id di istanza, quindi un cartello e il suo palo restano separabili.
#   - Un percorso di mesh sbagliato NON fa fallire lo spawn: nasce un attore vuoto,
#     con bounding box infinita. Per questo validate_meshes() le prova tutte prima
#     di registrare.
#   - Una static.prop.mesh senza massa e' immobile: ignora set_transform e
#     attach_to. Con massa > 0 si muove, ma set_transform riporta la scala a 1;
#     set_location no. I pezzi del monopattino si muovono quindi con set_location,
#     con la rotazione fissata allo spawn (il mezzo va dritto).
import math
import sys

import carla

import scene_geometry as geo
from boss_classes import BOSS_CLASS_BY_NAME

MESH_ROOT = "/Game/Carla/Static/"

# Un albero e' un oggetto Vegetation di almeno questa altezza: esclude siepi,
# cespugli e ciuffi d'erba, che nella mappa sono anch'essi Vegetation.
TREE_MIN_HEIGHT_M = 6.0

# Tolleranza fra il punto in cui lo scenario si aspetta un tronco e il tronco vero.
# Nel viale del seme 2 gli alberi distano 9-13 m: 3 m bastano a non sbagliare albero.
ANCHOR_RADIUS_M = 3.0

# La base del ramo sta dentro il tronco, spostata di poco verso il percorso: cosi'
# il ramo esce dalla corteccia invece di fluttuare accanto.
ANCHOR_TRUNK_OFFSET_M = 0.15

# Massa con cui una static.prop.mesh diventa mobile (vedi sopra). Il valore non
# conta: la fisica viene spenta subito dopo lo spawn.
MOVABLE_MASS_KG = 1.0

# Validazione delle mesh: spawn lontano da tutto, in aria.
PROBE_LOCATION = (0.0, 0.0, 800.0)


def transform_to_dict(tf):
    return {
        "location": {"x": tf.location.x, "y": tf.location.y, "z": tf.location.z},
        "rotation": {"pitch": tf.rotation.pitch, "yaw": tf.rotation.yaw, "roll": tf.rotation.roll},
    }


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


def spec_blueprint(library, spec, movable=False):
    """Blueprint di un attore dichiarato: un id del catalogo, oppure una mesh del pacchetto."""
    if "mesh" not in spec:
        return resolve_blueprint(library, spec["blueprint"])
    bp = library.find("static.prop.mesh")
    bp.set_attribute("mesh_path", MESH_ROOT + spec["mesh"])
    bp.set_attribute("scale", str(spec.get("scale", 1.0)))
    if movable:
        bp.set_attribute("mass", str(MOVABLE_MASS_KG))
    return bp


def spec_label(spec):
    """Come l'attore e' scritto in session.json: id del catalogo o percorso della mesh."""
    return spec["mesh"] if "mesh" in spec else spec["blueprint"]


def mesh_specs(scenario):
    """Tutte le dichiarazioni dello scenario che usano una mesh: ostacoli, sostegni, pezzi."""
    out = []
    for spec in scenario["actors"]:
        out.append(spec)
        if "support" in spec:
            out.append(spec["support"])
    for mover in scenario.get("movers", []):
        out.extend(mover["parts"])
    return [s for s in out if "mesh" in s]


def validate_meshes(world, library, scenario):
    """Prova ogni mesh dello scenario prima di registrare.

    Un mesh_path inesistente non fa fallire lo spawn: produce un attore senza
    geometria, con bounding box infinita (misurato su SM_Trafficpole_2, il cui
    oggetto interno si chiama SM_TrafficPole). La sessione verrebbe registrata con
    un ostacolo invisibile e A7 se ne accorgerebbe solo alla fine.
    """
    x, y, z = PROBE_LOCATION
    for i, spec in enumerate(mesh_specs(scenario)):
        bp = spec_blueprint(library, spec)
        actor = world.try_spawn_actor(bp, carla.Transform(carla.Location(x=x + 20.0 * i, y=y, z=z)))
        if actor is None:
            sys.exit("mesh non spawnabile: %s" % spec["mesh"])
        ext = actor.bounding_box.extent
        actor.destroy()
        dims = (ext.x, ext.y, ext.z)
        if not all(math.isfinite(v) and v > 0.0 for v in dims):
            sys.exit("mesh assente dal pacchetto o senza geometria: %s (extent %s)."
                     " Il percorso e' /Game/Carla/Static/<cartella>/<File>.<Oggetto>, e il"
                     " nome dell'oggetto puo' differire da quello del file."
                     % (spec["mesh"], dims))


def tree_trunks(world):
    """Posizione (x, y) dei tronchi degli alberi della mappa.

    La location dell'oggetto e' la base del tronco, non il centro della chioma: la
    bounding box di un albero e' spostata dal lato in cui pende la chioma (sul viale
    del seme 2 fino a 1,2 m), mentre la location coincide col tronco visibile.
    """
    trunks = []
    for obj in world.get_environment_objects(carla.CityObjectLabel.Vegetation):
        if 2.0 * obj.bounding_box.extent.z >= TREE_MIN_HEIGHT_M:
            loc = obj.transform.location
            trunks.append((loc.x, loc.y))
    return trunks


def spawn_static(world, library, spec, location, rotation):
    """Spawna un attore fermo con la fisica spenta, oppure None (collisione)."""
    actor = world.try_spawn_actor(spec_blueprint(library, spec), carla.Transform(location, rotation))
    if actor is not None and not spec.get("physics", False):
        # R6: senza questo gli ostacoli sospesi cadono al primo tick.
        actor.set_simulate_physics(False)
    return actor


def actor_entry(actor, spec, boss_class, transform):
    """La riga di spawned_actors per un attore che porta una classe BOSS."""
    return {
        "actor_id": actor.id,
        # L'instance segmentation codifica l'identita' su 16 bit fra i canali G e B
        # (R e' il tag semantico). La doc ufficiale descrive (G<<8)|B, ma la verifica
        # empirica su questa installazione (0.9.16) mostra (B<<8)|G, canale B byte
        # alto: e' la formula di acceptance.py, carla_export.py e carla_gt.py.
        "instance_id": actor.id & 0xFFFF,
        "blueprint": actor.type_id,
        "mesh": spec.get("mesh"),
        "scale": spec.get("scale") if "mesh" in spec else None,
        "boss_class": boss_class,
        "boss_class_id": BOSS_CLASS_BY_NAME[boss_class]["id"],
        "transform": transform_to_dict(transform),
        "_actor": actor,
    }


def place_obstacle(world, library, spec, frame, ahead, trunks):
    """Piazza un ostacolo dichiarato (con l'eventuale sostegno) davanti al pedone.

    frame: dict con origin (x, y), ground_z, fwd (versore di marcia) e yaw, presi
    nell'istante del piazzamento. ahead: metri lungo fwd a cui sta l'ostacolo.
    Restituisce (entry, attori creati, motivo del fallimento): entry e' None se il
    piazzamento e' fallito, e in quel caso nessun attore resta nel mondo.
    """
    origin, fwd = frame["origin"], frame["fwd"]
    anchor_info = None
    if spec.get("anchor") == "tree":
        search = geo.frame_point(origin, fwd, ahead, spec["lateral"])
        trunk, dist = geo.nearest_within(trunks, search, ANCHOR_RADIUS_M)
        if trunk is None:
            return None, [], ("nessun albero entro %.1f m dal punto atteso (%.2f, %.2f);"
                              " il piu' vicino a %.1f m" % (ANCHOR_RADIUS_M, search[0], search[1], dist))
        across, yaw, side = geo.across_from_anchor(trunk, origin, fwd, spec.get("yaw", 0.0))
        xy = (trunk[0] + across[0] * ANCHOR_TRUNK_OFFSET_M, trunk[1] + across[1] * ANCHOR_TRUNK_OFFSET_M)
        anchor_info = {"type": "tree", "trunk": {"x": trunk[0], "y": trunk[1]},
                       "distance_from_expected_m": dist,
                       "side": "sinistra" if side < 0 else "destra"}
    else:
        xy = geo.frame_point(origin, fwd, ahead, spec["lateral"])
        yaw = frame["yaw"] + spec["yaw"]

    location = carla.Location(x=xy[0], y=xy[1], z=frame["ground_z"] + spec["z"])
    rotation = carla.Rotation(pitch=spec.get("pitch", 0.0), yaw=yaw, roll=spec.get("roll", 0.0))
    actor = spawn_static(world, library, spec, location, rotation)
    if actor is None:
        return None, [], "spawn fallito per %s: collisione con la geometria della mappa" % spec_label(spec)
    entry = actor_entry(actor, spec, spec["boss_class"], carla.Transform(location, rotation))
    if anchor_info is not None:
        entry["anchor"] = anchor_info

    created = [actor]
    support = spec.get("support")
    if support is not None:
        sxy = geo.frame_point(xy, fwd, support["forward"], support["lateral"])
        s_loc = carla.Location(x=sxy[0], y=sxy[1], z=frame["ground_z"] + support["z"])
        s_rot = carla.Rotation(yaw=frame["yaw"] + support.get("yaw", 0.0))
        s_actor = spawn_static(world, library, support, s_loc, s_rot)
        if s_actor is None:
            actor.destroy()
            return None, [], "spawn fallito per il sostegno %s" % spec_label(support)
        created.append(s_actor)
        # Il sostegno non porta la classe dell'ostacolo: per l'export conta il suo tag
        # (un palo e' Pole, quindi palo_della_luce), e la box dell'insegna resta
        # quella del cartello e non arriva a terra.
        entry["support"] = {"actor_id": s_actor.id, "instance_id": s_actor.id & 0xFFFF,
                            "mesh": support.get("mesh"), "blueprint": s_actor.type_id,
                            "transform": transform_to_dict(carla.Transform(s_loc, s_rot))}
    return entry, created, None


def walker_origin_above_feet(walker):
    """Quanto l'origine dell'attore pedone sta sopra i suoi stessi piedi, in metri.

    L'origine non e' ai piedi ma al centro della capsula fisica (misurato: con un
    offset relativo fisso la camera finiva a ~2,6 m dal suolo invece dei 1,5 m di
    torace/testa richiesti dal par. 5.6, e un prop piazzato a walker.z + 0 finiva
    a ~1,1 m di altezza invece che a terra). Si ricava dalla bounding box
    dell'attore, generica per qualunque blueprint pedone: la quota dei piedi
    rispetto all'origine e' bb.location.z - bb.extent.z, quindi l'origine sta
    sopra i piedi di bb.extent.z - bb.location.z. Va sottratta ovunque nel codice
    si usi walker.get_transform().location.z come se fosse la quota del suolo, e
    aggiunta quando si posa un pedone (il conducente del monopattino) su una quota.
    """
    bb = walker.bounding_box
    return bb.extent.z - bb.location.z


class ApproachingRider:
    """Un mezzo con conducente che arriva incontro al pedone, lungo la sua linea di marcia.

    E' il monopattino dello scenario s2r_monopattino. Il catalogo di CARLA non ne
    contiene (boss_classes.py), quindi e' composto: un pedone in piedi e immobile,
    che trasla senza camminare come chi sta su una pedana, piu' i pezzi del mezzo
    (asta, manubrio, pedana, ruote) presi da mesh del pacchetto. Tutti gli attori
    portano la stessa classe e lo stesso `group`: per A7/A8 e per carla_export sono
    un oggetto solo, con la box sull'unione delle maschere.

    Il moto e' cinematico e dichiarato, non simulato: velocita' costante lungo la
    direzione opposta a quella del pedone nell'istante dello spawn, e una schivata
    laterale negli ultimi metri (scene_geometry.swerve_lateral). Cosi' la velocita'
    e' esattamente quella dello scenario e la sessione resta deterministica.
    """

    def __init__(self, spec, frame, fixed_delta, group):
        self.spec = spec
        self.group = group
        self.origin = frame["origin"]
        self.fwd = frame["fwd"]
        self.ground_z = frame["ground_z"]
        self.heading = (-self.fwd[0], -self.fwd[1])
        self.yaw = geo.yaw_of(self.heading)
        self.speed = spec["speed_kmh"] / 3.6
        self.dt = fixed_delta
        self.rider = None
        self.rider_above = 0.0
        self.parts = []
        self.entries = []

    def _state(self, elapsed_ticks, walker_xy):
        along = self.spec["forward"] - self.speed * elapsed_ticks * self.dt
        gap = along - geo.along_of(walker_xy, self.origin, self.fwd)
        lateral = geo.swerve_lateral(gap, self.spec["lateral"], self.spec["pass_lateral"],
                                     self.spec["swerve_from_m"], self.spec["swerve_to_m"])
        return geo.frame_point(self.origin, self.fwd, along, lateral), gap

    def _rider_location(self, base):
        z = self.ground_z + self.spec["deck_top_m"] + self.rider_above
        return carla.Location(x=base[0], y=base[1], z=z)

    def _part_location(self, base, part):
        p = geo.part_point(base, self.heading, part["dx"], part["dy"])
        return carla.Location(x=p[0], y=p[1], z=self.ground_z + part["z"])

    def spawn(self, world, library, walker_xy):
        """Crea conducente e pezzi alla posizione iniziale. Restituisce il motivo di un fallimento."""
        base, _ = self._state(0, walker_xy)
        rider_bp = resolve_blueprint(library, self.spec["rider"])
        rotation = carla.Rotation(yaw=self.yaw)
        # L'altezza dell'origine sopra i piedi si conosce solo a pedone creato: lo
        # spawn avviene un po' piu' in alto e la posa giusta arriva subito dopo.
        for lift in (1.3, 1.8):
            self.rider = world.try_spawn_actor(rider_bp, carla.Transform(
                carla.Location(x=base[0], y=base[1], z=self.ground_z + lift), rotation))
            if self.rider is not None:
                break
        if self.rider is None:
            return "spawn fallito per il conducente %s a %.1f m: posizione occupata" % (
                self.spec["rider"], self.spec["forward"])
        # Senza fisica il pedone non cade e non cammina: resta nella posa in piedi
        # e si sposta solo per teletrasporto, che e' la postura di chi sta su una pedana.
        self.rider.set_simulate_physics(False)
        self.rider_above = walker_origin_above_feet(self.rider)
        rider_tf = carla.Transform(self._rider_location(base), rotation)
        self.rider.set_transform(rider_tf)
        self.entries.append(self._entry(self.rider, {"blueprint": self.spec["rider"]}, "conducente", rider_tf))

        for part in self.spec["parts"]:
            tf = carla.Transform(self._part_location(base, part),
                                 carla.Rotation(pitch=part.get("pitch", 0.0), yaw=self.yaw + part.get("yaw", 0.0),
                                                roll=part.get("roll", 0.0)))
            actor = world.try_spawn_actor(spec_blueprint(library, part, movable=True), tf)
            if actor is None:
                return "spawn fallito per il pezzo %s (%s)" % (part["part"], part["mesh"])
            actor.set_simulate_physics(False)
            self.parts.append((part, actor))
            self.entries.append(self._entry(actor, part, part["part"], tf))
        return None

    def _entry(self, actor, spec, part_name, tf):
        entry = actor_entry(actor, spec, self.spec["boss_class"], tf)
        entry["group"] = self.group
        entry["part"] = part_name
        return entry

    def update(self, elapsed_ticks, walker_xy):
        """Comanda la posa che il prossimo tick rendera'. Restituisce la distanza dal pedone."""
        base, gap = self._state(elapsed_ticks, walker_xy)
        self.rider.set_transform(carla.Transform(self._rider_location(base), carla.Rotation(yaw=self.yaw)))
        for part, actor in self.parts:
            actor.set_location(self._part_location(base, part))
        return gap

    def actors(self):
        out = [] if self.rider is None else [self.rider]
        return out + [a for _, a in self.parts]

    def describe(self):
        """Il moto dichiarato, per session.json: la GT della velocita' sta qui."""
        return {
            "group": self.group,
            "boss_class": self.spec["boss_class"],
            "speed_mps": self.speed,
            "speed_kmh": self.spec["speed_kmh"],
            "heading_yaw_deg": self.yaw,
            "start_ahead_m": self.spec["forward"],
            "lateral_m": self.spec["lateral"],
            "pass_lateral_m": self.spec["pass_lateral"],
            "swerve_from_m": self.spec["swerve_from_m"],
            "swerve_to_m": self.spec["swerve_to_m"],
            "frame_origin": {"x": self.origin[0], "y": self.origin[1], "ground_z": self.ground_z},
            "frame_fwd": {"x": self.fwd[0], "y": self.fwd[1]},
        }
