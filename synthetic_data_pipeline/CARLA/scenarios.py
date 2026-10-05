# scenarios.py
#
# Registro degli scenari di acquisizione. Un dict per scenario, selezionato da
# carla_capture.py con --scenario NAME. Tenerli qui e non come costanti in cima
# allo script serve a una cosa sola: lo scenario finisce testualmente in
# session.json, quindi una sessione registrata dice da sola con quale
# configurazione e' nata (requisiti R7 e R8).
#
# I tre scenari NON coprono le classi in modo uniforme: sono costruiti sulle due
# priorita' del report interviste Petrucciani (ostacoli sospesi e mezzi
# silenziosi). urban_day e' la baseline che raccoglie le classi native da tag,
# gli altri due esistono per le priorita'.
#
# Nessun import di carla: i preset meteo sono stringhe risolte a runtime con
# getattr(carla.WeatherParameters, ...), cosi' questo file resta leggibile e
# importabile anche fuori dall'ambiente CARLA.

# --- posizionamento degli attori --------------------------------------------
#
# Gli attori NON hanno coordinate assolute di mappa. Le coordinate assolute di
# Town10HD non sono note a priori e cambierebbero cambiando mappa: uno scenario
# scritto su coordinate fisse e' uno scenario che smette di funzionare in
# silenzio, spawnando gli oggetti dentro un muro o a cento metri dal percorso.
#
# Ogni attore e' invece dichiarato in un sistema relativo alla posizione iniziale
# del pedone che porta la camera:
#   forward  metri lungo la direzione di marcia iniziale (spawn -> target)
#   lateral  metri a destra di quella direzione (negativo = sinistra)
#   z        metri sopra la quota del suolo nel punto di spawn
#   yaw      gradi di rotazione attorno all'asse verticale, rispetto alla marcia
#
# carla_capture.py converte questi valori in una carla.Transform assoluta. Il
# pedone non cammina in linea retta - il controller di navigazione segue la mesh
# - quindi valori di forward grandi diventano progressivamente meno affidabili:
# per questo gli ostacoli critici stanno entro ~20 m. L'arbitro finale e'
# comunque il controllo A7, che verifica sulle maschere di istanza che gli
# oggetti tracciati siano stati davvero inquadrati.

# --- catalogo dei blueprint usati -------------------------------------------
#
# Questi id sono presi dal catalogo documentato di CARLA 0.9.16, ma l'inventario
# effettivo dipende dall'installazione (par. 5.2 del piano). NON si assume che
# esistano: carla_capture.py li verifica tutti all'avvio contro la
# blueprint_library e, se uno manca, si ferma stampando i candidati simili
# trovati. Un id sbagliato diventa cosi' un errore immediato e leggibile invece
# di una scena silenziosamente diversa da quella progettata.

# --- chiavi delle dichiarazioni (dal 05/10) ----------------------------------
#
# Un ostacolo viene da "blueprint" (un id del catalogo) oppure da "mesh" (un
# percorso sotto /Game/Carla/Static/, spawnato con static.prop.mesh e "scale"
# uniforme): i percorsi validi stanno in CarlaUE4/AssetRegistry.bin, e
# scene_actors.validate_meshes() li prova tutti prima di registrare. Facoltativi:
#   pitch, roll       gradi, oltre allo yaw
#   anchor: "tree"    l'ostacolo parte dal tronco dell'albero vero piu' vicino al
#                     punto (forward, lateral); lo yaw lo ruota verso il pedone
#   support           un secondo attore (un palo) con offset forward/lateral
#                     relativi all'ostacolo e quota z dal suolo; non porta la
#                     classe dell'ostacolo
# "movers" dichiara mezzi in movimento (scene_actors.ApproachingRider); richiede
# place_at_start. carla_capture.validate_scenario rifiuta chiavi sconosciute.

_TOWN = "Town10HD_Opt"

# Punti di navigazione: indici in un pool di locazioni estratte in modo
# deterministico dalla nav mesh (vedi carla_capture.py). Non sono spawn point
# stradali - quelli metterebbero il pedone in mezzo alla carreggiata.
_SPAWN_A, _TARGET_A = 0, 7

# Comune alle scene Sim2Real: stessa mappa, meteo, seme e quindi percorso, stessa
# geometria di cattura. Ogni scena aggiunge i suoi attori (vedi sotto).
_S2R = {
    "map": _TOWN,
    "weather": "ClearNoon",
    "wind_intensity": 0.0,
    "seed": 2,
    "walker_bp_index": 1,
    "walker_spawn_index": _SPAWN_A,
    "walker_target_index": _TARGET_A,
    "traffic_vehicles": 10,
    "traffic_walkers": 5,
    "width": 1280,
    "height": 720,
    "fps": 48,
    "min_window_s": 5.9,
    "place_at_start": True,
}

SCENARIOS = {

    # Baseline. Raccoglie le 6 classi native da tag (pedone, automobile,
    # bicicletta, motociclo, semaforo, palo della luce) dal traffico e
    # dall'arredo urbano gia' presenti nella mappa, e aggiunge il gradino.
    "urban_day": {
        "map": _TOWN,
        "weather": "ClearNoon",
        "n_frames": 300,
        "seed": 0,
        "walker_bp_index": 1,
        "walker_spawn_index": _SPAWN_A,
        "walker_target_index": _TARGET_A,
        "traffic_vehicles": 40,
        "traffic_walkers": 30,
        "actors": [
            # Il gradino come forma geometrica semplice. Non serve un tag custom:
            # un prop spawnato e' un attore, quindi ha un instance id, quindi
            # l'override gli assegna la classe. Serve pero' una mesh gia' a
            # proporzioni di parallelepipedo basso, perche' l'API di CARLA non
            # espone alcun setter di scala e la forma non si puo' deformare.
            {"blueprint": "static.prop.box03", "forward": 9.0,  "lateral": 0.6, "z": 0.0, "yaw": 0.0,
             "boss_class": "gradino", "physics": False},
            {"blueprint": "static.prop.box03", "forward": 15.0, "lateral": -0.7, "z": 0.0, "yaw": 25.0,
             "boss_class": "gradino", "physics": False},
        ],
    },

    # Variante di illuminazione di urban_day: stessa mappa, stesso percorso,
    # stesso seme, cambia solo il meteo. Costa una voce di dict e aggiunge una
    # cella alla Matrice di Copertura, che e' incrociata classe x condizione.
    "urban_dusk": {
        "map": _TOWN,
        "weather": "WetSunset",
        "n_frames": 300,
        "seed": 0,
        "walker_bp_index": 1,
        "walker_spawn_index": _SPAWN_A,
        "walker_target_index": _TARGET_A,
        "traffic_vehicles": 40,
        "traffic_walkers": 30,
        "actors": [
            {"blueprint": "static.prop.box03", "forward": 9.0,  "lateral": 0.6, "z": 0.0, "yaw": 0.0,
             "boss_class": "gradino", "physics": False},
            {"blueprint": "static.prop.box03", "forward": 15.0, "lateral": -0.7, "z": 0.0, "yaw": 25.0,
             "boss_class": "gradino", "physics": False},
        ],
    },

    # Priorita' "mezzi silenziosi" (92% del campione intervistato).
    #
    # Le biciclette escono gia' corrette dal tag Bicycle. I veicoli elettrici no:
    # portano il tag Car esattamente come una qualsiasi utilitaria, e senza
    # override finirebbero contati come automobile. E' il caso che dimostra
    # perche' l'override sull'instance id vale piu' di un tag semantico custom.
    #
    # Gli attori sono fermi e con fisica disattivata, a distanze scalate da 5 a
    # 30 m: cosi' la sessione e' deterministica e la ground truth metrica copre
    # un intervallo di distanze utile alla valutazione MDE, invece di dipendere
    # da dove il traffic manager ha deciso di portarli.
    "silent_vehicles": {
        "map": _TOWN,
        "weather": "ClearNoon",
        "n_frames": 300,
        "seed": 1,
        "walker_bp_index": 1,
        "walker_spawn_index": _SPAWN_A,
        "walker_target_index": _TARGET_A,
        "traffic_vehicles": 10,
        "traffic_walkers": 20,
        "actors": [
            {"blueprint": "vehicle.diamondback.century", "forward": 5.0,  "lateral": 1.2,  "z": 0.0, "yaw": 90.0,
             "boss_class": "bicicletta", "physics": False},
            {"blueprint": "vehicle.gazelle.omafiets",    "forward": 12.0, "lateral": -1.4, "z": 0.0, "yaw": -75.0,
             "boss_class": "bicicletta", "physics": False},
            {"blueprint": "vehicle.bh.crossbike",        "forward": 22.0, "lateral": 1.0,  "z": 0.0, "yaw": 100.0,
             "boss_class": "bicicletta", "physics": False},
            {"blueprint": "vehicle.tesla.model3",        "forward": 8.0,  "lateral": 3.2,  "z": 0.0, "yaw": 0.0,
             "boss_class": "veicolo_elettrico_silenzioso", "physics": False},
            {"blueprint": "vehicle.audi.etron",          "forward": 18.0, "lateral": 3.4,  "z": 0.0, "yaw": 180.0,
             "boss_class": "veicolo_elettrico_silenzioso", "physics": False},
            {"blueprint": "vehicle.micro.microlino",     "forward": 30.0, "lateral": 3.0,  "z": 0.0, "yaw": 0.0,
             "boss_class": "veicolo_elettrico_silenzioso", "physics": False},
        ],
    },

    # Priorita' "ostacoli sospesi" (78% del campione: urto con rami, persiane,
    # insegne, perche' il bastone bianco copre il solo livello del suolo).
    # Copre la fascia 120-200 cm che il requisito di progetto indica.
    #
    # Visivamente sono oggetti che fluttuano, il che sembra assurdo, ma per il
    # par. 5.0 non lo e': al generativo serve la geometria di una massa
    # all'altezza della testa alla distanza giusta, e l'apparenza di ramo la
    # fornisce il prompt. E' precisamente il valore del ramo Sim2Real.
    #
    # I prop sono scelti per SILHOUETTE e non per nome (par. 5.3.3): la metrica
    # della Fase D confronta la maschera sorgente con quella del generato
    # (IoU >= 0,70), quindi per un ramo sporgente serve un oggetto allungato e
    # orizzontale, non un barile.
    "overhead_obstacle": {
        "map": _TOWN,
        "weather": "ClearNoon",
        "n_frames": 300,
        "seed": 2,
        "walker_bp_index": 1,
        "walker_spawn_index": _SPAWN_A,
        "walker_target_index": _TARGET_A,
        "traffic_vehicles": 15,
        "traffic_walkers": 20,
        "actors": [
            # Allungato e orizzontale, trasversale alla marcia: sagoma di ramo.
            {"blueprint": "static.prop.streetbarrier", "forward": 7.0,  "lateral": 0.0, "z": 2.10, "yaw": 90.0,
             "boss_class": "ramo_sporgente", "physics": False},
            {"blueprint": "static.prop.streetbarrier", "forward": 16.0, "lateral": 0.5, "z": 2.00, "yaw": 60.0,
             "boss_class": "ramo_sporgente", "physics": False},
            # Pannello piatto verticale a quota busto/testa: sagoma di insegna.
            {"blueprint": "static.prop.warningconstruction", "forward": 11.0, "lateral": -0.8, "z": 1.80, "yaw": 0.0,
             "boss_class": "insegna_cartello_basso", "physics": False},
            # Massa compatta a meta' della fascia 120-200 cm, senza pretesa di
            # somigliare a un oggetto reale: e' la classe generica.
            {"blueprint": "static.prop.box02", "forward": 20.0, "lateral": 0.3, "z": 1.60, "yaw": 30.0,
             "boss_class": "ostacolo_sospeso_generico", "physics": False},
        ],
    },

    # --- ramo Sim2Real ---------------------------------------------------------
    #
    # Sorgente per i generatori video-to-video (Cosmos-Transfer2.5, Wan2.2-Fun
    # Control): CARLA da' la scena esatta, il modello ne ridipinge l'aspetto.
    #
    # Dal 05/10 le scene rappresentano situazioni vere, non prop sospesi nel vuoto:
    # un ramo esce dal tronco di un albero del viale, un cartello sta sul suo palo,
    # una cassetta e' montata su un palo, il monopattino ha conducente, asta, pedana
    # e ruote. Tutte progettate e guardate a vista sul percorso del seme 2, un viale
    # dritto con alberi veri a ~2 m a sinistra della linea di marcia e facciate a
    # ~4 m a destra (tronchi a x -25,8 / -16,3 / -3,4 / 8,1 ...).
    #
    #   - 1280x720 a 48 Hz: 16:9 nativo, perche' ritagliare un 4:3 taglierebbe la
    #     fascia alta, dove stanno i sospesi; 48 Hz e' il minimo comune multiplo dei
    #     16 fps di Cosmos e dei 24 di Wan, che lo ottengono prendendo un frame su
    #     tre e uno su due, senza interpolare.
    #   - place_at_start: tutto si piazza all'ultimo tick di warmup, a "forward"
    #     metri dal pedone. Gli ostacoli ci sono dal frame 0 (niente pop-in) e la
    #     distanza dichiarata e' quella del frame 0. Il primo albero oltre i 10 m e'
    #     a ~15 m, quindi il ramo resta in vista ~10 s camminando a 1,4 m/s.
    #   - Vento a zero: le foglie non tremano fra un frame e l'altro e la sessione
    #     si ripete identica (col vento due catture uguali differivano sul 2,5-6 %
    #     della depth, revisione 05/10).
    #   - Stesso seme (2) e quindi stesso percorso di overhead_obstacle per tutte:
    #     fra le clip cambia l'ostacolo. Attenzione se qualcosa viene addestrato su
    #     queste e misurato su eval_set, che contiene overhead_obstacle.
    #   - Pochi pedoni NPC: uno che passa davanti all'ostacolo lo copre per qualche
    #     frame (A8 tollera buchi fino a 0,25 s).
    #
    # Le quote stanno nella fascia testa/busto (criterio dell'utente del 28/09), per
    # costruzione: e' il difetto che il text-to-video non riusciva a evitare (A14B
    # mette i sospesi ad altezza piedi). La camera e' a 1,5 m dal suolo.

    # Un ramo che esce dal tronco a 2,1 m e attraversa il marciapiede scendendo di
    # 10 gradi: la parte legnosa resta sopra la testa, le foglie occupano 1,4-2,3 m
    # proprio sulla linea di marcia. E' un acero giovane (SM_Maple_S_v1) in scala
    # 0,6 e coricato: il suo fusto fa da ramo e la sua chioma da fronda terminale.
    # pitch -100 punta il fusto lungo lo yaw, 10 gradi sotto l'orizzontale; yaw +20
    # lo ruota verso il pedone che arriva.
    "s2r_ramo": dict(_S2R, n_frames=576, actors=[
        {"mesh": "Vegetation/Trees/SM_Maple_S_v1.SM_Maple_S_v1", "scale": 0.6,
         "anchor": "tree", "forward": 14.0, "lateral": -2.0, "z": 2.10,
         "pitch": -100.0, "yaw": 20.0, "boss_class": "ramo_sporgente", "physics": False},
    ]),

    # La variante pesante: una fronda di quercia che dallo stesso tronco ricade nel
    # marciapiede fino all'altezza del petto. A vista sembra un ramo basso dello
    # stesso albero.
    "s2r_ramo_fronda": dict(_S2R, n_frames=576, actors=[
        {"mesh": "Vegetation/Trees/SM_Oak_S_v1.SM_Oak_S_v1", "scale": 0.42,
         "anchor": "tree", "forward": 14.0, "lateral": -2.0, "z": 2.30,
         "pitch": -100.0, "yaw": 20.0, "boss_class": "ramo_sporgente", "physics": False},
    ]),

    # Un cartello di sosta montato di lato sul suo palo, al bordo del marciapiede,
    # con il bordo basso a 1,60 m: 10 cm sopra gli occhi, all'altezza della fronte.
    # Il bastone trova il palo, non la lamiera. Palo e cartello sono attori
    # distinti: la box dell'insegna e' quella del solo cartello (il palo resta Pole,
    # cioe' palo_della_luce per l'export). La lamiera (0,52 x 0,76 m, origine al
    # centro) occupa da -0,56 a -0,04 m a sinistra della linea di marcia.
    "s2r_insegna": dict(_S2R, n_frames=480, actors=[
        {"mesh": "TrafficSign/TrafficSigns_A1/SM_A01parking.SM_A01parking",
         "forward": 11.0, "lateral": -0.30, "z": 1.98, "yaw": 90.0,
         "support": {"mesh": "Pole/SM_RoadSigns01.SM_RoadSigns01",
                     "forward": 0.0, "lateral": -0.30, "z": 0.0},
         "boss_class": "insegna_cartello_basso", "physics": False},
    ]),

    # Una cassetta (quadro tecnico, cassetta postale) montata su un palo a
    # 1,24-1,76 m, che sporge verso la linea di marcia fino a 0,24 m dal centro del
    # pedone: all'altezza della spalla. Il palo (RoadSigns01) e' interrato di 0,55 m
    # per finire dentro la cassetta.
    "s2r_sospeso_generico": dict(_S2R, n_frames=480, actors=[
        {"mesh": "Building/Building_pieces/props/AirConditioner/"
                 "SM_Prop_001_AirConditioner_001.SM_Prop_001_AirConditioner_001",
         "forward": 12.0, "lateral": -0.60, "z": 1.50, "yaw": 90.0,
         "support": {"mesh": "Pole/SM_RoadSigns01.SM_RoadSigns01",
                     "forward": 0.0, "lateral": -0.05, "z": -0.55},
         "boss_class": "ostacolo_sospeso_generico", "physics": False},
    ]),

    # Priorita' "mezzi silenziosi": un monopattino elettrico a 25 km/h che arriva
    # incontro al pedone SULLA SUA LINEA DI MARCIA, e schiva all'ultimo passandogli a
    # 0,9 m sulla destra (lato facciate, il piu' libero). Parte a 60 m: con 8,3 m/s
    # di avvicinamento resta in vista ~7 s, piu' dei 5,9 s di A8, e a 60 m e' ancora
    # ~22 px di altezza. Nessun pedone NPC: starebbero proprio sulla linea di vista.
    #
    # Il catalogo di CARLA non ha monopattini: e' composto (scene_actors.ApproachingRider).
    # Conducente in piedi e fermo, che trasla senza camminare; asta e manubrio da un
    # palo di cartello in scala; pedana da un cartone piatto; ruote da due piatti da
    # 18 cm messi di taglio. dx avanti nel verso di marcia, dy a destra, z dal suolo.
    "s2r_monopattino": dict(_S2R, n_frames=480, traffic_walkers=0, actors=[], movers=[
        {"kind": "approaching_rider", "boss_class": "monopattino",
         "rider": "walker.pedestrian.0026",
         "forward": 60.0, "lateral": 0.0, "speed_kmh": 25.0,
         "pass_lateral": 0.9, "swerve_from_m": 9.0, "swerve_to_m": 3.0,
         "deck_top_m": 0.135,
         "parts": [
             {"part": "pedana", "mesh": "Dynamic/Trash/SM_CreasedBox02.SM_CreasedBox02",
              "scale": 0.33, "dx": 0.0, "dy": 0.0, "z": 0.065},
             {"part": "asta", "mesh": "Pole/SM_RoadSigns01.SM_RoadSigns01",
              "scale": 0.5, "dx": 0.35, "dy": 0.0, "z": 0.10, "pitch": 5.0},
             {"part": "manubrio", "mesh": "Pole/SM_RoadSigns01.SM_RoadSigns01",
              "scale": 0.22, "dx": 0.25, "dy": -0.26, "z": 1.20, "roll": 90.0},
             {"part": "ruota_anteriore", "mesh": "Dynamic/Bar-Restaurant/SM_Plate.SM_Plate",
              "scale": 1.1, "dx": 0.36, "dy": 0.0, "z": 0.10, "roll": 90.0},
             {"part": "ruota_posteriore", "mesh": "Dynamic/Bar-Restaurant/SM_Plate.SM_Plate",
              "scale": 1.1, "dx": -0.30, "dy": 0.0, "z": 0.10, "roll": 90.0},
         ]},
    ]),
}
