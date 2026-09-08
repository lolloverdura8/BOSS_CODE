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

_TOWN = "Town10HD_Opt"

# Punti di navigazione: indici in un pool di locazioni estratte in modo
# deterministico dalla nav mesh (vedi carla_capture.py). Non sono spawn point
# stradali - quelli metterebbero il pedone in mezzo alla carreggiata.
_SPAWN_A, _TARGET_A = 0, 7

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
}
