# boss_classes.py
#
# Vocabolario delle classi B.O.S.S. e tabella di corrispondenza con i 29 tag
# semantici di CARLA. E' l'assunzione centrale della Fase A e il par. 5.8.4 del
# piano chiede esplicitamente che stia "nel repository e nel deliverable", con le
# classi CARLA scartate e quelle BOSS non copribili scritte e non nascoste: da qui
# le liste CARLA_TAGS_DISCARDED e BOSS_NOT_COVERED_BY_TAG in fondo al file.
#
# Fonte delle classi: Tabella 19 di OR4.1/OR 4.1 (v10).docx, meno le quattro voci
# rimosse su decisione del committente (marciapiede tattile, persiana/anta,
# transenna, cantiere/impalcatura). L'elenco della Tabella 19 e' dichiarato
# preliminare, "da consolidare con il partner Petrucciani": se cambia l'elenco
# cambia anche CARLA_TAG_TO_BOSS, ed e' l'unico punto da toccare.
#
# Nessun import: questo modulo e' dati, e viene letto sia da carla_capture.py (che
# valida le classi dichiarate negli scenari) sia da carla_export.py (che non
# importa affatto carla).

# Le due categorie che il report interviste Petrucciani indica come le sole degne
# di nota: il "vuoto superiore" (78% del campione, urto con ostacoli alti) e i
# "veicoli silenziosi" (92% del campione, impossibili da sentire nel caos urbano).
# Il campo priority non e' decorativo: carla_export.py ordina il riepilogo di
# copertura mettendo queste classi in testa, perche' sono quelle su cui la
# soglia di 100-200 istanze per classe va davvero verificata.
PRIORITY_SUSPENDED = "ostacoli_sospesi"
PRIORITY_SILENT = "mezzi_silenziosi"
PRIORITY_STANDARD = "standard"

MACRO_DYNAMIC = "ostacoli dinamici e veicoli"
MACRO_SIGNAGE = "segnaletica e infrastrutture"
MACRO_CRITICAL = "ostacoli critici per la mobilita' assistita"

BOSS_CLASSES = [
    {"id": 0,  "name": "pedone",                       "name_it": "Pedone",                             "macro": MACRO_DYNAMIC,  "priority": PRIORITY_STANDARD},
    {"id": 1,  "name": "automobile",                   "name_it": "Automobile",                         "macro": MACRO_DYNAMIC,  "priority": PRIORITY_STANDARD},
    {"id": 2,  "name": "bicicletta",                   "name_it": "Bicicletta",                         "macro": MACRO_DYNAMIC,  "priority": PRIORITY_SILENT},
    {"id": 3,  "name": "motociclo",                    "name_it": "Motociclo",                          "macro": MACRO_DYNAMIC,  "priority": PRIORITY_STANDARD},
    {"id": 4,  "name": "monopattino",                  "name_it": "Monopattino (incl. elettrico)",      "macro": MACRO_DYNAMIC,  "priority": PRIORITY_SILENT},
    {"id": 5,  "name": "veicolo_elettrico_silenzioso", "name_it": "Veicolo elettrico silenzioso",       "macro": MACRO_DYNAMIC,  "priority": PRIORITY_SILENT},
    {"id": 6,  "name": "semaforo",                     "name_it": "Semaforo",                           "macro": MACRO_SIGNAGE,  "priority": PRIORITY_STANDARD},
    {"id": 7,  "name": "attraversamento_pedonale",     "name_it": "Attraversamento pedonale",           "macro": MACRO_SIGNAGE,  "priority": PRIORITY_STANDARD},
    {"id": 8,  "name": "palo_della_luce",              "name_it": "Palo della luce",                    "macro": MACRO_SIGNAGE,  "priority": PRIORITY_STANDARD},
    {"id": 9,  "name": "crepa_buca",                   "name_it": "Crepa / buca sull'asfalto",          "macro": MACRO_CRITICAL, "priority": PRIORITY_STANDARD},
    {"id": 10, "name": "gradino",                      "name_it": "Gradino",                            "macro": MACRO_CRITICAL, "priority": PRIORITY_STANDARD},
    {"id": 11, "name": "ramo_sporgente",               "name_it": "Ramo sporgente",                     "macro": MACRO_CRITICAL, "priority": PRIORITY_SUSPENDED},
    {"id": 12, "name": "insegna_cartello_basso",       "name_it": "Insegna / cartello basso",           "macro": MACRO_CRITICAL, "priority": PRIORITY_SUSPENDED},
    {"id": 13, "name": "ostacolo_sospeso_generico",    "name_it": "Ostacolo sospeso generico (120-200 cm)", "macro": MACRO_CRITICAL, "priority": PRIORITY_SUSPENDED},
]

BOSS_CLASS_BY_NAME = dict((c["name"], c) for c in BOSS_CLASSES)

# Ordine in cui carla_export.py stampa il riepilogo di copertura.
PRIORITY_ORDER = [PRIORITY_SUSPENDED, PRIORITY_SILENT, PRIORITY_STANDARD]

# I 29 tag semantici di CARLA 0.9.16, indicizzati per id. Serve a stampare
# messaggi leggibili e a documentare cosa viene scartato: senza questa tabella un
# "tag 22 non mappato" non dice nulla a chi legge il log.
CARLA_TAG_NAMES = {
    0: "Unlabeled",   1: "Roads",        2: "SideWalks",   3: "Building",
    4: "Wall",        5: "Fence",        6: "Pole",        7: "TrafficLight",
    8: "TrafficSign", 9: "Vegetation",  10: "Terrain",    11: "Sky",
    12: "Pedestrian", 13: "Rider",      14: "Car",        15: "Truck",
    16: "Bus",        17: "Train",      18: "Motorcycle", 19: "Bicycle",
    20: "Static",     21: "Dynamic",    22: "Other",      23: "Water",
    24: "RoadLine",   25: "Ground",     26: "Bridge",     27: "RailTrack",
    28: "GuardRail",
}

TAG_SKY = 11
TAG_STATIC = 20
TAG_OTHER = 22

# Corrispondenza diretta tag CARLA -> classe BOSS. Copre 6 classi su 14: tutto il
# resto passa dall'override sugli attori spawnati (vedi sotto), che e' il
# meccanismo del par. 5.4 e non un ripiego.
CARLA_TAG_TO_BOSS = {
    6:  "palo_della_luce",   # il tag Pole copre ogni palo, inclusi i pali di cartello
    7:  "semaforo",
    12: "pedone",
    13: "pedone",            # un rider e' una persona; il veicolo che guida porta tag proprio
    14: "automobile",        # salvo override: un'elettrica ha lo stesso tag Car di un'auto qualsiasi
    18: "motociclo",
    19: "bicicletta",
}

# Tag CARLA deliberatamente scartati, con il motivo. Il par. 5.8.4 li vuole
# tracciati: sono decisioni, non dimenticanze.
CARLA_TAGS_DISCARDED = {
    8:  "TrafficSign: separabile da insegna_cartello_basso solo con un filtro di"
        " quota, non dal tag. Chi resta a quota alta non e' un ostacolo per il busto.",
    15: "Truck: nessuna classe BOSS corrispondente.",
    16: "Bus: nessuna classe BOSS corrispondente.",
    17: "Train: nessuna classe BOSS corrispondente.",
    24: "RoadLine: copre ogni segnaletica orizzontale, l'attraversamento pedonale"
        " non e' isolabile dal tag. Ricavabile invece da carla.Map.get_crosswalks().",
    2:  "SideWalks: e' superficie calpestabile, non un'istanza di ostacolo.",
}

# Classi BOSS senza tag CARLA. Le prime cinque sono coperte dall'override sugli
# attori spawnati (scenarios.py assegna la classe, carla_capture.py ne registra
# l'id in session.json, carla_export.py risolve per id prima che per tag). Le
# ultime tre no, e il motivo e' scritto qui perche' e' una limitazione della
# Fase A da riportare nel deliverable.
BOSS_NOT_COVERED_BY_TAG = {
    "veicolo_elettrico_silenzioso":
        "override: il tag Car non distingue un'elettrica. Coperta da silent_vehicles.",
    "gradino":
        "override: prop a scatola posato a filo marciapiede. Coperta da urban_day.",
    "ramo_sporgente":
        "override: prop allungato a quota 2,0-2,2 m. Coperta da overhead_obstacle.",
    "insegna_cartello_basso":
        "override: prop cartello a quota ~1,8 m. Coperta da overhead_obstacle.",
    "ostacolo_sospeso_generico":
        "override: prop sospeso nella fascia 120-200 cm. Coperta da overhead_obstacle.",
    "monopattino":
        "NON COPERTA. Il catalogo CARLA non contiene monopattini (par. 5.3.1) e il"
        " sostituto per silhouette e' proprio il caso che il par. 5.3.3 sconsiglia,"
        " perche' una sagoma sbagliata fa fallire lo IoU >= 0,70 della Fase D."
        " Resta al ramo reale: e' acquisibile in sicurezza.",
    "crepa_buca":
        "NON COPERTA. Una buca e' un volume negativo nel manto stradale: un prop"
        " aggiunge materia e nessun setter di scala permette di scavare. Un prop"
        " piatto avrebbe la depth di una strada, non di un incavo, cioe' una ground"
        " truth geometrica sbagliata. Resta al ramo reale.",
    "attraversamento_pedonale":
        "NON ANCORA COPERTA. Ricavabile da carla.Map.get_crosswalks() proiettando i"
        " poligoni con gli intrinseci e la posa gia' in session.json, ma serve anche"
        " un test di occlusione contro la depth. Si aggiunge a carla_export.py senza"
        " ri-registrare nulla.",
}
