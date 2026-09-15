# prompts.py
#
# I prompt di generazione, estratti alla lettera da wan2_2/generate_test_clips.py
# e non ridigitati: un carattere diverso cambierebbe le clip, e monopattino_00
# (seed 0, 569,1 s) e' il riferimento con cui si verifica l'adapter wan22_5b.
#
# Stanno qui e non dentro un adapter perche' sono la variabile CONTROLLATA del
# bake-off: tutti i modelli ricevono lo stesso testo. Se un modello lo capisce
# peggio, e' un suo risultato, non un handicap da compensare riscrivendogli il
# prompt addosso.
#
# Il negative prompt invece e' LIBERO per modello: e' quello ufficiale di Wan e
# vale per gli adapter Wan. CogVideoX non ne usa nessuno, HunyuanVideo e LTX
# hanno i propri. Ogni adapter dichiara il suo nella SPEC, e il runner lo scrive
# nel manifest, cosi' la differenza resta leggibile invece di sparire.

# 5 prompt per classe, una scena per prompt. Vincoli comuni: POV da pedone su
# marciapiede, UN solo oggetto bersaglio davanti a pochi metri, ben visibile. Per i
# sospesi il SUPPORTO deve stare in campo (tronco, palo), perche' e' il riferimento
# del controllo di coerenza di DA3, e l'oggetto si sviluppa di traverso al percorso
# e non verso la camera, cosi' oggetto e supporto stanno alla stessa profondita'.
PROMPTS = {
    "monopattino": [
        "POV shot from a person walking on a city sidewalk, steady forward motion, an electric "
        "kick scooter parked upright on its kickstand in the middle of the sidewalk about three "
        "meters ahead, clearly visible, sunny afternoon, sharp focus, realistic urban environment",
        "POV shot from a person walking on a city sidewalk, steady forward motion, an electric "
        "kick scooter lying on its side across the sidewalk about four meters ahead, overcast "
        "daylight, sharp focus, realistic urban environment",
        "POV shot from a person walking on a city sidewalk, steady forward motion, a rental "
        "electric kick scooter parked against a building wall on the right about three meters "
        "ahead, soft early morning light, sharp focus, realistic urban environment",
        "POV shot from a person walking on a city sidewalk, steady forward motion, a person riding "
        "an electric kick scooter towards the camera on the sidewalk about five meters ahead, "
        "daylight, sharp focus, realistic urban environment",
        "POV shot from a person walking on a city sidewalk, steady forward motion, an electric "
        "kick scooter parked at the edge of the sidewalk next to a pedestrian crossing about three "
        "meters ahead, wet pavement after rain, cloudy sky, sharp focus, realistic urban environment",
    ],
    "ramo_sporgente": [
        "POV shot from a person walking on a tree-lined sidewalk, steady forward motion, a low "
        "tree branch sticking out sideways from a tree trunk across the sidewalk at head height "
        "about three meters ahead, the trunk clearly visible on the right, sunny day, sharp focus, "
        "realistic urban environment",
        "POV shot from a person walking on a park path, steady forward motion, a thick tree branch "
        "growing sideways from a trunk on the left and crossing the path at head height about four "
        "meters ahead, overcast daylight, sharp focus, realistic environment",
        "POV shot from a person walking on a residential sidewalk, steady forward motion, a leafy "
        "branch extending sideways from a tree trunk next to a garden wall and crossing the "
        "sidewalk at head height about three meters ahead, late afternoon light, sharp focus, "
        "realistic urban environment",
        "POV shot from a person walking on a city sidewalk in winter, steady forward motion, a "
        "bare leafless branch extending sideways from a tree trunk on the right across the "
        "sidewalk at head height about three meters ahead, cloudy sky, sharp focus, realistic "
        "urban environment",
        "POV shot from a person walking on a shaded sidewalk, steady forward motion, a low "
        "branch with green leaves extending sideways from a tree trunk on the left across the "
        "sidewalk at head height about four meters ahead, dappled sunlight, sharp focus, "
        "realistic urban environment",
    ],
    "insegna_cartello_basso": [
        "POV shot from a person walking on a city sidewalk, steady forward motion, a rectangular "
        "shop sign hanging from a metal pole at the edge of the sidewalk, the sign panel at head "
        "height about three meters ahead, sunny afternoon, sharp focus, realistic urban environment",
        "POV shot from a person walking on a city sidewalk, steady forward motion, a traffic sign "
        "mounted low on a metal pole on the sidewalk, its panel at head height about three meters "
        "ahead, overcast daylight, sharp focus, realistic urban environment",
        "POV shot from a person walking on a narrow old town sidewalk, steady forward motion, a "
        "small cafe sign hanging from a short pole bracket at head height about three meters "
        "ahead, soft morning light, sharp focus, realistic urban environment",
        "POV shot from a person walking on a city sidewalk, steady forward motion, a temporary "
        "road works sign fixed to a metal pole at head height on the sidewalk about four meters "
        "ahead, daylight, sharp focus, realistic urban environment",
        "POV shot from a person walking on a city sidewalk, steady forward motion, a street name "
        "sign mounted low on a pole at head height about three meters ahead, warm evening light, "
        "sharp focus, realistic urban environment",
    ],
    "ostacolo_sospeso_generico": [
        "POV shot from a person walking on a city sidewalk, steady forward motion, a horizontal "
        "metal bar fixed between two poles across the sidewalk at head height about three meters "
        "ahead, sunny day, sharp focus, realistic urban environment",
        "POV shot from a person walking on a city sidewalk, steady forward motion, a horizontal "
        "metal pipe supported by two vertical posts crossing the sidewalk at head height about "
        "four meters ahead, overcast daylight, sharp focus, realistic urban environment",
        "POV shot from a person walking on a city sidewalk, steady forward motion, the low "
        "horizontal metal beam of an awning frame supported by two poles crossing the sidewalk at "
        "head height about three meters ahead, afternoon light, sharp focus, realistic urban "
        "environment",
        "POV shot from a person walking on a pedestrian passage, steady forward motion, a "
        "horizontal wooden beam resting on two wooden posts across the passage at head height "
        "about three meters ahead, soft daylight, sharp focus, realistic environment",
        "POV shot from a person walking on a city sidewalk, steady forward motion, a horizontal "
        "barrier bar mounted between two poles across the sidewalk at head height about four "
        "meters ahead, cloudy sky, sharp focus, realistic urban environment",
    ],
}

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


# I due indici per classe usati dal bake-off: 8 prompt in tutto, 2 per classe.
# Scelti 0 e 2 perche' nelle quattro classi sono le due scene piu' diverse fra
# loro per luce e posizione dell'oggetto, e non i due casi adiacenti.
BAKEOFF_PROMPT_INDICES = (0, 2)

# Ordine stabile delle classi nei report: prima il monopattino (mezzi
# silenziosi), poi i tre sospesi (vuoto superiore). Sono le due categorie che il
# report interviste Petrucciani indica come prioritarie.
SUSPENDED = ("ramo_sporgente", "insegna_cartello_basso", "ostacolo_sospeso_generico")
CLASS_ORDER = ("monopattino",) + SUSPENDED


def bakeoff_prompts():
    """Le coppie (classe, indice, prompt) del bake-off, in ordine stabile."""
    out = []
    for cls in CLASS_ORDER:
        for i in BAKEOFF_PROMPT_INDICES:
            out.append((cls, i, PROMPTS[cls][i]))
    return out


def all_prompts():
    """Tutte le coppie (classe, indice, prompt), per la produzione."""
    out = []
    for cls in CLASS_ORDER:
        for i, p in enumerate(PROMPTS[cls]):
            out.append((cls, i, p))
    return out
