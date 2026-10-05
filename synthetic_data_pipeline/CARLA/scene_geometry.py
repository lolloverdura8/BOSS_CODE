# scene_geometry.py
#
# Geometria pura del piazzamento degli attori degli scenari: nessun import di carla,
# cosi' i conti si provano senza server (test_scene_geometry.py). I punti sono tuple
# (x, y) nel piano della mappa, le direzioni versori 2D.
#
# Convenzione CARLA, la stessa di path_frame() in carla_capture.py: sistema
# levogiro con X avanti, Y a destra e yaw crescente verso +Y, quindi la destra di
# una direzione (fx, fy) e' (-fy, fx). Sbagliare questo segno specchia gli ostacoli
# senza che nulla lo segnali.
import math


def right_of(fwd):
    return (-fwd[1], fwd[0])


def yaw_of(vec):
    return math.degrees(math.atan2(vec[1], vec[0]))


def frame_point(origin, fwd, forward, lateral):
    """Il punto `forward` metri avanti e `lateral` metri a destra di origin."""
    r = right_of(fwd)
    return (origin[0] + fwd[0] * forward + r[0] * lateral,
            origin[1] + fwd[1] * forward + r[1] * lateral)


def along_of(point, origin, fwd):
    """Coordinata di point lungo fwd, misurata da origin."""
    return (point[0] - origin[0]) * fwd[0] + (point[1] - origin[1]) * fwd[1]


def lateral_of(point, origin, fwd):
    """Coordinata laterale di point rispetto a origin: positiva a destra di fwd."""
    r = right_of(fwd)
    return (point[0] - origin[0]) * r[0] + (point[1] - origin[1]) * r[1]


def nearest_within(points, target, radius):
    """(punto, distanza) del punto piu' vicino a target, oppure (None, distanza minima).

    None quando nessun punto sta entro radius: chi chiama deve trattarlo come un
    fallimento del piazzamento, non ripiegare sul piu' vicino comunque, perche' un
    ancoraggio a un albero lontano mette l'ostacolo dove la scena non lo prevede.
    """
    best, best_d = None, float("inf")
    for p in points:
        d = math.hypot(p[0] - target[0], p[1] - target[1])
        if d < best_d:
            best, best_d = p, d
    if best is None or best_d > radius:
        return None, best_d
    return best, best_d


def across_from_anchor(anchor, origin, fwd, yaw_toward_walker):
    """Direzione di un oggetto che parte da un'ancora laterale e attraversa il percorso.

    L'oggetto (un ramo che esce dal tronco) punta dall'ancora verso la linea di
    marcia. yaw_toward_walker > 0 lo ruota verso il pedone che arriva, cioe' contro
    fwd, da qualunque lato stia l'ancora: per questo il verso della rotazione
    dipende dal lato. Restituisce (versore attraverso il percorso, yaw in gradi,
    lato), con lato -1 se l'ancora sta a sinistra della marcia e +1 se a destra.
    """
    side = -1 if lateral_of(anchor, origin, fwd) < 0 else 1
    r = right_of(fwd)
    across = (-side * r[0], -side * r[1])
    return across, yaw_of(across) - side * yaw_toward_walker, side


def smoothstep(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def swerve_lateral(gap, lateral, pass_lateral, swerve_from, swerve_to):
    """Scostamento laterale di un mezzo in arrivo, in funzione della distanza dal pedone.

    Fino a swerve_from metri il mezzo resta su `lateral` (la linea di marcia del
    pedone), poi scarta con una smoothstep fino a pass_lateral, che raggiunge a
    swerve_to metri e mantiene dopo il sorpasso. E' la schivata all'ultimo momento
    di chi arriva veloce, non una traiettoria gia' parallela da lontano.
    """
    if swerve_from <= swerve_to:
        raise ValueError("swerve_from (%.2f) deve superare swerve_to (%.2f)"
                         % (swerve_from, swerve_to))
    t = (swerve_from - gap) / (swerve_from - swerve_to)
    return lateral + (pass_lateral - lateral) * smoothstep(t)


def part_point(base, heading, dx, dy):
    """Punto di un pezzo solidale a un mezzo: dx avanti nel verso di marcia, dy a destra."""
    r = right_of(heading)
    return (base[0] + heading[0] * dx + r[0] * dy,
            base[1] + heading[1] * dx + r[1] * dy)
