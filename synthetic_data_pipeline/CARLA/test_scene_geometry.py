# test_scene_geometry.py
#
# Casi per scene_geometry.py (piazzamento puro, senza server) e per la validazione
# degli scenari in carla_capture.py con un catalogo finto. I numeri attesi vengono
# dalle scene guardate a vista il 05/10 sul viale del seme 2: marcia verso +X,
# alberi a sinistra (y minore), ramo con yaw assoluto 110.
#
# pytest non e' installato: si esegue con  python test_scene_geometry.py
# (nel venv CARLA, che ha il pacchetto carla: serve solo per importare carla_capture).
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scene_geometry as geo

EAST = (1.0, 0.0)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_la_destra_di_est_e_piu_y():
    assert geo.right_of(EAST) == (-0.0, 1.0)
    x, y = geo.frame_point((0.0, 0.0), EAST, 10.0, -2.0)
    assert close(x, 10.0) and close(y, -2.0)


def test_coordinate_lungo_e_di_lato():
    p = geo.frame_point((5.0, 3.0), EAST, 4.0, 1.5)
    assert close(geo.along_of(p, (5.0, 3.0), EAST), 4.0)
    assert close(geo.lateral_of(p, (5.0, 3.0), EAST), 1.5)


def test_ramo_da_un_tronco_a_sinistra_punta_a_destra_verso_il_pedone():
    across, yaw, side = geo.across_from_anchor((-16.33, 76.59), (-31.5, 78.4), EAST, 20.0)
    assert side == -1
    assert close(across[0], 0.0) and close(across[1], 1.0)
    assert close(yaw, 110.0)


def test_ramo_da_un_tronco_a_destra_e_lo_specchio():
    _, yaw, side = geo.across_from_anchor((10.0, 80.5), (0.0, 78.4), EAST, 20.0)
    assert side == 1
    assert close(yaw, -110.0)


def test_nessun_albero_entro_il_raggio_e_un_fallimento():
    trees = [(-25.77, 76.40), (-16.33, 76.59), (-3.39, 76.37)]
    tree, d = geo.nearest_within(trees, (-17.5, 76.4), 3.0)
    assert tree == (-16.33, 76.59) and d < 1.3
    tree, d = geo.nearest_within(trees, (-10.0, 76.4), 3.0)
    assert tree is None and d > 3.0


def test_schivata_resta_in_linea_poi_scarta():
    args = (0.0, 0.9, 9.0, 3.0)
    assert geo.swerve_lateral(60.0, *args) == 0.0
    assert geo.swerve_lateral(9.0, *args) == 0.0
    assert close(geo.swerve_lateral(6.0, *args), 0.45)
    assert geo.swerve_lateral(3.0, *args) == 0.9
    assert geo.swerve_lateral(-5.0, *args) == 0.9
    values = [geo.swerve_lateral(g, *args) for g in (9.0, 8.0, 6.0, 4.0, 3.0)]
    assert values == sorted(values)


def test_schivata_con_estremi_invertiti_si_ferma():
    try:
        geo.swerve_lateral(5.0, 0.0, 0.9, 3.0, 9.0)
    except ValueError:
        return
    raise AssertionError("swerve_from <= swerve_to doveva sollevare ValueError")


def test_pezzi_del_monopattino_seguono_il_verso_di_marcia():
    west = (-1.0, 0.0)
    x, y = geo.part_point((0.0, 0.0), west, 0.35, 0.0)
    assert close(x, -0.35) and close(y, 0.0)
    # chi va verso -X ha la destra verso -Y
    x, y = geo.part_point((0.0, 0.0), west, 0.0, -0.26)
    assert close(x, 0.0) and close(y, 0.26)


def test_yaw_della_marcia():
    assert close(geo.yaw_of((0.0, 1.0)), 90.0)
    assert close(abs(geo.yaw_of((-1.0, 0.0))), 180.0)
    assert close(geo.yaw_of((math.sqrt(0.5), math.sqrt(0.5))), 45.0)


# --- validazione degli scenari, con un catalogo finto --------------------------

class FakeLibrary:
    def __init__(self, ids):
        self.ids = set(ids)

    def find(self, bp_id):
        if bp_id not in self.ids:
            raise IndexError(bp_id)
        return bp_id

    def filter(self, pattern):
        return []


def expect_exit(fn):
    try:
        fn()
    except SystemExit as exc:
        return str(exc)
    raise AssertionError("doveva fermarsi con sys.exit")


def test_gli_scenari_del_repository_sono_validi():
    import carla_capture as C
    from scenarios import SCENARIOS
    ids = set()
    for sc in SCENARIOS.values():
        ids |= set(a["blueprint"] for a in sc["actors"] if "blueprint" in a)
        ids |= set(m["rider"] for m in sc.get("movers", []))
    lib = FakeLibrary(ids)
    for sc in SCENARIOS.values():
        C.validate_scenario(sc, lib)


def test_refuso_in_fps_si_ferma():
    import carla_capture as C
    from scenarios import SCENARIOS
    sc = dict(SCENARIOS["s2r_insegna"])
    sc["fsp"] = sc.pop("fps")
    msg = expect_exit(lambda: C.validate_scenario(sc, FakeLibrary([])))
    assert "fsp" in msg


def test_geometria_s2r_incompleta_si_ferma():
    import carla_capture as C
    from scenarios import SCENARIOS
    sc = dict(SCENARIOS["s2r_insegna"])
    del sc["min_window_s"]
    msg = expect_exit(lambda: C.validate_scenario(sc, FakeLibrary([])))
    assert "min_window_s" in msg


def test_le_scene_s2r_fissano_il_viale():
    from scenarios import SCENARIOS
    for name, sc in SCENARIOS.items():
        if name.startswith("s2r_"):
            assert sc["walker_route"]["spawn"][:2] == [-34.1138916015625, 78.43087005615234], name


def test_percorso_con_punto_incompleto_si_ferma():
    import carla_capture as C
    from scenarios import SCENARIOS
    sc = dict(SCENARIOS["s2r_insegna"])
    sc["walker_route"] = {"spawn": [-34.1, 78.4], "target": [89.2, -1.5, 0.16]}
    msg = expect_exit(lambda: C.validate_scenario(sc, FakeLibrary([])))
    assert "walker_route.spawn" in msg


def test_ostacolo_con_blueprint_e_mesh_si_ferma():
    import carla_capture as C
    from scenarios import SCENARIOS
    sc = dict(SCENARIOS["s2r_ramo"])
    sc["actors"] = [dict(sc["actors"][0], blueprint="static.prop.box02")]
    expect_exit(lambda: C.validate_scenario(sc, FakeLibrary(["static.prop.box02"])))


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok     %s" % name)
        except AssertionError as exc:
            failed += 1
            print("FALLITO %s %s" % (name, exc))
    print("\n%d test, %d falliti" % (len(tests), failed))
    sys.exit(1 if failed else 0)
