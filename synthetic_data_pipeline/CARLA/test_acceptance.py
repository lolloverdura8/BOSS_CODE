# test_acceptance.py
#
# Sessioni sintetiche minime (160x90, 48 Hz) per i controlli di acceptance.py e
# per l'unione dei gruppi in carla_export.py / carla_gt.py. Ogni caso riproduce
# un difetto trovato nella revisione del 05/10: l'ostacolo mai nato che passava
# tutti i controlli, la visibilita' a un pixel, un buco di un frame che spezzava
# la finestra, la camera ferma, il monopattino contato come cinque oggetti.
#
# pytest non e' installato: si esegue con  python test_acceptance.py
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import acceptance as A
import carla_export as E
import carla_gt as G

W, H = 160, 90
DT = 1.0 / 48.0
N = 40
SKY, POLE = 11, 6


def make_session(objects, n=N, speed=1.4, min_window_s=0.5, declared=None, failures=()):
    """objects: lista di (id, tag, gruppo, frames_visibili, area_px)."""
    root = tempfile.mkdtemp(prefix="acc_")
    for sub in ("rgb", "depth", "semantic", "instance"):
        os.makedirs(os.path.join(root, sub))
    poses = []
    for i in range(n):
        stem = "%06d.png" % i
        inst = np.zeros((H, W, 3), np.uint8)
        sem = np.zeros((H, W, 3), np.uint8)
        sem[:10, :, 2] = SKY                       # cielo in alto
        depth = np.full((H, W), 5000, np.uint16)
        depth[:10, :] = 65535
        depth[80:, :] = 1000 + 10 * i              # un po' di varieta' per A3
        col = 0
        for iid, tag, _, frames, area in objects:
            if i in frames:
                rows = max(1, area // 10)
                inst[20:20 + rows, col:col + 10, 0] = iid >> 8
                inst[20:20 + rows, col:col + 10, 1] = iid & 0xFF
                inst[20:20 + rows, col:col + 10, 2] = tag
                extra = area - rows * 10
                if extra > 0:
                    inst[20 + rows, col:col + extra, 0] = iid >> 8
                    inst[20 + rows, col:col + extra, 1] = iid & 0xFF
                    inst[20 + rows, col:col + extra, 2] = tag
            col += 12
        cv2.imwrite(os.path.join(root, "rgb", stem), np.full((H, W, 3), 128, np.uint8))
        cv2.imwrite(os.path.join(root, "depth", stem), depth)
        cv2.imwrite(os.path.join(root, "semantic", stem), sem)
        cv2.imwrite(os.path.join(root, "instance", stem), inst)
        x = speed * i * DT
        poses.append({"index": i, "frame": 100 + i, "frames": {"rgb": 100 + i, "depth": 100 + i},
                      "camera": {"location": {"x": x, "y": 0.0, "z": 1.5}}})
    with open(os.path.join(root, "poses.jsonl"), "w") as f:
        for p in poses:
            f.write(json.dumps(p) + "\n")
    actors = []
    for iid, _, group, _, _ in objects:
        a = {"instance_id": iid, "boss_class": "monopattino" if group else "ramo_sporgente"}
        if group:
            a["group"] = group
        actors.append(a)
    n_declared = declared if declared is not None else len({o[2] or o[0] for o in objects})
    session = {
        "scenario": "sintetica", "scenario_config": {"actors": [{}] * n_declared,
                                                    "min_window_s": min_window_s},
        "intrinsics": {}, "fov": 82, "width": W, "height": H, "depth_scale": 1000.0,
        "carla_version": "test", "map": "x", "seed": 0, "fixed_delta_seconds": DT,
        "camera_transform_relative": {}, "spawned_actors": actors, "rgb_sha256": "x",
        "walker_speed_mps": 1.4, "spawn_failures": list(failures),
    }
    with open(os.path.join(root, "session.json"), "w") as f:
        json.dump(session, f)
    return root


def results_of(root):
    try:
        res, info = A.run_acceptance_checks(root)
        return dict((n, ok) for n, ok, _ in res), info
    finally:
        shutil.rmtree(root, ignore_errors=True)


ALL = set(range(N))


def test_sessione_buona_passa_tutto():
    r, info = results_of(make_session([(700, POLE, None, ALL, 100)]))
    assert r["A1"] and r["A2"] and r["A3"] and r["A4"] and r["A7"] and r["A8"] and r["A9"], r
    assert info["windows"]["ramo_sporgente#700"] == [0, N - 1]


def test_ostacolo_mai_nato_fa_fallire_a7_e_a8():
    root = make_session([], declared=1,
                        failures=[{"reason": "spawn fallito per prova: collisione"}])
    r, _ = results_of(root)
    assert r["A7"] is False and r["A8"] is False, r


def test_un_pixel_non_basta():
    r, _ = results_of(make_session([(700, POLE, None, ALL, 1)]))
    assert r["A7"] is False and r["A8"] is False, r


def test_buco_breve_non_spezza_la_finestra():
    frames = ALL - {20}
    r, info = results_of(make_session([(700, POLE, None, frames, 100)]))
    assert r["A8"] is True, r
    assert info["windows"]["ramo_sporgente#700"] == [0, N - 1]


def test_buco_lungo_spezza_la_finestra():
    frames = ALL - set(range(15, 30))          # 15 frame = 0,31 s > 0,25 s
    r, _ = results_of(make_session([(700, POLE, None, frames, 100)]))
    assert r["A8"] is False, r


def test_camera_ferma_fa_fallire_a9():
    r, _ = results_of(make_session([(700, POLE, None, ALL, 100)], speed=0.0))
    assert r["A9"] is False, r


def test_il_gruppo_e_un_oggetto_solo_visibile_per_unione():
    objs = [(801, 12, "monopattino_0", ALL, 40), (802, POLE, "monopattino_0", ALL, 40)]
    r, info = results_of(make_session(objs, declared=1))
    assert r["A7"] and r["A8"], r
    assert list(info["windows"]) == ["monopattino_0"]


def test_finestra_migliore():
    vis = [False, True, True, False, True, False, False, False, True]
    assert A.best_window(vis, 1) == (1, 4)
    assert A.best_window(vis, 0) == (1, 2)
    assert A.best_window([False, False], 1) is None


def test_export_unisce_i_pezzi_di_un_gruppo():
    a = np.zeros((H, W), bool)
    a[10:20, 10:20] = True
    b = np.zeros((H, W), bool)
    b[30:32, 50:52] = True                     # 4 px: da solo sotto soglia
    c = np.zeros((H, W), bool)
    c[60:70, 100:110] = True
    depth = np.full((H, W), 4000, np.uint16)
    recs = [
        {"instance_id": 801, "carla_tag": 12, "bbox": [10, 10, 10, 10], "area_px": 100,
         "truncated": False, "distance_m": 4.0, "boss_class": "monopattino"},
        {"instance_id": 802, "carla_tag": POLE, "bbox": [50, 30, 2, 2], "area_px": 4,
         "truncated": False, "distance_m": 4.0, "boss_class": "monopattino"},
        {"instance_id": 900, "carla_tag": 14, "bbox": [100, 60, 10, 10], "area_px": 100,
         "truncated": False, "distance_m": 4.0, "boss_class": "automobile"},
    ]
    out = E.merge_groups(recs, [a, b, c], {801: "m0", 802: "m0"}, depth, "median_mask", 1000.0, 64)
    assert len(out) == 2
    merged = next(r for r in out if r.get("group") == "m0")
    assert merged["bbox"] == [10, 10, 42, 22] and merged["area_px"] == 104
    assert merged["instance_id"] == 801 and merged["parts"] == [[12, 801], [POLE, 802]]
    assert abs(merged["distance_m"] - 4.0) < 1e-9


def test_senza_gruppi_l_export_non_cambia():
    rec = {"instance_id": 1, "area_px": 100}
    assert E.merge_groups([rec], [None], {}, None, "median_mask", 1000.0, 64) == [rec]


def test_record_mask_ricostruisce_l_unione():
    tag = np.zeros((H, W), np.uint32)
    ids = np.zeros((H, W), np.uint32)
    tag[0:2, 0:2], ids[0:2, 0:2] = 12, 801
    tag[5:6, 5:8], ids[5:6, 5:8] = POLE, 802
    rec = {"carla_tag": 12, "instance_id": 801, "parts": [[12, 801], [POLE, 802]]}
    assert int(G.record_mask(tag, ids, rec).sum()) == 7
    del rec["parts"]
    assert int(G.record_mask(tag, ids, rec).sum()) == 4


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
