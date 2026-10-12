# test_prepare_controls.py
#
# Casi per prepare_controls.py su array e sessioni sintetiche minime (40x32 px,
# 48 Hz). Ogni caso fissa una scelta del piano del 12/10: la ricetta CARLA copiata
# senza derive, il cielo saturo che deve tornare nero, l'ostacolo che senza la sua
# maschera non avrebbe bordi, la finestra ancorata alla fine di A8, il ritaglio di
# Wan che non sposta la box, il monopattino come unione di pezzi, il pacchetto che
# non si sovrascrive.
#
# Dopo la revisione a mutazioni del 12/10 (45 mutazioni su 62 sopravvivevano) le
# sessioni sintetiche cambiano a ogni frame (sfondo, depth, posizione, distanza),
# il ritaglio di prova e' asimmetrico (6 sopra, 2 sotto) con righe-marcatore, e i
# file del pacchetto si rileggono da disco invece di fidarsi di cio' che la funzione
# restituisce. Un ffmpeg finto (FAKE_FFMPEG) prova la codifica senza WSL.
#
# pytest non e' installato: si esegue con  python test_prepare_controls.py
# (Python 3.11 del PC, quello con cv2 e numpy).
import hashlib
import json
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import prepare_controls as P
from carla_gt import GroundTruthError

H, W = 40, 32
# Le sessioni di prova sono 32x40: il controllo sulla cattura 1280x720 si sposta qui.
P.SOURCE_SIZE = (W, H)
VEG, SKY, ROAD, PED, DYN, POLE, FENCE = 9, 11, 1, 12, 21, 6, 5
TINY = {"fps": 16, "n_frames": 5, "crop_top": 8, "crop_bottom": 8}
FULL = {"fps": 16, "n_frames": 5, "crop_top": 0, "crop_bottom": 0}
ASYM = {"fps": 16, "n_frames": 5, "crop_top": 6, "crop_bottom": 2}
# Righe-marcatore: la prima e l'ultima riga che il ritaglio ASYM tiene.
TOP_ROW, BOTTOM_ROW = 6, H - 3
TOP_RGB, BOTTOM_RGB = (250, 0, 0), (0, 0, 250)          # RGB
TOP_DEPTH, BOTTOM_DEPTH = 1234, 4321
# Un punto Fence alla colonna i % W della riga 25: dice da quale frame viene la seg.
FENCE_ROW = 25

FAKE_FFMPEG = r'''
import json, os, sys
import cv2
args = sys.argv[1:]
if args and args[0] == "--drop":
    drop, args = 1, args[1:]
else:
    drop = 0
if args == ["-version"]:
    print("ffmpeg finto 1.0")
    sys.exit(0)
with open(os.environ["FAKE_FFMPEG_LOG"], "a") as f:
    f.write(json.dumps(args) + "\n")
fps = float(args[args.index("-framerate") + 1])
pattern, out = args[args.index("-i") + 1], args[-1]
frames, k = [], 0
while os.path.exists(pattern % k):
    frames.append(cv2.imread(pattern % k))
    k += 1
frames = frames[:len(frames) - drop]
h, w = frames[0].shape[:2]
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for fr in frames:
    vw.write(fr)
vw.release()
'''


def write_instance(path, objects):
    """objects: lista di (id, tag, righe, colonne)."""
    img = np.zeros((H, W, 3), np.uint8)
    for iid, tag, rows, cols in objects:
        img[rows, cols, 0] = iid >> 8
        img[rows, cols, 1] = iid & 0xFF
        img[rows, cols, 2] = tag
    cv2.imwrite(path, img)


def frame_objects(i, objects, moving, absent=()):
    if i in absent:
        return []
    if objects:
        return objects
    cols = slice(2 + i, 13 + i) if moving else slice(5, 16)
    return [(147, VEG, slice(10, 20) if moving else slice(2, 11), cols)]


def write_frame_files(root, i, objects):
    """Un frame sintetico: sfondo, depth e distanze cambiano con i, righe-marcatore fisse."""
    stem = f"{i:06d}.png"
    sem = np.zeros((H, W, 3), np.uint8)
    sem[:2, :, 2] = SKY
    sem[30:, :, 2] = ROAD
    sem[TOP_ROW, :, 2] = POLE
    sem[FENCE_ROW, i % W, 2] = FENCE                     # un punto che cambia a ogni frame
    depth = np.full((H, W), 5000 + 10 * i, np.uint16)
    depth[:2, :] = 65535
    depth[TOP_ROW, :], depth[BOTTOM_ROW, :] = TOP_DEPTH, BOTTOM_DEPTH
    rgb = np.full((H, W, 3), 10 + i, np.uint8)
    rgb[TOP_ROW], rgb[BOTTOM_ROW] = TOP_RGB, BOTTOM_RGB
    for _, tag, rows, cols in objects:
        sem[rows, cols, 2] = tag
        depth[rows, cols] = 3000 + 100 * i
        rgb[rows, cols] = 230
    write_instance(os.path.join(root, "instance", stem), objects)
    cv2.imwrite(os.path.join(root, "semantic", stem), sem)
    cv2.imwrite(os.path.join(root, "depth", stem), depth)
    cv2.imwrite(os.path.join(root, "rgb", stem), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def make_session(n=13, objects=None, actors=None, moving=False, absent=()):
    """Sessione sintetica: cielo in alto, strada in basso, ostacolo Vegetation 147.

    moving: l'ostacolo parte dalla colonna 2 e si sposta di una colonna a frame,
    cosi' un frame preso al posto di un altro si vede nella box.
    absent: frame in cui l'ostacolo non c'e'.
    """
    actors = actors or [{"instance_id": 147, "boss_class": "ramo_sporgente"}]
    root = tempfile.mkdtemp(prefix="prep_")
    for sub in ("rgb", "depth", "semantic", "instance"):
        os.makedirs(os.path.join(root, sub))
    poses = []
    for i in range(n):
        write_frame_files(root, i, frame_objects(i, objects, moving, absent))
        poses.append({"index": i, "frame": 1000 + i, "timestamp": i / 48.0,
                      "tracked_distances_m": {str(a["instance_id"]): 3.2 + 0.1 * i
                                              + a["instance_id"] % 2 for a in actors}})
    with open(os.path.join(root, "poses.jsonl"), "w") as f:
        f.writelines(json.dumps(p) + "\n" for p in poses)
    session = {"scenario": "s2r_test", "fixed_delta_seconds": 1.0 / 48.0, "width": W,
               "height": H, "intrinsics": {"fx": 20.0, "fy": 20.0, "cx": 16.0, "cy": 20.0},
               "depth_scale": 1000.0, "spawned_actors": actors, "code_version": "test",
               "n_frames_written": n}
    for sub in ("rgb", "depth", "instance"):
        session[sub + "_sha256"] = P.dir_sha256(os.path.join(root, sub))
    name = actors[0].get("group") or f"{actors[0]['boss_class']}#{actors[0]['instance_id']}"
    acceptance = {"valid": True, "failed": [], "windows": {name: [0, n - 1]}}
    for fname, data in (("session.json", session), ("acceptance.json", acceptance)):
        with open(os.path.join(root, fname), "w") as f:
            json.dump(data, f)
    return root, session, acceptance


def expect_error(exc_type, fn, *args, contains=""):
    try:
        fn(*args)
    except exc_type as exc:
        assert contains in str(exc), exc
        return
    raise AssertionError(f"{fn.__name__} non ha sollevato {exc_type.__name__}")


class Format:
    """Registra un formato di prova in MODEL_FORMATS e lo toglie all'uscita."""

    def __init__(self, name, fmt):
        self.name, self.fmt = name, fmt

    def __enter__(self):
        P.MODEL_FORMATS[self.name] = self.fmt
        return self.name

    def __exit__(self, *exc):
        del P.MODEL_FORMATS[self.name]


def fmt_with_size(fmt):
    return dict(fmt, width=W, height=H)


# --- ricetta -----------------------------------------------------------------------

def test_log_depth_come_la_ricetta_su_valori_noti():
    gray = P.depth_to_log_grayscale(np.array([0.01, 1000.0, 1.0, 10.0, 100.0], np.float32))
    assert gray[0] == 255 and gray[1] == 0, gray.tolist()
    # log uniforme: ogni decade vale 255/5 = 51 livelli, vicino = chiaro
    assert abs(int(gray[2]) - 153) <= 1 and abs(int(gray[3]) - 102) <= 1, gray.tolist()
    assert gray[2] > gray[3] > gray[4] > gray[1]


def test_cielo_saturo_e_zeri_tornano_neri():
    depth = np.array([[10, 65535, 0, 1000]], np.uint16)
    gray, n_sat, n_zero = P.depth_control(depth, 1000.0)
    assert gray[0, 0] == 255 and gray[0, 1] == 0 and gray[0, 2] == 0, gray.tolist()
    assert gray[0, 3] == P.depth_to_log_grayscale(np.array([1.0]))[0]
    assert (n_sat, n_zero) == (1, 1)


def test_depth_control_rispetta_la_scala():
    gray, _, _ = P.depth_control(np.array([[500]], np.uint16), 500.0)
    assert gray[0, 0] == P.depth_to_log_grayscale(np.array([1.0]))[0]


def test_palette_cityscapes_ufficiale():
    pal = P.CITYSCAPES_PALETTE
    assert len(pal) == 29
    assert tuple(pal[VEG]) == (107, 142, 35) and tuple(pal[8]) == (220, 220, 0)
    assert tuple(pal[P.TAG_RIDER]) == (255, 0, 0) and tuple(pal[P.TAG_BICYCLE]) == (119, 11, 32)
    assert tuple(pal[14]) == (0, 0, 142) and tuple(pal[28]) == (180, 165, 180)
    expect_error(GroundTruthError, P.cityscapes, np.array([[29]], np.uint32))


def square_scene(seg_color):
    rgb = np.full((H, W, 3), 30, np.uint8)
    rgb[10:25, 8:22] = 220
    seg = np.zeros((H, W, 3), np.uint8)
    seg[:] = seg_color
    mask = np.zeros((H, W), bool)
    mask[10:25, 8:22] = True
    return rgb, seg, mask


def test_edge_ostacolo_vegetation_senza_maschera_non_ha_bordi():
    rgb, seg, mask = square_scene(P.CITYSCAPES_PALETTE[VEG])
    _, edges = P.masked_edges_from_semseg(rgb, seg, P.CANNY_CLASSES)
    assert not edges.any()
    _, edges = P.masked_edges_from_semseg(rgb, seg, P.CANNY_CLASSES, extra_mask=mask)
    ring = np.zeros((H, W), bool)
    ring[8:27, 6:24] = True
    ring[12:23, 10:20] = False
    assert edges.any() and not edges[~ring].any(), "bordi fuori dalla sagoma"


def test_edge_classe_in_lista_ha_bordi_anche_senza_maschera():
    rgb, seg, _ = square_scene(P.CITYSCAPES_PALETTE[ROAD])
    _, edges = P.masked_edges_from_semseg(rgb, seg, P.CANNY_CLASSES)
    assert edges.any()


# --- finestra e formato --------------------------------------------------------------

def test_finestra_ancorata_alla_fine_di_a8():
    src = P.clip_sources((0, 501), 93, 3)
    assert (src[0], src[-1], len(src)) == (225, 501, 93)
    src = P.clip_sources((3, 340), 121, 2)
    assert (src[0], src[-1], len(src)) == (100, 340, 121)
    expect_error(P.PrepareError, P.clip_sources, (0, 200), 93, 3, contains="troppo corta")


def test_passo_dai_48_hz():
    s = {"fixed_delta_seconds": 1.0 / 48.0}
    assert P.model_stride(s, 16) == 3 and P.model_stride(s, 24) == 2
    # rumore di virgola mobile sotto la tolleranza: 2,9999999 e' 3, non 2
    assert P.model_stride({"fixed_delta_seconds": 1.0 / 48.0 * (1 + 2e-7)}, 16) == 3
    expect_error(P.PrepareError, P.model_stride, {"fixed_delta_seconds": 1.0 / 50.0}, 16)


def test_percorso_per_ffmpeg_in_wsl():
    assert P.to_ffmpeg_path("C:\\Users\\pc\\x\\%04d.png", P.FFMPEG_WSL) == "/mnt/c/Users/pc/x/%04d.png"
    assert P.to_ffmpeg_path("C:\\a b\\c.mp4", ["WSL.EXE", "-e", "f"]) == "/mnt/c/a b/c.mp4"
    assert P.to_ffmpeg_path("C:\\a\\b.mp4", ["ffmpeg"]) == "C:\\a\\b.mp4"


# --- ostacolo -------------------------------------------------------------------------

def test_un_solo_oggetto_tracciato_e_la_sua_finestra():
    two = {"spawned_actors": [{"instance_id": 1, "boss_class": "ramo_sporgente"},
                              {"instance_id": 2, "boss_class": "insegna_cartello_basso"}]}
    expect_error(P.PrepareError, P.the_obstacle, two, {"windows": {}}, contains="atteso 1")
    one = {"spawned_actors": two["spawned_actors"][:1]}
    expect_error(P.PrepareError, P.the_obstacle, one, {"windows": {}}, contains="A8")


def test_pezzo_mai_visibile_finisce_in_unseen():
    actors = [{"instance_id": 147, "boss_class": "ramo_sporgente"},
              {"instance_id": 999, "boss_class": "ramo_sporgente", "part": "fantasma"}]
    root, _, _ = make_session(actors=actors[:1])
    try:
        parts, unseen = P.obstacle_parts(root, actors, [0, 3])
        assert [p["instance_id"] for p in parts] == [147] and unseen == ["fantasma"]
        expect_error(P.PrepareError, P.obstacle_parts, root, actors[1:], [0],
                     contains="mai visibile")
    finally:
        shutil.rmtree(root)


def test_id_condiviso_con_altro_tag_ferma_lo_script():
    # stesso id 147 su due oggetti di tag diverso, 50 % e 50 %
    objects = [(147, VEG, slice(2, 7), slice(5, 15)), (147, ROAD, slice(30, 35), slice(5, 15))]
    root, session, acceptance = make_session(objects=objects)
    try:
        _, actors, _ = P.the_obstacle(session, acceptance)
        expect_error(P.PrepareError, P.obstacle_parts, root, actors, [0, 3],
                     contains="tag dominante")
    finally:
        shutil.rmtree(root)


# --- frame e GT ----------------------------------------------------------------------

def test_ritaglio_wan_ricalcola_box_e_tronca_in_alto():
    root, session, acceptance = make_session()
    try:
        _, actors, _ = P.the_obstacle(session, acceptance)
        parts, unseen = P.obstacle_parts(root, actors, [0])
        assert parts[0]["carla_tag"] == VEG and not unseen
        fr = P.build_frame(root, 0, parts, fmt_with_size(TINY), 1000.0)
        assert fr["rgb"].shape == (24, W, 3) and fr["depth_u16"].shape == (24, W)
        lab = P.frame_label(fr["mask"], fr["depth_u16"], 1000.0, {}, parts)
        # ostacolo alle righe 2..10 della sorgente: dopo 8 righe tolte resta 0..2
        assert lab["bbox"] == [5, 0, 11, 3] and lab["truncated"], lab
        assert lab["area_px"] == 33 and not lab["visible"]
        assert abs(lab["distance_m"] - 3.0) < 1e-9
    finally:
        shutil.rmtree(root)


def test_ritaglio_asimmetrico_sposta_marcatori_e_contenuto():
    root, session, acceptance = make_session(moving=True)
    try:
        _, actors, _ = P.the_obstacle(session, acceptance)
        parts, _ = P.obstacle_parts(root, actors, [4])
        fr = P.build_frame(root, 4, parts, fmt_with_size(ASYM), 1000.0)
        assert fr["rgb"].shape == (H - 8, W, 3)
        assert tuple(fr["rgb"][0, 0]) == TOP_RGB and tuple(fr["rgb"][-1, 0]) == BOTTOM_RGB
        assert fr["depth_u16"][0, 0] == TOP_DEPTH and fr["depth_u16"][-1, 0] == BOTTOM_DEPTH
        assert tuple(fr["seg"][0, 0]) == tuple(P.CITYSCAPES_PALETTE[POLE])
        # ostacolo alle righe 10..19 della sorgente -> 4..13 dopo 6 righe tolte
        rows = np.nonzero(fr["mask"].any(axis=1))[0]
        assert (rows.min(), rows.max()) == (4, 13), rows
        assert fr["rgb"][1, 0, 0] == 10 + 4, "sfondo di un altro frame"
    finally:
        shutil.rmtree(root)


def test_edge_del_frame_usa_la_maschera_dell_ostacolo():
    root, session, acceptance = make_session()
    try:
        _, actors, _ = P.the_obstacle(session, acceptance)
        parts, _ = P.obstacle_parts(root, actors, [0])
        fr = P.build_frame(root, 0, parts, fmt_with_size(FULL), 1000.0)
        sem = cv2.imread(os.path.join(root, "semantic", "000000.png"))[:, :, 2]
        _, expected = P.masked_edges_from_semseg(fr["rgb"], P.cityscapes(sem), P.CANNY_CLASSES,
                                                 extra_mask=fr["mask"])
        assert (fr["edge"] == expected).all()
        _, without = P.masked_edges_from_semseg(fr["rgb"], P.cityscapes(sem), P.CANNY_CLASSES)
        assert (fr["edge"] != without).any(), "la maschera non cambia nulla: prova inutile"
    finally:
        shutil.rmtree(root)


def test_monopattino_unione_dei_pezzi_e_seg_ricolorata():
    objects = [(349, PED, slice(12, 20), slice(4, 10)), (350, DYN, slice(20, 22), slice(2, 14))]
    actors = [{"instance_id": 349, "boss_class": "monopattino", "group": "monopattino_0",
               "part": "conducente"},
              {"instance_id": 350, "boss_class": "monopattino", "group": "monopattino_0",
               "part": "pedana"}]
    root, session, acceptance = make_session(objects=objects, actors=actors)
    try:
        name, actors_found, _ = P.the_obstacle(session, acceptance)
        assert name == "monopattino_0" and len(actors_found) == 2
        parts, _ = P.obstacle_parts(root, actors_found, [0])
        fr = P.build_frame(root, 0, parts, fmt_with_size(FULL), 1000.0)
        assert int(fr["mask"].sum()) == 8 * 6 + 2 * 12
        assert tuple(fr["seg"][15, 6]) == (255, 0, 0)       # conducente -> Rider
        assert tuple(fr["seg"][21, 12]) == (119, 11, 32)    # pedana -> Bicycle
        assert tuple(fr["seg"][35, 20]) == tuple(P.CITYSCAPES_PALETTE[ROAD])
        lab = P.frame_label(fr["mask"], fr["depth_u16"], 1000.0,
                            {"tracked_distances_m": {"349": 4.2, "350": 3.2}}, parts)
        assert lab["bbox"] == [2, 12, 12, 10] and lab["center_distance_m"] == 3.2
    finally:
        shutil.rmtree(root)


def label_of(mask, depth=None, scale=1000.0):
    depth = np.full(mask.shape, 2000, np.uint16) if depth is None else depth
    return P.frame_label(mask, depth, scale, {}, [])


def test_truncated_su_ogni_lato_e_soglia_di_visibilita():
    for rows, cols in ((slice(5, 9), slice(0, 3)), (slice(5, 9), slice(W - 3, W)),
                       (slice(H - 3, H), slice(5, 9)), (slice(0, 3), slice(5, 9))):
        m = np.zeros((H, W), bool)
        m[rows, cols] = True
        assert label_of(m)["truncated"], (rows, cols)
    m = np.zeros((H, W), bool)
    m[5:13, 5:13] = True                                   # 64 px, lontano dai bordi
    lab = label_of(m)
    assert lab["visible"] and not lab["truncated"] and lab["area_px"] == 64
    m[12, 12] = False
    assert not label_of(m)["visible"]


def test_maschera_vuota_satura_e_scala():
    lab = label_of(np.zeros((H, W), bool))
    assert lab["bbox"] is None and lab["distance_m"] is None and not lab["visible"]
    m = np.zeros((H, W), bool)
    m[5:10, 5:10] = True
    depth = np.full((H, W), 65535, np.uint16)
    lab = label_of(m, depth)
    assert lab["distance_m"] is None and lab["saturated_px_in_mask"] == 25, lab
    depth[5:7, 5:10] = 3000                                 # 10 px misurati, 15 saturi
    lab = label_of(m, depth)
    assert lab["distance_m"] == 3.0 and lab["saturated_px_in_mask"] == 15
    assert label_of(m, depth, scale=500.0)["distance_m"] == 6.0


def test_depth_a_8_bit_rifiutata():
    root, _, _ = make_session(n=1)
    try:
        path = os.path.join(root, "depth", "000000.png")
        cv2.imwrite(path, np.zeros((H, W), np.uint8))
        expect_error(GroundTruthError, P.read_depth_u16, path)
    finally:
        shutil.rmtree(root)


# --- sessione --------------------------------------------------------------------------

def test_check_session_rifiuta_frame_alterati():
    for sub in ("rgb", "depth", "instance"):
        root, _, _ = make_session()
        try:
            _, _, prov = P.check_session(root)
            assert prov["semantic_sha256"] == P.dir_sha256(os.path.join(root, "semantic"))
            assert [c["check"] for c in prov["conventions"]] == [
                "canale del tag", "formula dell'id", "chiave (tag, id)"]
            path = os.path.join(root, sub, "000004.png")
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img[0, 0] ^= 1
            cv2.imwrite(path, img)
            expect_error(P.PrepareError, P.check_session, root, contains=sub)
        finally:
            shutil.rmtree(root)


def test_check_session_sessione_non_valida_o_incompleta():
    root, _, acceptance = make_session()
    try:
        acc = os.path.join(root, "acceptance.json")
        with open(acc, "w") as f:
            json.dump(dict(acceptance, valid=False, failed=["A8"]), f)
        expect_error(P.PrepareError, P.check_session, root, contains="non accettata")
        os.remove(acc)
        expect_error(P.PrepareError, P.check_session, root, contains="acceptance.json")
        os.remove(os.path.join(root, "session.json"))
        expect_error(P.PrepareError, P.check_session, root, contains="session.json")
    finally:
        shutil.rmtree(root)


def test_numerazione_con_buchi_rifiutata():
    root, _, _ = make_session()
    try:
        os.rename(os.path.join(root, "semantic", "000005.png"),
                  os.path.join(root, "semantic", "000013.png"))
        expect_error(P.PrepareError, P.check_session, root, contains="semantic")
    finally:
        shutil.rmtree(root)


def test_pose_lette_per_indice_non_per_posizione():
    root, _, _ = make_session()
    try:
        path = os.path.join(root, "poses.jsonl")
        with open(path) as f:
            poses = [json.loads(line) for line in f]
        with open(path, "w") as f:
            f.writelines(json.dumps(p) + "\n" for p in reversed(poses))
        got = P.poses_by_index(root, [3, 6])
        assert got[3]["timestamp"] == 3 / 48.0 and got[6]["timestamp"] == 6 / 48.0
        with open(path, "w") as f:
            f.writelines(json.dumps(p) + "\n" for p in poses if p["index"] != 6)
        expect_error(P.PrepareError, P.poses_by_index, root, [3, 6], contains="poses")
    finally:
        shutil.rmtree(root)


# --- pacchetto -----------------------------------------------------------------------

def read_png(pack, sub, k):
    return cv2.imread(os.path.join(pack, sub, f"{k:04d}.png"), cv2.IMREAD_UNCHANGED)


def check_frames_on_disk(pack, labels):
    """Rilegge dal disco frame, GT e labels e li confronta coi valori della sessione."""
    with open(os.path.join(pack, "gt", "labels.jsonl")) as f:
        assert [json.loads(line) for line in f] == labels
    for k, lab in enumerate(labels):
        src = lab["src_index"]
        rgb = cv2.cvtColor(read_png(pack, "frames/rgb", k), cv2.COLOR_BGR2RGB)
        assert tuple(rgb[0, 0]) == TOP_RGB and rgb[1, 0, 0] == 10 + src, (k, rgb[1, 0])
        seg = cv2.cvtColor(read_png(pack, "frames/seg", k), cv2.COLOR_BGR2RGB)
        assert tuple(seg[0, 0]) == tuple(P.CITYSCAPES_PALETTE[POLE])
        assert tuple(seg[FENCE_ROW - ASYM['crop_top'], src % W]) == tuple(P.CITYSCAPES_PALETTE[FENCE])
        depth = read_png(pack, "gt/depth_mm", k)
        assert depth.dtype == np.uint16 and depth[0, 0] == TOP_DEPTH
        mask = read_png(pack, "gt/mask", k)
        assert set(np.unique(mask).tolist()) == {0, 255}
        ys, xs = np.nonzero(mask)
        assert (xs.min(), ys.min()) == (2 + src, 4), (k, xs.min(), ys.min())
        assert (depth[mask > 0] == 3000 + 100 * src).all()
        assert depth[1, 0] == 5000 + 10 * src
        ctl = read_png(pack, "frames/depth", k)
        assert ctl[ys[0], xs[0]] == P.depth_to_log_grayscale(np.array([(3000 + 100 * src) / 1000]))[0]
        assert lab["distance_m"] == (3000 + 100 * src) / 1000
        assert abs(lab["center_distance_m"] - (4.2 + 0.1 * src)) < 1e-9
        assert lab["boss_class"] == "ramo_sporgente" and lab["parts"] == [[VEG, 147]]


def test_pacchetto_riletto_dal_disco():
    root, session, acceptance = make_session(n=16, moving=True)
    out = tempfile.mkdtemp(prefix="packs_")
    try:
        with Format("asym", ASYM) as model:
            pack, labels = P.build_package(root, session, acceptance, model, out, None,
                                           {"session_path": root})
        assert os.path.basename(pack) == "s2r_test__asym" and not os.path.exists(pack + ".partial")
        # finestra A8 0-15: ancorata alla fine, non all'inizio (che darebbe 0..12)
        assert [lab["src_index"] for lab in labels] == [3, 6, 9, 12, 15]
        assert [lab["timestamp"] for lab in labels] == [s / 48.0 for s in (3, 6, 9, 12, 15)]
        check_frames_on_disk(pack, labels)
        with open(os.path.join(pack, "manifest.json")) as f:
            man = json.load(f)
        assert man["resolution"] == [W, H - 8] and man["intrinsics"]["cy"] == 14.0
        assert man["window"]["output_frame_to_src"] == [3, 6, 9, 12, 15]
        assert man["obstacle"]["visible_frames"] == 5 and man["obstacle"]["parts_never_visible"] == []
        for sub in ("rgb", "depth", "instance"):
            assert man["session"][sub + "_sha256"] == session[sub + "_sha256"], sub
        assert man["intrinsics"]["cx"] == 16.0 and man["intrinsics"]["fx"] == 20.0
        assert man["provenance"] == {"session_path": root} and man["encoding"] is None
        on_disk = {os.path.relpath(os.path.join(d, f), pack).replace("\\", "/")
                   for d, _, fs in os.walk(pack) for f in fs}
        assert set(man["files"]) == on_disk - {"manifest.json"}
        for rel, digest in man["files"].items():
            with open(os.path.join(pack, rel), "rb") as f:
                assert hashlib.sha256(f.read()).hexdigest() == digest, rel
        with Format("asym", ASYM) as model:
            expect_error(P.PrepareError, P.build_package, root, session, acceptance, model, out,
                         None, contains="esiste gia'")
    finally:
        shutil.rmtree(root)
        shutil.rmtree(out)


def test_due_formati_dalla_stessa_sessione_non_si_contaminano():
    root, session, acceptance = make_session()
    out = tempfile.mkdtemp(prefix="packs_")
    try:
        cys = []
        for name, fmt in (("asym", ASYM), ("full", FULL)):
            with Format(name, fmt) as model:
                pack, _ = P.build_package(root, session, acceptance, model, out, None)
            with open(os.path.join(pack, "manifest.json")) as f:
                cys.append(json.load(f)["intrinsics"]["cy"])
        assert cys == [14.0, 20.0] and session["intrinsics"]["cy"] == 20.0, cys
    finally:
        shutil.rmtree(root)
        shutil.rmtree(out)


def test_saturi_contati_senza_ritaglio():
    root, session, acceptance = make_session()
    out = tempfile.mkdtemp(prefix="packs_")
    try:
        with Format("full", FULL) as model:
            pack, _ = P.build_package(root, session, acceptance, model, out, None)
        with open(os.path.join(pack, "manifest.json")) as f:
            man = json.load(f)
        assert man["recipe"]["depth_far_filled"] == {"saturated_px": 5 * 2 * W, "zero_px": 0}
    finally:
        shutil.rmtree(root)
        shutil.rmtree(out)


def test_sessione_di_dimensione_diversa_rifiutata():
    root, session, acceptance = make_session()
    out = tempfile.mkdtemp(prefix="packs_")
    try:
        with Format("full", FULL) as model:
            expect_error(P.PrepareError, P.build_package, root, dict(session, width=W + 1),
                         acceptance, model, out, None, contains="formati")
        assert os.listdir(out) == []
        _, actors, _ = P.the_obstacle(session, acceptance)
        parts, _ = P.obstacle_parts(root, actors, [0])
        expect_error(GroundTruthError, P.build_frame, root, 0, parts,
                     dict(FULL, width=W, height=H + 2), 1000.0)
    finally:
        shutil.rmtree(root)
        shutil.rmtree(out)


def run_fake_encode(work, drop):
    """build_package con l'ffmpeg finto; tutto dentro la cartella temporanea work."""
    root, session, acceptance = make_session()
    out = os.path.join(work, "packs")
    os.makedirs(out)
    fake, log = os.path.join(work, "fake_ffmpeg.py"), os.path.join(work, "calls.jsonl")
    with open(fake, "w") as f:
        f.write(FAKE_FFMPEG)
    os.environ["FAKE_FFMPEG_LOG"] = log
    try:
        with Format("asym", ASYM) as model:
            pack, _ = P.build_package(root, session, acceptance, model, out,
                                      [sys.executable, fake] + (["--drop"] if drop else []))
        with open(log) as f:
            return pack, [json.loads(line) for line in f]
    finally:
        shutil.rmtree(root)


def test_codifica_con_ffmpeg_finto():
    work = tempfile.mkdtemp(prefix="enc_")
    try:
        pack, calls = run_fake_encode(work, drop=False)
        assert len(calls) == 4
        for c, args in zip(P.CONTROL_NAMES, calls):
            assert args[args.index("-framerate") + 1] == "16" and "-n" in args
            assert args[-1].endswith(os.path.join("controls", c + ".mp4"))
            assert args[args.index("-i") + 1].endswith(os.path.join("frames", c, "%04d.png"))
        with open(os.path.join(pack, "manifest.json")) as f:
            man = json.load(f)
        assert man["encoding"]["ffmpeg_version"] == "ffmpeg finto 1.0"
        assert "controls/rgb.mp4" in man["files"]
    finally:
        shutil.rmtree(work)


def test_video_con_un_frame_in_meno_ferma_e_pulisce():
    work = tempfile.mkdtemp(prefix="enc_")
    try:
        expect_error(P.PrepareError, run_fake_encode, work, True,
                     contains="decodificati 4 frame")
        assert os.listdir(os.path.join(work, "packs")) == []
    finally:
        shutil.rmtree(work)


def test_un_fallimento_non_lascia_il_partial():
    root, session, acceptance = make_session()
    out = tempfile.mkdtemp(prefix="packs_")
    try:
        with Format("tiny", TINY) as model:
            expect_error(OSError, P.build_package, root, session, acceptance, model, out,
                         ["comando-ffmpeg-inesistente"])
        assert os.listdir(out) == [], os.listdir(out)
    finally:
        shutil.rmtree(root)
        shutil.rmtree(out)


def test_partial_rimasto_da_un_crash_blocca():
    root, session, acceptance = make_session()
    out = tempfile.mkdtemp(prefix="packs_")
    try:
        os.makedirs(os.path.join(out, "s2r_test__full.partial"))
        with Format("full", FULL) as model:
            expect_error(P.PrepareError, P.build_package, root, session, acceptance, model, out,
                         None, contains=".partial")
    finally:
        shutil.rmtree(root)
        shutil.rmtree(out)


def test_pezzo_del_gruppo_mai_visibile_nel_manifest():
    actors = [{"instance_id": 147, "boss_class": "ramo_sporgente", "group": "g", "part": "ramo"},
              {"instance_id": 999, "boss_class": "ramo_sporgente", "group": "g",
               "part": "fantasma"}]
    root, session, acceptance = make_session(actors=actors)
    out = tempfile.mkdtemp(prefix="packs_")
    try:
        with Format("full", FULL) as model:
            pack, _ = P.build_package(root, session, acceptance, model, out, None)
        with open(os.path.join(pack, "manifest.json")) as f:
            man = json.load(f)
        assert man["obstacle"]["parts_never_visible"] == ["fantasma"]
        assert [p["instance_id"] for p in man["obstacle"]["parts"]] == [147]
    finally:
        shutil.rmtree(root)
        shutil.rmtree(out)


def test_ostacolo_assente_nei_primi_frame_e_visible_frames():
    # ostacolo assente nei frame 0 e 3: i primi due della clip [0, 3, 6, 9, 12]
    root, session, acceptance = make_session(absent=(0, 3))
    out = tempfile.mkdtemp(prefix="packs_")
    try:
        _, actors, _ = P.the_obstacle(session, acceptance)
        parts, unseen = P.obstacle_parts(root, actors, [0, 3, 6, 9, 12])
        assert [p["instance_id"] for p in parts] == [147] and not unseen
        with Format("full", FULL) as model:
            pack, labels = P.build_package(root, session, acceptance, model, out, None)
        with open(os.path.join(pack, "manifest.json")) as f:
            man = json.load(f)
        assert [lab["visible"] for lab in labels] == [False, False, True, True, True]
        assert man["obstacle"]["visible_frames"] == 3
    finally:
        shutil.rmtree(root)
        shutil.rmtree(out)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok     {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FALLITO {name} {exc}")
    print(f"\n{len(tests)} test, {failed} falliti")
    sys.exit(1 if failed else 0)
