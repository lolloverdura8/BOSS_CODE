# prepare_controls.py
#
# Fase 2 del ramo Sim2Real: da una sessione registrata da carla_capture.py produce,
# per ciascun modello video-to-video, un PACCHETTO con
#   - i video di controllo che il modello legge (rgb, depth, edge, seg), gia' nel
#     suo formato: Cosmos-Transfer2.5 93 frame a 16 fps 1280x720, Wan2.2-Fun-5B
#     121 frame a 24 fps 1280x704;
#   - la ground truth allineata frame per frame alla clip che uscira' (maschera
#     dell'ostacolo, depth metrica, box, distanza), che la Fase 4 usa per decidere
#     se l'etichetta di CARLA vale ancora per l'immagine generata.
#
# I controlli replicano la ricetta ufficiale CARLA -> Cosmos (carla-simulator/carla,
# branch ue4-dev, PythonAPI/examples/nvidia/cosmos/client/carla_cosmos_gen.py e
# filter_semantic_classes.yaml, licenza MIT, (c) NVIDIA 2025), con due aggiunte
# dichiarate: la maschera dell'ostacolo entra nell'edge (la ricetta non da' bordi a
# Vegetation, Building e Pedestrian) e la depth satura va a 1000 m (la nostra
# satura a 65,5 m e il cielo sarebbe grigio invece che nero).
#
# Non importa carla e non usa la GPU: legge e scrive PNG un frame alla volta. Gli
# mp4 li scrive l'ffmpeg di WSL, cosi' Smart App Control non entra in gioco.
#   python prepare_controls.py --all-s2r --model both
import argparse
import glob
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime

import cv2
import numpy as np

from acceptance import (
    MIN_VISIBLE_PX,
    load_poses,
    load_session,
    scan_instances,
    tracked_objects,
)
from carla_export import distance_from_depth
from carla_gt import (
    DEPTH_SATURATED_U16,
    GroundTruthError,
    instance_mask,
    read_instance,
    read_rgb,
    read_semantic,
    verify_conventions,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "outputs", "s2r_packs")

# Formato d'ingresso dei due modelli (Fase 0 del piano, fonti ufficiali del 02/10).
# Il passo e' quello sui 48 Hz della cattura: nessun frame e' interpolato. Wan vuole
# 704 righe: se ne tolgono 8 sopra e 8 sotto, senza ridimensionare.
MODEL_FORMATS = {
    "cosmos": {"fps": 16, "n_frames": 93, "crop_top": 0, "crop_bottom": 0},
    "wanfun": {"fps": 24, "n_frames": 121, "crop_top": 8, "crop_bottom": 8},
}
# Entrambi i formati partono da questa cattura: una sessione 1008x756 non diventa
# un pacchetto "Cosmos" senza che nessuno se ne accorga.
SOURCE_SIZE = (1280, 720)
# Quota minima di pixel che il tag dominante di un id deve avere sui frame della clip.
MIN_TAG_PURITY = 0.9
CONTROL_NAMES = ("rgb", "depth", "edge", "seg")

RECIPE_SOURCE = ("https://github.com/carla-simulator/carla/blob/ue4-dev/PythonAPI/examples/"
                 "nvidia/cosmos/client/carla_cosmos_gen.py")
DEPTH_NEAR_M, DEPTH_FAR_M = 0.01, 1000.0
GAUSSIAN_KERNEL, GAUSSIAN_SIGMA = (5, 5), 1.0
CANNY_T1, CANNY_T2 = 100, 200

# canny_classes di filter_semantic_classes.yaml, copiata cosi' com'e', varianti BGR
# comprese. [140,0,0] e' "vehicles" ma il colore Car di CARLA e' [0,0,142]: con la
# ricetta le auto restano senza bordi. Non si corregge, si segnala nel report.
CANNY_CLASSES = [
    [128, 64, 128], [244, 35, 232], [232, 35, 244], [81, 0, 81], [50, 234, 157],
    [55, 90, 80], [80, 90, 55], [220, 220, 0], [0, 220, 220], [250, 170, 30],
    [30, 170, 250], [153, 153, 153], [140, 0, 0],
]

# Palette CityScapes di CARLA 0.9.16 (ref_sensors, tag 0..28), in RGB.
CITYSCAPES_PALETTE = np.array([
    (0, 0, 0), (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35),
    (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
    (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
    (110, 190, 160), (170, 120, 50), (55, 90, 80), (45, 60, 150), (157, 234, 50),
    (81, 0, 81), (150, 100, 100), (230, 150, 140), (180, 165, 180),
], dtype=np.uint8)
TAG_RIDER, TAG_BICYCLE = 13, 19

# Con che tag si dipinge l'ostacolo nella seg, per classe BOSS e pezzo. Assente =
# resta il suo tag: il ramo e' gia' Vegetation, l'insegna gia' TrafficSign, e per
# l'ostacolo sospeso generico CityScapes non ha una classe. Il monopattino non esiste
# in CityScapes: conducente Rider, mezzo Bicycle (scelta confermata il 12/10).
SEG_RECOLOR = {"monopattino": {"conducente": TAG_RIDER, "*": TAG_BICYCLE}}

FFMPEG_WSL_BIN = ("/home/lollo/or4-cogvideox/.venv/lib/python3.12/site-packages/"
                  "imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2")
FFMPEG_WSL = ["wsl.exe", "-d", "Ubuntu-24.04", "-e", FFMPEG_WSL_BIN]
ENCODE_ARGS = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", "-threads", "4"]
PREVIEW_THUMB_W = 320
# I moduli che decidono i pixel e la GT del pacchetto: il loro sha va nel manifest,
# perche' finche' non sono committati il commit da solo non li identifica.
CODE_MODULES = ("prepare_controls.py", "carla_gt.py", "acceptance.py", "carla_export.py",
                "boss_classes.py")


class PrepareError(RuntimeError):
    """La sessione non puo' diventare un pacchetto senza produrre dati sbagliati."""


# --- ricetta CARLA -> Cosmos ---------------------------------------------------

def depth_to_log_grayscale(depth_map, near_clip=DEPTH_NEAR_M, far_clip=DEPTH_FAR_M,
                           inverted_depth=True):
    """Verbatim dalla ricetta, salvo il ritorno: ndarray invece di PIL.Image."""
    clipped = np.clip(depth_map, near_clip, far_clip)
    log_depth = np.log(clipped)
    norm_log = (log_depth - np.log(near_clip)) / (np.log(far_clip) - np.log(near_clip))
    if inverted_depth:
        norm_log = 1.0 - norm_log
    gray_img = (norm_log * 255).astype(np.uint8)
    return gray_img


def masked_edges_from_semseg(rgb_img, semseg_img, classes, *, gaussian_kernel=GAUSSIAN_KERNEL,
                             gaussian_sigma=GAUSSIAN_SIGMA, canny_thresh1=CANNY_T1,
                             canny_thresh2=CANNY_T2, extra_mask=None):
    """Verbatim dalla ricetta, piu' extra_mask (aggiunta OR4): pixel tenuti oltre alle classi."""
    blurred_rgb = cv2.GaussianBlur(rgb_img, gaussian_kernel, gaussian_sigma)
    mask = np.zeros(semseg_img.shape[:2], dtype=np.uint8)
    for color in classes:
        lower = np.array(color, dtype=np.uint8)
        upper = np.array(color, dtype=np.uint8)
        mask |= cv2.inRange(semseg_img, lower, upper)
    mask_bool = mask.astype(bool)
    if extra_mask is not None:
        mask_bool |= extra_mask
    masked_rgb = np.zeros_like(rgb_img)
    masked_rgb[mask_bool] = blurred_rgb[mask_bool]
    gray = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, canny_thresh1, canny_thresh2)
    return masked_rgb, edges


def depth_control(depth_u16, depth_scale):
    """(grigio log-inverso, pixel saturi, pixel a zero).

    Saturi e zeri vanno a DEPTH_FAR_M: 65535 vuol dire "oltre 65,5 m", e passato
    cosi' alla ricetta darebbe al cielo un grigio ~60 invece del nero di CARLA.
    """
    far = (depth_u16 >= DEPTH_SATURATED_U16) | (depth_u16 == 0)
    meters = depth_u16.astype(np.float32) / float(depth_scale)
    meters[far] = DEPTH_FAR_M
    n_sat = int((depth_u16 >= DEPTH_SATURATED_U16).sum())
    return depth_to_log_grayscale(meters), n_sat, int((depth_u16 == 0).sum())


def cityscapes(tag):
    """Immagine RGB CityScapes dai tag semantici; un tag fuori da 0..28 e' un errore."""
    if int(tag.max()) >= len(CITYSCAPES_PALETTE):
        raise GroundTruthError(f"tag semantico {int(tag.max())} fuori palette"
                               f" (0..{len(CITYSCAPES_PALETTE) - 1})")
    return CITYSCAPES_PALETTE[tag]


def seg_tag_for(actor):
    """Tag con cui dipingere un pezzo dell'ostacolo nella seg, o None se resta il suo."""
    rule = SEG_RECOLOR.get(actor["boss_class"])
    if not rule:
        return None
    return rule.get(actor.get("part"), rule.get("*"))


# --- finestra e ostacolo ------------------------------------------------------

def clip_sources(a8_window, n_frames, stride):
    """Indici sorgente della clip: n_frames presi ogni stride, finendo sulla fine di A8."""
    a8_start, end = a8_window
    span = (n_frames - 1) * stride
    start = end - span
    if start < a8_start:
        raise PrepareError(f"finestra A8 {list(a8_window)} troppo corta: servono {span + 1}"
                           f" frame sorgente, ne ha {end - a8_start + 1}")
    return list(range(start, end + 1, stride))


def model_stride(session, fps):
    session_fps = 1.0 / session["fixed_delta_seconds"]
    stride = session_fps / fps
    if abs(stride - round(stride)) > 1e-6:
        raise PrepareError(f"{session_fps:.3f} Hz non e' multiplo di {fps} fps:"
                           " servirebbe interpolare")
    return round(stride)


def the_obstacle(session, acceptance):
    """(nome, attori, finestra A8) dell'unico oggetto tracciato della sessione."""
    objects = tracked_objects(session)
    if len(objects) != 1:
        raise PrepareError(f"atteso 1 oggetto tracciato, trovati {len(objects)}:"
                           f" {sorted(objects)}")
    name = next(iter(objects))
    window = (acceptance.get("windows") or {}).get(name)
    if not window:
        raise PrepareError(f"acceptance.json non ha la finestra A8 di {name}")
    actors = [a for a in session["spawned_actors"] if a["instance_id"] in objects[name]["ids"]]
    return name, actors, window


def obstacle_parts(session_dir, actors, sources):
    """Coppie (tag, id) dei pezzi, col tag dominante sui frame della clip.

    La coppia e non il solo id: l'id e' a 16 bit e un oggetto della mappa puo'
    condividerlo (carla_gt.instance_mask). Un pezzo mai visibile nella clip resta
    fuori e viene elencato nel manifest.
    """
    names = [f"{s:06d}.png" for s in sources]
    counts = scan_instances(session_dir, names, [a["instance_id"] for a in actors])
    parts, unseen = [], []
    for a in actors:
        total = np.zeros(256, dtype=np.int64)
        for frame in counts:
            if a["instance_id"] in frame:
                total += frame[a["instance_id"]]
        if not total.any():
            unseen.append(a.get("part") or a["instance_id"])
            continue
        purity = total.max() / total.sum()
        if purity < MIN_TAG_PURITY:
            # L'id e' condiviso con un oggetto della mappa di altro tag: il dominante
            # potrebbe essere quello dell'oggetto sbagliato.
            raise PrepareError(f"id {a['instance_id']}: tag dominante solo al {purity:.0%}"
                               f" dei pixel, {dict(enumerate(total.tolist()))}")
        parts.append({"instance_id": a["instance_id"], "carla_tag": int(total.argmax()),
                      "part": a.get("part"), "boss_class": a["boss_class"],
                      "seg_tag": seg_tag_for(a)})
    if not parts:
        raise PrepareError("l'ostacolo non e' mai visibile nei frame della clip")
    return parts, unseen


# --- un frame ------------------------------------------------------------------

def read_depth_u16(path):
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype != np.uint16 or raw.ndim != 2:
        raise GroundTruthError(f"depth assente o non uint16 a un canale: {path}")
    return raw


def frame_label(mask, depth_u16, depth_scale, pose, parts):
    """Riga di labels.jsonl: box ricalcolata sulla maschera gia' ritagliata."""
    area = int(mask.sum())
    label = {"area_px": area, "visible": area >= MIN_VISIBLE_PX, "bbox": None,
             "truncated": None, "distance_m": None}
    if area:
        rows, cols = np.nonzero(mask)
        y0, y1, x0, x1 = int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())
        h, w = mask.shape
        label["bbox"] = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
        label["truncated"] = bool(x0 == 0 or y0 == 0 or x1 == w - 1 or y1 == h - 1)
        # distance_from_depth scarta solo gli zeri: un 65535 ("oltre 65,5 m") entrerebbe
        # nella mediana come misura. Qui i saturi restano fuori; senza pixel validi la
        # distanza e' None e non 65,535.
        measured = mask & (depth_u16 < DEPTH_SATURATED_U16)
        label["saturated_px_in_mask"] = int((mask & ~measured).sum())
        label["distance_m"] = distance_from_depth(depth_u16, measured, label["bbox"],
                                                  "median_mask", depth_scale)
    tracked = pose.get("tracked_distances_m") or {}
    centers = [tracked[str(p["instance_id"])] for p in parts if str(p["instance_id"]) in tracked]
    label["center_distance_m"] = min(centers) if centers else None
    return label


def build_frame(session_dir, src, parts, fmt, depth_scale):
    """Controlli e GT di un frame sorgente, gia' ritagliati al formato del modello."""
    path = {sub: os.path.join(session_dir, sub, f"{src:06d}.png")
            for sub in ("rgb", "depth", "instance", "semantic")}
    rows = slice(fmt["crop_top"], None if not fmt["crop_bottom"] else -fmt["crop_bottom"])
    rgb = read_rgb(path["rgb"])
    if rgb.shape[:2] != (fmt["height"], fmt["width"]):
        raise GroundTruthError(f"{path['rgb']} e' {rgb.shape[1]}x{rgb.shape[0]},"
                               f" session.json dice {fmt['width']}x{fmt['height']}")
    rgb = rgb[rows]
    depth_u16 = read_depth_u16(path["depth"])[rows]
    tag, ids = (a[rows] for a in read_instance(path["instance"]))
    semantic = read_semantic(path["semantic"])[rows]

    seg_carla = cityscapes(semantic)
    seg = seg_carla.copy()
    mask = np.zeros(semantic.shape, dtype=bool)
    for p in parts:
        m = instance_mask(tag, ids, p["carla_tag"], p["instance_id"])
        mask |= m
        if p["seg_tag"] is not None:
            seg[m] = CITYSCAPES_PALETTE[p["seg_tag"]]
    _, edge = masked_edges_from_semseg(rgb, seg_carla, CANNY_CLASSES, extra_mask=mask)
    depth_gray, n_sat, n_zero = depth_control(depth_u16, depth_scale)
    return {"rgb": rgb, "depth": depth_gray, "edge": edge, "seg": seg,
            "mask": mask, "depth_u16": depth_u16, "n_sat": n_sat, "n_zero": n_zero}


# --- scrittura -------------------------------------------------------------------

def write_png(path, img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(path, img):
        raise OSError(f"scrittura fallita: {path}")


def write_frame(tmp_dir, k, fr):
    name = f"{k:04d}.png"
    for c in CONTROL_NAMES:
        write_png(os.path.join(tmp_dir, "frames", c, name), fr[c])
    write_png(os.path.join(tmp_dir, "gt", "mask", name), fr["mask"].astype(np.uint8) * 255)
    write_png(os.path.join(tmp_dir, "gt", "depth_mm", name), fr["depth_u16"])


def preview_row(fr, label):
    """Striscia rgb | depth | edge | seg di un frame, col contorno della maschera GT."""
    rgb = fr["rgb"].copy()
    contours, _ = cv2.findContours(fr["mask"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, (255, 0, 255), 3)
    tiles = [rgb, cv2.cvtColor(fr["depth"], cv2.COLOR_GRAY2RGB),
             cv2.cvtColor(fr["edge"], cv2.COLOR_GRAY2RGB), fr["seg"]]
    h = round(rgb.shape[0] * PREVIEW_THUMB_W / rgb.shape[1])
    row = np.hstack([cv2.resize(t, (PREVIEW_THUMB_W, h), interpolation=cv2.INTER_AREA)
                     for t in tiles])
    dist = label["distance_m"]
    where = f"{dist:.2f} m" if dist is not None else "non visibile"
    text = f"out {label['out_index']}  src {label['src_index']}  {where}"
    cv2.putText(row, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return row


def to_ffmpeg_path(path, ffmpeg_cmd):
    """Percorso Windows -> /mnt/<disco>/... quando ffmpeg gira in WSL."""
    path = os.path.abspath(path)
    if os.path.basename(ffmpeg_cmd[0]).lower() not in ("wsl", "wsl.exe"):
        return path
    drive, rest = os.path.splitdrive(path)
    return "/mnt/{}{}".format(drive.rstrip(":").lower(), rest.replace("\\", "/"))


def decoded_frames(path):
    """(frame decodificati, larghezza, altezza) leggendo davvero il video.

    CAP_PROP_FRAME_COUNT viene dall'intestazione: un mp4 con il centro rovinato ma
    l'indice intatto dichiara ancora tutti i frame (verificato in revisione il 12/10).
    """
    cap = cv2.VideoCapture(path)
    n, size = 0, (0, 0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n, size = n + 1, (frame.shape[1], frame.shape[0])
    cap.release()
    return n, size[0], size[1]


def ffmpeg_version(ffmpeg_cmd):
    try:
        out = subprocess.run(ffmpeg_cmd + ["-version"], check=True, capture_output=True,
                             text=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"non rilevata: {exc}"
    return out.splitlines()[0] if out else "non rilevata"


def encode_controls(tmp_dir, fps, n_frames, size, ffmpeg_cmd):
    """frames/<controllo>/%04d.png -> controls/<controllo>.mp4, poi lo decodifica tutto."""
    os.makedirs(os.path.join(tmp_dir, "controls"))
    for c in CONTROL_NAMES:
        src = os.path.join(tmp_dir, "frames", c, "%04d.png")
        out = os.path.join(tmp_dir, "controls", c + ".mp4")
        cmd = (ffmpeg_cmd + ["-hide_banner", "-loglevel", "error", "-n",
                             "-framerate", str(fps), "-i", to_ffmpeg_path(src, ffmpeg_cmd)]
               + ENCODE_ARGS + [to_ffmpeg_path(out, ffmpeg_cmd)])
        subprocess.run(cmd, check=True)
        got = decoded_frames(out)
        if got != (n_frames, *size):
            raise PrepareError(f"{out}: decodificati {got[0]} frame {got[1]}x{got[2]},"
                               f" attesi {n_frames} {size[0]}x{size[1]}")
    return {"command": ffmpeg_cmd, "args": ENCODE_ARGS, "fps": fps,
            "ffmpeg_version": ffmpeg_version(ffmpeg_cmd)}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_hashes(root):
    out = {}
    for dirpath, _, files in os.walk(root):
        for name in sorted(files):
            full = os.path.join(dirpath, name)
            out[os.path.relpath(full, root).replace("\\", "/")] = sha256_file(full)
    return dict(sorted(out.items()))


def code_version():
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, check=True,
                             capture_output=True, text=True).stdout.strip()
        # Tutta la cartella CARLA: anche carla_gt, acceptance e carla_export cambiano la GT.
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "*.py"],
                               cwd=HERE, check=True, capture_output=True, text=True).stdout.split()
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "error": str(exc)}
    return {"commit": rev, "modified_py": [p for p in dirty if p.endswith(".py")],
            "sha256": {m: sha256_file(os.path.join(HERE, m)) for m in CODE_MODULES},
            "python": sys.version.split()[0], "cv2": cv2.__version__, "numpy": np.__version__}


def dir_sha256(directory):
    """sha256 dei PNG concatenati in ordine, come carla_capture per rgb/depth/instance."""
    h = hashlib.sha256()
    for path in sorted(glob.glob(os.path.join(directory, "*.png"))):
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


# --- sessione --------------------------------------------------------------------

def check_frame_numbering(session_dir, n_frames):
    """I frame devono essere 000000..n-1 senza buchi in ogni flusso.

    Le finestre di A8 sono posizioni nell'elenco ordinato dei file; qui si usano
    come numeri di file. Coincidono solo se la numerazione e' contigua da 0.
    """
    expected = [f"{i:06d}.png" for i in range(n_frames)]
    for sub in ("rgb", "depth", "semantic", "instance"):
        names = sorted(n for n in os.listdir(os.path.join(session_dir, sub)) if n.endswith(".png"))
        if names != expected:
            raise PrepareError(f"{sub}/ non e' numerato 000000..{n_frames - 1:06d} senza buchi")


def check_session(session_dir):
    """Passo 0: accettazione valida, convenzioni verificate, frame uguali alla cattura.

    Restituisce (session, acceptance, provenance): provenance finisce nel manifest,
    con gli hash anche dei flussi che session.json non copre (semantic, pose).
    """
    session = load_session(session_dir)
    if session is None:
        raise PrepareError(f"session.json assente in {session_dir}")
    acc_path = os.path.join(session_dir, "acceptance.json")
    if not os.path.exists(acc_path):
        raise PrepareError("acceptance.json assente: rieseguire acceptance.py")
    with open(acc_path) as f:
        acceptance = json.load(f)
    if not acceptance.get("valid"):
        raise PrepareError("sessione non accettata: {}".format(acceptance.get("failed")))
    check_frame_numbering(session_dir, session["n_frames_written"])
    conventions = verify_conventions(session_dir)
    for sub in ("rgb", "depth", "instance"):
        if dir_sha256(os.path.join(session_dir, sub)) != session[sub + "_sha256"]:
            raise PrepareError(f"{sub}/ non coincide con lo sha256 di session.json")
    provenance = {
        "session_path": os.path.abspath(session_dir),
        "conventions": [{"check": n, "ok": ok, "detail": d} for n, ok, d in conventions],
        "semantic_sha256": dir_sha256(os.path.join(session_dir, "semantic")),
        "poses_sha256": sha256_file(os.path.join(session_dir, "poses.jsonl")),
        "session_json_sha256": sha256_file(os.path.join(session_dir, "session.json")),
        "acceptance_json_sha256": sha256_file(acc_path),
    }
    for name, ok, detail in conventions:
        if ok is None:
            print(f"  convenzione non verificabile ({name}): {detail}")
    return session, acceptance, provenance


def build_package(session_dir, session, acceptance, model, out_root, ffmpeg_cmd,
                  provenance=None):
    """Scrive il pacchetto in <nome>.partial e lo rinomina solo a lavoro finito.

    Se qualcosa fallisce, rinomina compresa, il .partial viene tolto: un resto a
    meta' bloccherebbe ogni riesecuzione con "esiste gia'".
    """
    pack_dir = os.path.join(out_root, "{}__{}".format(session["scenario"], model))
    tmp_dir = pack_dir + ".partial"
    for d in (pack_dir, tmp_dir):
        if os.path.exists(d):
            raise PrepareError(f"{d} esiste gia': cancellarlo a mano per rifarlo")
    try:
        labels = fill_package(tmp_dir, session_dir, session, acceptance, model, ffmpeg_cmd,
                              provenance)
        os.rename(tmp_dir, pack_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return pack_dir, labels


def fill_package(tmp_dir, session_dir, session, acceptance, model, ffmpeg_cmd, provenance):
    if (session["width"], session["height"]) != SOURCE_SIZE:
        raise PrepareError(f"sessione {session['width']}x{session['height']}, i formati dei"
                           f" modelli partono da {SOURCE_SIZE[0]}x{SOURCE_SIZE[1]}")
    fmt = dict(MODEL_FORMATS[model], width=session["width"], height=session["height"])
    stride = model_stride(session, fmt["fps"])
    name, actors, a8 = the_obstacle(session, acceptance)
    sources = clip_sources(a8, fmt["n_frames"], stride)
    parts, unseen = obstacle_parts(session_dir, actors, sources)

    for sub in ["frames/" + c for c in CONTROL_NAMES] + ["gt/mask", "gt/depth_mm"]:
        os.makedirs(os.path.join(tmp_dir, sub))
    labels, n_sat, n_zero = write_frames(tmp_dir, session_dir, session, parts, fmt, sources)

    out_size = (fmt["width"], fmt["height"] - fmt["crop_top"] - fmt["crop_bottom"])
    encoding = None if ffmpeg_cmd is None else encode_controls(
        tmp_dir, fmt["fps"], fmt["n_frames"], out_size, ffmpeg_cmd)
    manifest = make_manifest(session_dir, session, model, fmt, stride, name, a8, sources,
                             parts, unseen, labels, n_sat, n_zero, encoding)
    manifest["provenance"] = provenance
    # Gli hash coprono ogni file del pacchetto tranne manifest.json, che li contiene.
    manifest["files"] = file_hashes(tmp_dir)
    with open(os.path.join(tmp_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return labels


def poses_by_index(session_dir, sources):
    """Pose dei frame sorgente per indice, non per posizione nel file: un buco non le sposta."""
    poses = {p["index"]: p for p in load_poses(session_dir)}
    missing = [s for s in sources if s not in poses]
    if missing:
        raise PrepareError(f"poses.jsonl senza i frame {missing[:5]}")
    return poses


def write_frames(tmp_dir, session_dir, session, parts, fmt, sources):
    """Frame, GT e anteprima. Restituisce (labels, pixel saturi, pixel a zero)."""
    poses = poses_by_index(session_dir, sources)
    labels, previews, n_sat, n_zero = [], [], 0, 0
    preview_at = {round(i * (len(sources) - 1) / 3) for i in range(4)}
    for k, src in enumerate(sources):
        fr = build_frame(session_dir, src, parts, fmt, session["depth_scale"])
        write_frame(tmp_dir, k, fr)
        label = {"out_index": k, "src_index": src, "timestamp": poses[src]["timestamp"]}
        label.update(frame_label(fr["mask"], fr["depth_u16"], session["depth_scale"],
                                 poses[src], parts))
        label.update({"boss_class": parts[0]["boss_class"],
                      "parts": [[p["carla_tag"], p["instance_id"]] for p in parts]})
        labels.append(label)
        n_sat, n_zero = n_sat + fr["n_sat"], n_zero + fr["n_zero"]
        if k in preview_at:
            previews.append(preview_row(fr, label))
    with open(os.path.join(tmp_dir, "gt", "labels.jsonl"), "w") as f:
        f.writelines(json.dumps(lab) + "\n" for lab in labels)
    write_png(os.path.join(tmp_dir, "preview.png"), np.vstack(previews))
    return labels, n_sat, n_zero


def make_manifest(session_dir, session, model, fmt, stride, name, a8, sources, parts, unseen,
                  labels, n_sat, n_zero, encoding):
    height = session["height"] - fmt["crop_top"] - fmt["crop_bottom"]
    intr = dict(session["intrinsics"])
    intr["cy"] = intr["cy"] - fmt["crop_top"]
    return {
        "model": model, "fps": fmt["fps"], "n_frames": len(sources), "stride": stride,
        "resolution": [session["width"], height],
        "crop_rows": {"top": fmt["crop_top"], "bottom": fmt["crop_bottom"]},
        "intrinsics": intr,
        "session": {"dir": os.path.basename(os.path.normpath(session_dir)),
                    "scenario": session["scenario"], "code_version": session.get("code_version"),
                    "rgb_sha256": session["rgb_sha256"], "depth_sha256": session["depth_sha256"],
                    "instance_sha256": session["instance_sha256"]},
        "window": {"a8": list(a8), "src_start": sources[0], "src_end": sources[-1],
                   "anchor": "fine della finestra A8",
                   # frame k dell'uscita del modello = frame sorgente output_frame_to_src[k]
                   "output_frame_to_src": sources},
        "obstacle": {"name": name, "parts": parts, "parts_never_visible": unseen,
                     "visible_frames": sum(1 for lab in labels if lab["visible"])},
        "recipe": {"source": RECIPE_SOURCE, "depth_clip_m": [DEPTH_NEAR_M, DEPTH_FAR_M],
                   "depth_inverted": True, "gaussian": [list(GAUSSIAN_KERNEL), GAUSSIAN_SIGMA],
                   "canny": [CANNY_T1, CANNY_T2], "canny_classes": CANNY_CLASSES,
                   "edge_extra": "maschera dell'ostacolo (aggiunta OR4)",
                   "depth_far_filled": {"saturated_px": n_sat, "zero_px": n_zero},
                   "seg_recolor": SEG_RECOLOR},
        "gt": {"distance_criterion": "median_mask", "visible_min_px": MIN_VISIBLE_PX,
               "depth_unit": "mm", "depth_saturated_u16": DEPTH_SATURATED_U16,
               "mask_and_bbox_on": "frame ritagliato: visible e truncated possono"
                                   " differire da A8 e da carla_export, che vedono 720 righe"},
        "encoding": encoding,
        "code_version": code_version(),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pacchetti di controllo e GT per Cosmos-Transfer2.5 e Wan2.2-Fun.")
    parser.add_argument("--session", action="append", default=[],
                        help="cartella di una sessione (ripetibile)")
    parser.add_argument("--all-s2r", action="store_true", help="tutte le outputs/s2r_*")
    parser.add_argument("--model", choices=sorted(MODEL_FORMATS) + ["both"], default="both")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--ffmpeg", default=None,
                        help="comando ffmpeg (default: quello di WSL in ~/or4-cogvideox)")
    parser.add_argument("--no-encode", action="store_true", help="solo PNG, niente mp4")
    args = parser.parse_args()

    sessions = list(args.session)
    if args.all_s2r:
        sessions += sorted(d for d in glob.glob(os.path.join(HERE, "outputs", "s2r_*"))
                           if os.path.isdir(d) and not d.endswith("s2r_packs"))
    if not sessions:
        sys.exit("indicare --session o --all-s2r")
    models = sorted(MODEL_FORMATS) if args.model == "both" else [args.model]
    # posix=False su Windows: in modo POSIX shlex mangerebbe i backslash di C:\...
    ffmpeg_cmd = None if args.no_encode else (
        shlex.split(args.ffmpeg, posix=os.name != "nt") if args.ffmpeg else FFMPEG_WSL)
    os.makedirs(args.out, exist_ok=True)

    for session_dir in sessions:
        print(f"\n{session_dir}")
        session, acceptance, provenance = check_session(session_dir)
        for model in models:
            pack, labels = build_package(session_dir, session, acceptance, model, args.out,
                                         ffmpeg_cmd, provenance)
            vis = [lab for lab in labels if lab["visible"]]
            dists = [lab["distance_m"] for lab in vis if lab["distance_m"] is not None] or [
                float("nan")]
            print(f"  {model:<7} {os.path.basename(pack)}: {len(labels)} frame, ostacolo"
                  f" visibile in {len(vis)}, distanza {min(dists):.1f}-{max(dists):.1f} m")


if __name__ == "__main__":
    try:
        main()
    except (PrepareError, GroundTruthError, subprocess.CalledProcessError) as exc:
        sys.exit(f"ERRORE: {exc}")
