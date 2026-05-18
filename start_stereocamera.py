import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from thop import profile as thop_profile
from datetime import datetime
from pathlib import Path

# ---------- Config ----------
WIDTH, HEIGHT, FPS = 640, 480, 30
YOLO_MODEL = "yolov8n.pt"
CONF_THRES = 0.4
MIN_VALID_DEPTH_M = 0.25
MAX_VALID_DEPTH_M = 8.0

# Classi COCO da ignorare (ID numerici) — oggetti piccoli o non rilevanti per mobilità
# Classi mantenute: person (0), bicycle (1), car (2), motorcycle (3), bus (5), truck (7),
#                   bench (13), sports ball (32), chair (56), couch (57), bed (59),
#                   dining table (60), toilet (61), sink(71), refrigerator (72),vase (75), door (no COCO), stairs (no COCO)
EXCLUDED_CLASSES = {
    14,  # bird
    15,  # cat
    16,  # dog
    17,  # horse
    18,  # sheep
    19,  # cow
    20,  # elephant
    21,  # bear
    22,  # zebra
    23,  # giraffe
    24,  # backpack
    25,  # umbrella
    26,  # handbag
    27,  # tie
    28,  # suitcase
    29,  # frisbee
    30,  # skis
    31,  # snowboard
    33,  # kite
    34,  # baseball bat
    35,  # baseball glove
    36,  # skateboard
    37,  # surfboard
    38,  # tennis racket
    39,  # bottle
    40,  # wine glass
    41,  # cup
    42,  # fork
    43,  # knife
    44,  # spoon
    45,  # bowl
    46,  # banana
    47,  # apple
    48,  # sandwich
    49,  # orange
    50,  # broccoli
    51,  # carrot
    52,  # hot dog
    53,  # pizza
    54,  # donut
    55,  # cake
    62,  # tv
    63,  # laptop
    64,  # mouse
    65,  # remote
    66,  # keyboard
    67,  # cell phone
    68,  # microwave
    69,  # oven
    70,  # toaster
    73,  # book
    74,  # clock
    76,  # scissors
    77,  # teddy bear
    78,  # hair drier
    79,  # toothbrush
}

# ---------- Output dirs ----------
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = Path(__file__).parent / "recordings" / ts
frames_dir = out_dir / "frames"
frames_dir.mkdir(parents=True, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# video visualizzabile (con bbox disegnate)
writer_vis = cv2.VideoWriter(str(out_dir / "video_vis.mp4"), fourcc, FPS, (WIDTH, HEIGHT))
# video grezzo (frame RGB puliti, senza bbox — per training)
writer_raw = cv2.VideoWriter(str(out_dir / "video_raw.mp4"), fourcc, FPS, (WIDTH, HEIGHT))

print(f"[INFO] Salvataggio in: {out_dir}")

# ---------- Device selection ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Inference device: {device}")

# ---------- YOLO init ----------
model = YOLO(YOLO_MODEL)
model.to(device)
_ = model.predict(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8), device=device, verbose=False)

# ---------- GFLOPs ----------
_dummy = torch.zeros(1, 3, HEIGHT, WIDTH).to(device)
_flops, _params = thop_profile(model.model.eval(), inputs=(_dummy,), verbose=False)
print(f"[INFO] GFLOPs per frame: {_flops / 1e9:.3f} | Params: {_params / 1e6:.2f} M")

# ---------- RealSense pipeline ----------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"[INFO] Depth scale: {depth_scale}")

align = rs.align(rs.stream.color)

spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 2)
spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial.set_option(rs.option.filter_smooth_delta, 20)
spatial.set_option(rs.option.holes_fill, 3)

hole_filling = rs.hole_filling_filter()

color_buf = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
depth_buf = np.empty((HEIGHT, WIDTH), dtype=np.uint16)
frame_idx = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_frame = spatial.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)

        np.copyto(color_buf, np.asarray(color_frame.get_data()))
        np.copyto(depth_buf, np.asarray(depth_frame.get_data()))

        # salva frame grezzo (prima di disegnare bbox) per training
        raw_frame = color_buf.copy()
        writer_raw.write(raw_frame)
        # ogni 30 frame salva anche il singolo JPEG grezzo (per annotazione manuale)
        if frame_idx % 30 == 0:
            cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.jpg"), raw_frame)

        # ---------- YOLO inference ----------
        results = model.predict(
            color_buf,
            device=device,
            half=(device == "cuda"),
            conf=CONF_THRES,
            verbose=False,
        )[0]

        # ---------- Distance extraction + filtro classi ----------
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy().astype(np.int32)
            confs = results.boxes.conf.cpu().numpy()
            cls_ids = results.boxes.cls.cpu().numpy().astype(np.int32)

            for (x1, y1, x2, y2), conf, cid in zip(boxes, confs, cls_ids):
                if int(cid) in EXCLUDED_CLASSES:
                    continue

                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(WIDTH - 1, x2); y2 = min(HEIGHT - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                roi = depth_buf[y1:y2, x1:x2]
                roi_m = roi.astype(np.float32) * depth_scale
                valid = (roi_m > MIN_VALID_DEPTH_M) & (roi_m < MAX_VALID_DEPTH_M)

                if np.count_nonzero(valid) < 20:
                    distance = float("nan")
                else:
                    distance = float(np.median(roi_m[valid]))

                label = model.names.get(int(cid), str(cid))
                dist_txt = f"{distance:.2f} m" if not np.isnan(distance) else "n/a"
                cv2.rectangle(color_buf, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    color_buf,
                    f"{label} {conf:.2f} | {dist_txt}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
                )

        writer_vis.write(color_buf)
        cv2.imshow("BOSS - RealSense + YOLOv8", color_buf)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

finally:
    pipeline.stop()
    writer_vis.release()
    writer_raw.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Video salvati in: {out_dir}")
