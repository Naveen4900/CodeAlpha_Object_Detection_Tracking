import cv2
import time
import argparse
from ultralytics import YOLO
from utils.draw import draw_detections, draw_hud

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="YOLOv8 Object Detection & Tracking — CodeAlpha")
parser.add_argument("--source",  type=str, default="1",
                    help="Video source: 0 for webcam, or path to video file")
parser.add_argument("--model",   type=str, default="yolov8n.pt",
                    help="YOLO model weights (yolov8n/s/m/l/x.pt)")
parser.add_argument("--conf",    type=float, default=0.4,  help="Confidence threshold")
parser.add_argument("--iou",     type=float, default=0.5,  help="IoU threshold")
parser.add_argument("--save",    action="store_true",      help="Save output video")
parser.add_argument("--tracker", type=str,
                    default="utils/tracker_config.yaml",   help="Tracker config path")
args = parser.parse_args()

# ── Model ─────────────────────────────────────────────────────────────────
model = YOLO(args.model)

# ── Source ────────────────────────────────────────────────────────────────
source = int(args.source) if args.source.isdigit() else args.source
cap    = cv2.VideoCapture(source)

if not cap.isOpened():
    raise RuntimeError(
        f"Cannot open source: {args.source}\n"
        "Mac users: System Settings → Privacy & Security → Camera → Enable for Terminal"
    )

W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS_SRC = cap.get(cv2.CAP_PROP_FPS) or 30

# ── Output writer (optional) ──────────────────────────────────────────────
writer = None
if args.save:
    out_path = "output_tracked.mp4"
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                FPS_SRC, (W, H))
    print(f"[INFO] Saving output → {out_path}")

# ── Main loop ─────────────────────────────────────────────────────────────
print("[INFO] Running — press Q to quit, S to screenshot")
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        conf=args.conf,
        iou=args.iou,
        tracker=args.tracker,
        verbose=False
    )

    frame = draw_detections(frame, results[0], model.names)

    now  = time.time()
    fps  = 1.0 / (now - prev_time + 1e-9)
    prev_time = now

    obj_count = len(results[0].boxes) if results[0].boxes is not None else 0
    frame = draw_hud(frame, obj_count, fps)

    if writer:
        writer.write(frame)

    cv2.imshow("Object Detection & Tracking — CodeAlpha", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        fname = f"screenshot_{int(time.time())}.jpg"
        cv2.imwrite(fname, frame)
        print(f"[INFO] Screenshot saved: {fname}")

cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()
print("[INFO] Done.")
