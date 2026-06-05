import cv2
import numpy as np
from collections import defaultdict

_id_colors: dict[int, tuple] = {}

def _color_for_id(track_id: int) -> tuple:
    if track_id not in _id_colors:
        rng = np.random.default_rng(seed=track_id * 7)
        _id_colors[track_id] = tuple(int(c) for c in rng.integers(80, 230, 3))
    return _id_colors[track_id]


trails: dict[int, list] = defaultdict(list)
MAX_TRAIL = 40


def draw_detections(frame: np.ndarray, results, model_names: dict) -> np.ndarray:
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return frame

    xyxy   = boxes.xyxy.cpu().numpy().astype(int)
    clsids = boxes.cls.cpu().numpy().astype(int)
    confs  = boxes.conf.cpu().numpy()
    ids    = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)

    for (x1, y1, x2, y2), cls_id, conf, tid in zip(xyxy, clsids, confs, ids):
        color = _color_for_id(tid) if tid is not None else (200, 200, 200)
        label = model_names[cls_id]
        tag   = f"{label} #{tid}  {conf:.0%}" if tid is not None else f"{label} {conf:.0%}"

        # Trail
        if tid is not None:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            trails[tid].append((cx, cy))
            if len(trails[tid]) > MAX_TRAIL:
                trails[tid].pop(0)
            pts = trails[tid]
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                thick = max(1, int(alpha * 3))
                fade  = tuple(int(c * alpha) for c in color)
                cv2.line(frame, pts[i - 1], pts[i], fade, thick)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label pill
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, tag, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 15, 15), 1, cv2.LINE_AA)

    return frame


def draw_hud(frame: np.ndarray, count: int, fps: float) -> np.ndarray:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (320, 85), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, f"Objects: {count}",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 220, 50), 2, cv2.LINE_AA)
    cv2.putText(frame, "CodeAlpha | YOLOv8 + ByteTrack",
                (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
    return frame
