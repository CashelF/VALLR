#!/usr/bin/env python3
"""
Convert a raw talking-head MP4 into a VALLR-ready cropped clip:
- Face detection + lip-landmark crop via MediaPipe Face Mesh
- EMA-smoothed square ROI centered on the mouth
- 224x224 RGB, 25 FPS, no audio
- MP4 (mp4v) ready for: python main.py --mode infer --model_path ... --infer_video_path out.mp4

Usage:
    python prep_for_vallr.py --in input.mp4 --out out_cropped.mp4
Options:
    --size 224         # output square size
    --fps 25           # output frame rate (VALLR/LRS-style)
    --margin 1.6       # scale mouth bbox by this factor
    --min_conf 0.5     # min detection conf
    --ema 0.6          # smoothing factor (0=no smooth, closer to 1 = more smooth)
"""

import argparse
import cv2
import numpy as np
import math
from collections import deque

# MediaPipe 0.10.x
import mediapipe as mp

# Outer+inner lip landmark indices (MediaPipe Face Mesh 468-landmark topology)
MOUTH_LANDMARKS = [
    # outer
    61,146,91,181,84,17,314,405,321,375,291,
    # inner
    78,95,88,178,87,14,317,402,318,324,308
]

def clamp01(x): return max(0.0, min(1.0, x))

def get_mouth_bbox(face_landmarks, img_w, img_h, margin_scale=1.6):
    xs, ys = [], []
    for idx in MOUTH_LANDMARKS:
        lm = face_landmarks.landmark[idx]
        xs.append(clamp01(lm.x) * img_w)
        ys.append(clamp01(lm.y) * img_h)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Make square around mouth with margin
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    w = (x_max - x_min)
    h = (y_max - y_min)
    side = max(w, h) * margin_scale
    return cx, cy, side

def square_to_xyxy(cx, cy, side, img_w, img_h):
    half = 0.5 * side
    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = int(round(cx + half))
    y2 = int(round(cy + half))
    # clamp to image bounds
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img_w, x2); y2 = min(img_h, y2)
    # if out of bounds caused aspect issues, fix to square by padding inside bounds
    w = x2 - x1; h = y2 - y1
    if w != h:
        if w > h:
            d = w - h
            y2 = min(img_h, y2 + math.ceil(d/2))
            y1 = max(0, y1 - (d - math.ceil(d/2)))
        else:
            d = h - w
            x2 = min(img_w, x2 + math.ceil(d/2))
            x1 = max(0, x1 - (d - math.ceil(d/2)))
    return x1, y1, x2, y2

def ema_box(prev, curr, alpha):
    if prev is None: return curr
    px1, py1, px2, py2 = prev
    cx1, cy1, cx2, cy2 = curr
    x1 = int(round(alpha*px1 + (1-alpha)*cx1))
    y1 = int(round(alpha*py1 + (1-alpha)*cy1))
    x2 = int(round(alpha*px2 + (1-alpha)*cx2))
    y2 = int(round(alpha*py2 + (1-alpha)*cy2))
    return (x1, y1, x2, y2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--margin", type=float, default=1.6)
    ap.add_argument("--min_conf", type=float, default=0.5)
    ap.add_argument("--ema", type=float, default=0.6, help="EMA smoothing factor for bbox (0-1)")
    ap.add_argument("--model_sel", type=int, default=0, help="0=short-range, 1=full-range")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.in_path}")

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(in_fps / args.fps)))
    # writer later after first frame shape known
    writer = None
    last_box = None

    mp_face_mesh = mp.solutions.face_mesh
    # fast, video-optimised mode
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,   # better lip contours
        min_detection_confidence=args.min_conf,
        min_tracking_confidence=0.5
    )

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # downsample frames to ~args.fps by simple frame skipping
        if (frame_idx % step) != 0:
            frame_idx += 1
            continue
        frame_idx += 1

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(frame_rgb)

        if res.multi_face_landmarks:
            fl = res.multi_face_landmarks[0]
            cx, cy, side = get_mouth_bbox(fl, w, h, margin_scale=args.margin)
            x1, y1, x2, y2 = square_to_xyxy(cx, cy, side, w, h)
            box = (x1, y1, x2, y2)
        else:
            # fallback: use center square crop based on previous box or central square
            if last_box is not None:
                box = last_box
            else:
                side = min(w, h)
                x1 = (w - side) // 2
                y1 = (h - side) // 2
                box = (x1, y1, x1 + side, y1 + side)

        # smooth
        box = ema_box(last_box, box, args.ema)
        last_box = box
        x1, y1, x2, y2 = box

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue  # skip corrupted frame

        out_frame = cv2.resize(crop, (args.size, args.size), interpolation=cv2.INTER_AREA)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.out_path, fourcc, args.fps, (args.size, args.size))

        writer.write(out_frame)

    cap.release()
    face_mesh.close()
    if writer is not None:
        writer.release()
    print(f"Saved VALLR-ready video to: {args.out_path}")

if __name__ == "__main__":
    main()
