import math
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np
from decord import VideoReader, cpu
from tqdm import tqdm

import sys
# sys.path.append('..')
from data.face_cropper import FaceCropper
from config import load_args


# Outer+inner lip landmark indices (MediaPipe Face Mesh 468-landmark topology)
MOUTH_LANDMARKS = [
    # outer
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    # inner
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
]


class Preprocess:
    def __init__(self, video_dir, output_dir, split="train", frame_size=(224, 224), output_format='mp4'):
        self.video_path = video_dir
        self.output_dir = output_dir
        self.split = split
        self.frame_size = frame_size
        self.output_format = output_format
        self.class_to_index = {}
        self.video_paths = []
        self.labels = []
        # Load the FaceCropper
        self.face_cropper = FaceCropper(min_face_detector_confidence=0.5, face_detector_model_selection="SHORT_RANGE", 
                                        landmark_detector_static_image_mode="STATIC_MODE", min_landmark_detector_confidence=0.5)

    def get_video_path(self):
        for class_name in tqdm(os.listdir(self.video_path), desc="Video Loading", leave=False):
            class_path = os.path.join(self.video_path, class_name)
            count = 0
            if os.path.isdir(class_path):  # Check if it's a directory
                if class_name not in self.class_to_index:
                    self.class_to_index[class_name] = len(self.class_to_index)
                split_path = os.path.join(class_path, self.split)
                if os.path.isdir(split_path):
                    for video_name in os.listdir(split_path):
                        count += 1
                        video_path = os.path.join(split_path, video_name)
                        # if count > 10:
                        #     break
                        if video_name.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Valid video formats
                            self.video_paths.append((video_path, class_name, video_name))
                            self.labels.append(self.class_to_index[class_name])
                        elif video_name.endswith('.txt'):  # Check for label files
                            continue
                        else:
                            print(f"Skipped file with unsupported format: {video_name}")

    def create_output_dir(self, class_name):
        # Create a corresponding directory structure in the output folder
        class_output_dir = os.path.join(self.output_dir, class_name, self.split)
        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)
        return class_output_dir

    def save_preprocessed_video(self, frames, output_path, fps=30):
        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Save as .mp4 file
        height, width, _ = frames[0].shape
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            video_writer.write(frame)  # Write each frame to the video file

        video_writer.release()  # Release the writer
        # print(f"Saved preprocessed video to {output_path}")

    def load_and_preprocess_video(self, video_path):
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        except Exception as e:
            print(f"Error loading video: {video_path}. Error: {e}")
            return None

        frames = []
        for frame in vr:  # Iterate over all frames in the video
            frame = frame.asnumpy()
            frame = cv2.resize(frame, self.frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            faces = self.face_cropper.get_faces(frame, remove_background=False, correct_roll=True)
            if faces:  # If faces are detected
                frame = faces[0]  # Use the first detected face
                frame = cv2.resize(frame, self.frame_size)
            else:
                frame = cv2.resize(frame, self.frame_size)

            frames.append(frame)

        if len(frames) == 0:
            return None  # Skip video if no frames are found

        return frames

    def process_videos(self):
        self.get_video_path()
        for video_path, class_name, video_name in tqdm(
            self.video_paths, desc="Video Preprocessing", leave=False
        ):
            video_frames = self.load_and_preprocess_video(video_path)
            if video_frames is not None:
                # Create output directory structure
                class_output_dir = self.create_output_dir(class_name)

                # Save the video in the specified format (.mp4 by default)
                output_path = os.path.join(
                    class_output_dir, os.path.splitext(video_name)[0] + f".{self.output_format}"
                )
                self.save_preprocessed_video(video_frames, output_path)
            else:
                print(f"Skipping video: {video_name}")


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def get_mouth_bbox(face_landmarks, img_w: int, img_h: int, margin_scale: float = 1.6):
    xs, ys = [], []
    for idx in MOUTH_LANDMARKS:
        lm = face_landmarks.landmark[idx]
        xs.append(clamp01(lm.x) * img_w)
        ys.append(clamp01(lm.y) * img_h)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    w = (x_max - x_min)
    h = (y_max - y_min)
    side = max(w, h) * margin_scale
    return cx, cy, side


def square_to_xyxy(cx: float, cy: float, side: float, img_w: int, img_h: int):
    half = 0.5 * side
    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = int(round(cx + half))
    y2 = int(round(cy + half))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    w = x2 - x1
    h = y2 - y1
    if w != h:
        if w > h:
            d = w - h
            y2 = min(img_h, y2 + math.ceil(d / 2))
            y1 = max(0, y1 - (d - math.ceil(d / 2)))
        else:
            d = h - w
            x2 = min(img_w, x2 + math.ceil(d / 2))
            x1 = max(0, x1 - (d - math.ceil(d / 2)))

    return x1, y1, x2, y2


def ema_box(prev, curr, alpha: float):
    if prev is None:
        return curr

    px1, py1, px2, py2 = prev
    cx1, cy1, cx2, cy2 = curr
    x1 = int(round(alpha * px1 + (1 - alpha) * cx1))
    y1 = int(round(alpha * py1 + (1 - alpha) * cy1))
    x2 = int(round(alpha * px2 + (1 - alpha) * cx2))
    y2 = int(round(alpha * py2 + (1 - alpha) * cy2))
    return (x1, y1, x2, y2)


def preprocess_video_with_mouth_crop(
    video_path: str | Path,
    frame_size: Tuple[int, int] = (224, 224),
    target_fps: float | None = 25.0,
    margin: float = 1.6,
    min_conf: float = 0.5,
    ema: float = 0.6,
    model_selection: int = 0,
) -> Tuple[np.ndarray, float]:
    """Load a video, crop around the mouth, and return frames + effective FPS."""

    reader = VideoReader(str(video_path), ctx=cpu(0))
    in_fps = float(reader.get_avg_fps()) or 30.0
    step = 1
    if target_fps:
        step = max(1, int(round(in_fps / target_fps)))
    effective_fps = in_fps / step

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_kwargs = dict(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=min_conf,
        min_tracking_confidence=0.5,
    )

    # `model_selection` is only available in newer MediaPipe versions; fall back
    # silently if the installed version does not support it.
    if model_selection is not None:
        try:
            face_mesh = mp_face_mesh.FaceMesh(
                **face_mesh_kwargs, model_selection=model_selection
            )
        except TypeError:
            face_mesh = mp_face_mesh.FaceMesh(**face_mesh_kwargs)
    else:
        face_mesh = mp_face_mesh.FaceMesh(**face_mesh_kwargs)

    processed_frames: List[np.ndarray] = []
    last_box = None

    for idx in range(0, len(reader), step):
        frame_rgb = reader[idx].asnumpy()
        h, w = frame_rgb.shape[:2]
        res = face_mesh.process(frame_rgb)

        if res.multi_face_landmarks:
            fl = res.multi_face_landmarks[0]
            cx, cy, side = get_mouth_bbox(fl, w, h, margin_scale=margin)
            x1, y1, x2, y2 = square_to_xyxy(cx, cy, side, w, h)
            box = (x1, y1, x2, y2)
        else:
            if last_box is not None:
                box = last_box
            else:
                side = min(w, h)
                x1 = (w - side) // 2
                y1 = (h - side) // 2
                box = (x1, y1, x1 + side, y1 + side)

        box = ema_box(last_box, box, ema)
        last_box = box
        x1, y1, x2, y2 = box

        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        processed_frames.append(cv2.resize(crop, frame_size, interpolation=cv2.INTER_AREA))

    face_mesh.close()

    if not processed_frames:
        processed_frames = [
            cv2.resize(reader[idx].asnumpy(), frame_size, interpolation=cv2.INTER_AREA)
            for idx in range(0, len(reader), step)
        ]

    video_np = np.stack(processed_frames, axis=0)
    return video_np, effective_fps

def main():
    # Extract args
    args = load_args()

    # Extract specific values from args
    video_dir = args.videos_root
    output_dir = args.videos_output
    preprocessor = Preprocess(video_dir=video_dir, output_dir=output_dir, split="val")
    preprocessor.process_videos()

if __name__ == '__main__':
    main()

# python -m data.preprocess --videos_root videos --videos_output videos_processed