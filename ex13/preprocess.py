"""
Step 1: Extract skeleton sequences from badminton videos using MediaPipe Pose.
Parallel version: uses multiprocessing to stay within time limits.
Saves X_train.npy, X_test.npy, y_train.npy, y_test.npy, label_map.json to ./data/
"""

import os
import sys
import json
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode
from multiprocessing import Pool
import time

DATA_ROOT = r"D:\羽毛球视频数据集"
OUT_DIR = r"D:\实验13_骨架Transformer羽毛球动作识别\data"
MODEL_PATH = r"C:\mpmodel\pose_landmarker_lite.task"
TARGET_FRAMES = 30
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
NUM_WORKERS = 4

LABEL_MAP = {
    "forehand_drive": 0,
    "forehand_lift": 1,
    "forehand_net_shot": 2,
    "forehand_clear": 3,
    "backhand_drive": 4,
    "backhand_net_shot": 5,
}

os.makedirs(OUT_DIR, exist_ok=True)


def process_video(args):
    """Worker function: process a single video, returns (seq, label) or (None, label)."""
    video_path, label, model_path, target_frames = args

    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    opts = vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
    )
    landmarker = vision.PoseLandmarker.create_from_options(opts)

    try:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            frames_rgb = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            if not frames_rgb:
                return None, label
            total = len(frames_rgb)
            indices = np.linspace(0, total - 1, target_frames).astype(int)
            selected = [frames_rgb[i] for i in indices]
        else:
            indices = np.linspace(0, total - 1, target_frames).astype(int)
            selected = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    selected.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                elif selected:
                    selected.append(selected[-1])
                else:
                    selected.append(np.zeros((256, 256, 3), dtype=np.uint8))
            cap.release()

        frames_raw = []
        last_valid = [0.0] * 132
        for rgb in selected:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                kps = []
                for lm in result.pose_landmarks[0]:
                    kps.extend([lm.x, lm.y, lm.z, lm.visibility])
                last_valid = kps
            else:
                kps = last_valid
            frames_raw.append(kps)

        arr = np.array(frames_raw, dtype=np.float32)

        # Normalize: center on hip midpoint, scale by shoulder width
        LEFT_HIP, RIGHT_HIP = 23, 24
        LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
        normalized = arr.copy()
        for i in range(target_frames):
            lhx = arr[i, LEFT_HIP * 4];   lhy = arr[i, LEFT_HIP * 4 + 1]
            rhx = arr[i, RIGHT_HIP * 4];  rhy = arr[i, RIGHT_HIP * 4 + 1]
            cx, cy = (lhx + rhx) / 2, (lhy + rhy) / 2
            lsx = arr[i, LEFT_SHOULDER * 4]
            rsx = arr[i, RIGHT_SHOULDER * 4]
            shoulder_w = max(abs(lsx - rsx), 1e-6)
            for j in range(33):
                normalized[i, j * 4]     = (arr[i, j * 4]     - cx) / shoulder_w
                normalized[i, j * 4 + 1] = (arr[i, j * 4 + 1] - cy) / shoulder_w
                normalized[i, j * 4 + 2] =  arr[i, j * 4 + 2]       / shoulder_w

        return normalized, label

    except Exception as e:
        return None, label
    finally:
        landmarker.close()


def collect_all_tasks():
    tasks = []
    for class_name, label in LABEL_MAP.items():
        folder = os.path.join(DATA_ROOT, class_name)
        if not os.path.isdir(folder):
            print(f"[WARN] not found: {folder}")
            continue
        videos = [
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in VIDEO_EXTS
        ]
        for vid in videos:
            tasks.append((os.path.join(folder, vid), label, MODEL_PATH, TARGET_FRAMES))
    return tasks


if __name__ == "__main__":
    print(f"=== Skeleton Extraction (workers={NUM_WORKERS}) ===", flush=True)
    tasks = collect_all_tasks()
    print(f"Total videos: {len(tasks)}", flush=True)

    t0 = time.time()
    X, y = [], []
    ok, skip = 0, 0

    with Pool(processes=NUM_WORKERS) as pool:
        for i, (seq, label) in enumerate(pool.imap_unordered(process_video, tasks), 1):
            if seq is not None:
                X.append(seq)
                y.append(label)
                ok += 1
            else:
                skip += 1
            if i % 50 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                eta = elapsed / i * (len(tasks) - i)
                print(f"  [{i}/{len(tasks)}] ok={ok} skip={skip} "
                      f"elapsed={elapsed:.0f}s ETA={eta:.0f}s", flush=True)

    print(f"\nDone in {time.time()-t0:.1f}s. ok={ok}, skip={skip}", flush=True)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"Dataset: X={X.shape}, y={y.shape}", flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUT_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "y_test.npy"),  y_test)

    inv_map = {v: k for k, v in LABEL_MAP.items()}
    with open(os.path.join(OUT_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(inv_map, f, ensure_ascii=False, indent=2)

    print(f"Train: {X_train.shape}  Test: {X_test.shape}", flush=True)
    print(f"Saved to {OUT_DIR}", flush=True)
