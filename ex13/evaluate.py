"""
Step 3: Evaluate model on test set and run inference on a single video sample.
Outputs confusion matrix, classification report, and single-sample prediction.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode

DATA_DIR = r"D:\实验13_骨架Transformer羽毛球动作识别\data"
RESULTS_DIR = r"D:\实验13_骨架Transformer羽毛球动作识别\results"
DATA_ROOT = r"D:\羽毛球视频数据集"
MODEL_PATH = r"C:\mpmodel\pose_landmarker_lite.task"

INPUT_DIM = 132
TARGET_FRAMES = 30
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
NUM_CLASSES = 6
DROPOUT = 0.1


class SkeletonTransformer(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, d_model=D_MODEL, nhead=NHEAD,
                 num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
                 num_classes=NUM_CLASSES, dropout=DROPOUT, seq_len=TARGET_FRAMES):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.input_proj(x) + self.pos_emb(pos)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


def extract_skeleton_single(video_path, target_frames=TARGET_FRAMES):
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
    )
    landmarker = vision.PoseLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(video_path)
    frames_raw = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            kps = []
            for lm in result.pose_landmarks[0]:
                kps.extend([lm.x, lm.y, lm.z, lm.visibility])
            frames_raw.append(kps)
        else:
            if frames_raw:
                frames_raw.append(frames_raw[-1])
            else:
                frames_raw.append([0.0] * 132)
    cap.release()
    landmarker.close()
    if not frames_raw:
        return None
    arr = np.array(frames_raw, dtype=np.float32)
    T = arr.shape[0]
    indices = np.linspace(0, T - 1, target_frames)
    lo = np.floor(indices).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    alpha = (indices - lo)[:, None]
    resampled = arr[lo] * (1 - alpha) + arr[hi] * alpha

    # Normalize
    LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER = 23, 24, 11, 12
    normalized = resampled.copy()
    for i in range(target_frames):
        lhx = resampled[i, LEFT_HIP * 4]; lhy = resampled[i, LEFT_HIP * 4 + 1]
        rhx = resampled[i, RIGHT_HIP * 4]; rhy = resampled[i, RIGHT_HIP * 4 + 1]
        cx, cy = (lhx + rhx) / 2, (lhy + rhy) / 2
        lsx = resampled[i, LEFT_SHOULDER * 4]; rsx = resampled[i, RIGHT_SHOULDER * 4]
        shoulder_w = max(abs(lsx - rsx), 1e-6)
        for j in range(33):
            normalized[i, j * 4] = (resampled[i, j * 4] - cx) / shoulder_w
            normalized[i, j * 4 + 1] = (resampled[i, j * 4 + 1] - cy) / shoulder_w
            normalized[i, j * 4 + 2] = resampled[i, j * 4 + 2] / shoulder_w
    return normalized


def evaluate():
    # Load data and model
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    with open(os.path.join(DATA_DIR, "label_map.json"), "r", encoding="utf-8") as f:
        label_map = json.load(f)
    class_names = [label_map[str(i)] for i in range(NUM_CLASSES)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SkeletonTransformer().to(device)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "model.pth"),
                                     map_location=device))
    model.eval()

    # Predict all test samples
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    acc = (preds == y_test).mean()
    print(f"Test Accuracy: {acc:.4f} ({(preds==y_test).sum()}/{len(y_test)})")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=class_names))

    # Save classification report
    report = classification_report(y_test, preds, target_names=class_names, output_dict=True)
    with open(os.path.join(RESULTS_DIR, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor="gray", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved.")

    # Per-class accuracy bar chart
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(class_names, per_class_acc, color="black", alpha=0.7)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy")
    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "per_class_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close()

    return acc, preds, y_test, class_names


def inference_demo(video_path, class_names):
    """Run inference on a single video."""
    print(f"\n=== Inference Demo ===")
    print(f"Video: {video_path}")
    seq = extract_skeleton_single(video_path)
    if seq is None:
        print("Failed to extract skeleton from video.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SkeletonTransformer().to(device)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "model.pth"),
                                     map_location=device))
    model.eval()

    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = probs.argmax()
    confidence = probs[pred_idx]
    print(f"Predicted class: {class_names[pred_idx]}")
    print(f"Confidence: {confidence:.4f}")
    print("All class probabilities:")
    for i, (name, p) in enumerate(zip(class_names, probs)):
        print(f"  {i}: {name:<25} {p:.4f}")

    # Save result
    result = {
        "video": video_path,
        "predicted_class": class_names[pred_idx],
        "confidence": float(confidence),
        "all_probs": {name: float(p) for name, p in zip(class_names, probs)}
    }
    with open(os.path.join(RESULTS_DIR, "inference_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Inference result saved.")


if __name__ == "__main__":
    acc, preds, y_test, class_names = evaluate()

    # Pick a demo video (first video from forehand_clear)
    demo_folder = os.path.join(DATA_ROOT, "forehand_clear")
    demo_files = [f for f in os.listdir(demo_folder) if f.endswith(".mp4")]
    if demo_files:
        demo_video = os.path.join(demo_folder, demo_files[0])
        inference_demo(demo_video, class_names)
    else:
        print("No demo video found.")
