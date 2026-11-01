#!/usr/bin/env python3
import os
import random
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    random.seed(42)
    output_dir = "datasets/batden_fast"
    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    train_lbl_dir = os.path.join(output_dir, "labels", "train")
    val_lbl_dir = os.path.join(output_dir, "labels", "val")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    teacher = YOLO("models/nhamang_batden_seg.pt")

    videos = [
        {
            "path": "nhamang_20261025_030235.mp4",
            "prefix": "nhamang",
            "pos_range": (0, 440, 4),    # Every 4 frames
            "neg_range": (460, 610, 10),  # Every 10 frames
        },
        {
            "path": "Quỳnh Robotics [13-10-2026 20_33]/thamden.mp4",
            "prefix": "thamden",
            "pos_range": (0, 450, 6),     # Every 6 frames
            "neg_range": None,
        }
    ]

    samples = [] # (img, polygon_list, name)

    for vinfo in videos:
        vpath = vinfo["path"]
        prefix = vinfo["prefix"]
        if not os.path.exists(vpath):
            print(f"Warning: {vpath} not found")
            continue

        cap = cv2.VideoCapture(vpath)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Reading {vpath} ({total_frames} frames, {w}x{h})...")

        # Positive frames
        start, end, step = vinfo["pos_range"]
        for fidx in range(start, min(end, total_frames), step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                continue

            res = teacher.predict(frame, imgsz=416, conf=0.35, verbose=False)[0]
            if res.masks is not None and len(res.masks.xyn) > 0:
                # Find largest mask polygon
                polys = res.masks.xyn
                best_poly = None
                max_pts = 0
                for poly in polys:
                    if len(poly) > max_pts and len(poly) >= 6:
                        max_pts = len(poly)
                        best_poly = poly

                if best_poly is not None:
                    # Simplify polygon slightly to reduce noise (e.g. max 100 points)
                    poly_px = (best_poly * np.array([w, h])).astype(np.int32)
                    epsilon = 0.003 * cv2.arcLength(poly_px, True)
                    approx = cv2.approxPolyDP(poly_px, epsilon, True)
                    if len(approx) >= 4:
                        approx_norm = approx.reshape(-1, 2) / np.array([w, h], dtype=np.float32)
                        samples.append((frame, approx_norm, f"{prefix}_frame_{fidx:04d}"))

        # Negative frames (if any)
        if vinfo["neg_range"] is not None:
            n_start, n_end, n_step = vinfo["neg_range"]
            for fidx in range(n_start, min(n_end, total_frames), n_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ret, frame = cap.read()
                if not ret:
                    continue
                samples.append((frame, None, f"{prefix}_neg_frame_{fidx:04d}"))

        cap.release()

    print(f"Total collected samples: {len(samples)}")
    random.shuffle(samples)

    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    def save_set(subset, img_dir, lbl_dir):
        for frame, poly, name in subset:
            img_path = os.path.join(img_dir, f"{name}.jpg")
            lbl_path = os.path.join(lbl_dir, f"{name}.txt")

            cv2.imwrite(img_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            with open(lbl_path, "w") as f:
                if poly is not None and len(poly) >= 4:
                    # Format: class_id x1 y1 x2 y2 ...
                    coords_str = " ".join([f"{x:.5f} {y:.5f}" for x, y in poly])
                    f.write(f"0 {coords_str}\n")
                else:
                    # Negative sample: empty label file
                    pass

    save_set(train_samples, train_img_dir, train_lbl_dir)
    save_set(val_samples, val_img_dir, val_lbl_dir)

    # Write data.yaml
    data_yaml_path = os.path.abspath(os.path.join(output_dir, "data.yaml"))
    with open(data_yaml_path, "w") as f:
        f.write(f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val

names:
  0: bat_loi_di
""")

    print(f"data.yaml created at {data_yaml_path}")

if __name__ == "__main__":
    main()

