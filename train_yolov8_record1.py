#!/usr/bin/env python3
"""
Huấn luyện (Fine-tune) mô hình đa nhiệm Ultralytics YOLOv8 Segmentation (.pt)
trên tập dữ liệu thực địa robot_record_20261105_133306.mp4 (kết hợp bộ dữ liệu đa góc độ & ánh sáng):
- Class 0: bat_loi_di (Đường bạt đen)
- Class 1: qr_code (Bảng mã QR thực địa & khi quay xe bị nhòe chuyển động)
Lưu tại models/yolov8_multitask_new.pt và cập nhật models/nhamang_batden_seg.pt.
"""

import glob
import os
import random
import shutil
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    os.nice(15)
except Exception:
    pass

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(WORKSPACE, "datasets", "batden_multi_angle_v3")
IMG_SIZE = 320


def prepare_yolov8_multitask_labels():
    """
    1. Đồng bộ nhãn phân đoạn đa giác (YOLO-seg polygon format) cho cả 2 class:
       - Class 0: bat_loi_di
       - Class 1: qr_code (từ qr_masks/{train,val}/*.png)
    2. Sinh thêm các mẫu tăng cường ánh sáng + nhòe chuyển động ngang (Horizontal Motion Blur)
       cho bảng mã QR trực tiếp trong tập train.
    3. Cập nhật dataset.yaml với 2 class: 0: bat_loi_di, 1: qr_code.
    """
    from train_yolofastestv2_qr_ncnn import extract_all_multitask_data

    qr_patches = extract_all_multitask_data()
    stats = {"train_c0": 0, "train_c1": 0, "val_c0": 0, "val_c1": 0}

    # Xóa các mẫu copy-paste tạm từ lần chạy trước (nếu có) để tránh nhân bản
    for old_cp in glob.glob(os.path.join(DATASET_DIR, "images", "train", "yolov8_mb_qr_*.jpg")):
        base_cp = os.path.splitext(os.path.basename(old_cp))[0]
        for sub, ext in [("images", ".jpg"), ("labels", ".txt"), ("qr_masks", ".png")]:
            p = os.path.join(DATASET_DIR, sub, "train", base_cp + ext)
            if os.path.exists(p):
                os.remove(p)

    for split in ["train", "val"]:
        img_dir = os.path.join(DATASET_DIR, "images", split)
        lbl_dir = os.path.join(DATASET_DIR, "labels", split)
        qr_dir = os.path.join(DATASET_DIR, "qr_masks", split)

        for img_path in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
            base = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, f"{base}.txt")
            qr_path = os.path.join(qr_dir, f"{base}.png")

            # Đọc các dòng class 0 (bat_loi_di) hiện có
            c0_lines = []
            if os.path.exists(lbl_path):
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if s.startswith("0 "):
                            c0_lines.append(s)

            # Chuyển mask QR trong qr_masks/*.png sang đa giác chuẩn hóa class 1 (qr_code)
            c1_lines = []
            if os.path.exists(qr_path):
                qr_m = cv2.imread(qr_path, cv2.IMREAD_GRAYSCALE)
                if qr_m is not None and np.any(qr_m > 127):
                    mh, mw = qr_m.shape[:2]
                    cnts, _ = cv2.findContours((qr_m > 127).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in cnts:
                        if cv2.contourArea(c) < 25:
                            continue
                        x, y, w, h = cv2.boundingRect(c)
                        x1 = np.clip(x / float(mw), 0.0, 1.0)
                        y1 = np.clip(y / float(mh), 0.0, 1.0)
                        x2 = np.clip((x + w) / float(mw), 0.0, 1.0)
                        y2 = np.clip((y + h) / float(mh), 0.0, 1.0)
                        c1_lines.append(f"1 {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}")

            if c0_lines:
                stats[f"{split}_c0"] += len(c0_lines)
            if c1_lines:
                stats[f"{split}_c1"] += len(c1_lines)

            with open(lbl_path, "w", encoding="utf-8") as f:
                for l in c0_lines + c1_lines:
                    f.write(l + "\n")

    # Sinh thêm 140 mẫu Copy-Paste QR nhòe chuyển động (Motion Blur) + biến thiên ánh sáng cho tập train YOLOv8
    train_imgs = sorted(glob.glob(os.path.join(DATASET_DIR, "images", "train", "*.jpg")))
    if qr_patches and train_imgs:
        sampled_bg = random.sample(train_imgs, min(140, len(train_imgs)))
        for idx, bg_path in enumerate(sampled_bg):
            base_bg = os.path.splitext(os.path.basename(bg_path))[0]
            lbl_bg = os.path.join(DATASET_DIR, "labels", "train", f"{base_bg}.txt")
            img = cv2.imread(bg_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            c0_lines = []
            if os.path.exists(lbl_bg):
                with open(lbl_bg, "r", encoding="utf-8") as f:
                    c0_lines = [l.strip() for l in f if l.strip().startswith("0 ")]

            patch = random.choice(qr_patches).copy()
            pw = random.randint(22, 68)
            ph = int(pw * random.uniform(0.95, 1.2))
            patch_r = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_LINEAR)

            # Hiệu ứng nhòe chuyển động ngang (Horizontal Motion Blur) khi robot đang xoay
            ksize = random.choice([3, 5, 7, 9])
            kernel_mb = np.zeros((ksize, ksize), dtype=np.float32)
            kernel_mb[ksize // 2, :] = 1.0 / ksize
            patch_r = cv2.filter2D(patch_r, -1, kernel_mb)

            px = random.randint(6, max(7, w - pw - 6))
            py = random.randint(int(h * 0.06), max(int(h * 0.08), int(h * 0.76) - ph))
            img_aug = img.copy()
            img_aug[py : py + ph, px : px + pw] = patch_r

            # Tăng cường độ sáng/tương phản ngoài trời
            alpha = random.uniform(0.82, 1.22)
            beta = random.uniform(-18, 18)
            img_aug = np.clip(img_aug.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

            x1, y1 = px / float(w), py / float(h)
            x2, y2 = (px + pw) / float(w), (py + ph) / float(h)
            c1_line = f"1 {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}"

            out_name = f"yolov8_mb_qr_{idx:03d}"
            cv2.imwrite(os.path.join(DATASET_DIR, "images", "train", f"{out_name}.jpg"), img_aug)
            with open(os.path.join(DATASET_DIR, "labels", "train", f"{out_name}.txt"), "w", encoding="utf-8") as f:
                for l in c0_lines:
                    f.write(l + "\n")
                f.write(c1_line + "\n")
            stats["train_c1"] += 1

    yaml_path = os.path.join(DATASET_DIR, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(DATASET_DIR)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        f.write("  0: bat_loi_di\n")
        f.write("  1: qr_code\n")

    print("==================================================================")
    print(f"✔ ĐÃ GÁN NHÃN PHÂN ĐOẠN 2 CLASS (0: bat_loi_di, 1: qr_code): {stats}")
    print("==================================================================")
    return yaml_path


def main():
    torch.set_num_threads(4)
    dataset_yaml = prepare_yolov8_multitask_labels()

    base_model_path = os.path.join(WORKSPACE, "models", "nhamang_batden_seg.pt")
    backup_path = os.path.join(WORKSPACE, "models", "nhamang_batden_seg_prev.pt")
    out_multitask_pt = os.path.join(WORKSPACE, "models", "yolov8_multitask_new.pt")

    if os.path.exists(base_model_path) and not os.path.exists(backup_path):
        shutil.copy2(base_model_path, backup_path)

    print("==================================================================")
    print("🚀 BẮT ĐẦU HUẤN LUYỆN ULTRALYTICS YOLOv8-SEG ĐA NHIỆM (bat_loi_di + qr_code)")
    print("==================================================================")
    t0 = time.time()

    start_ckpt = backup_path if os.path.exists(backup_path) else base_model_path
    model = YOLO(start_ckpt)
    model.train(
        data=dataset_yaml,
        epochs=15,
        imgsz=320,
        batch=16,
        device="cpu",
        workers=2,
        freeze=8,
        lr0=0.0025,
        lrf=0.1,
        warmup_epochs=1.5,
        degrees=10.0,
        translate=0.08,
        scale=0.25,
        fliplr=0.5,
        hsv_h=0.02,
        hsv_s=0.45,
        hsv_v=0.4,
        mosaic=0.4,
        project="runs/segment",
        name="yolov8_multitask_133306",
        exist_ok=True,
        verbose=True,
    )

    candidates = sorted(glob.glob("runs/**/yolov8_multitask_133306/weights/best.pt", recursive=True))
    if not candidates:
        candidates = sorted(glob.glob("runs/**/yolov8_multitask_133306/weights/last.pt", recursive=True))
    best_pt = candidates[-1]

    shutil.copy2(best_pt, out_multitask_pt)
    shutil.copy2(best_pt, base_model_path)

    elapsed = time.time() - t0
    print(f"\n✅ Hoàn tất huấn luyện YOLOv8-seg đa nhiệm trong {elapsed:.1f}s!")
    print(f"📦 Đã lưu model .pt tại: {out_multitask_pt}")
    print(f"📦 Đã cập nhật model tại: {base_model_path}")

    # =========================================================================
    # 4. CHẠY THỬ NGHIỆM DỰ ĐOÁN TRÊN VIDEO robot_record_20261105_133306.mp4
    # =========================================================================
    trained_model = YOLO(out_multitask_pt)
    vid_path = os.path.join(WORKSPACE, "output_qr/recordings/robot_record_20261105_133306.mp4")
    cap = cv2.VideoCapture(vid_path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    test_indices = [40, 200, 380, 415, 500, 620, 720, 780, 1080]
    vis_tiles = []

    for idx in test_indices:
        if idx >= len(frames):
            continue
        f = frames[idx]
        res = trained_model.predict(f, imgsz=320, conf=0.25, verbose=False)[0]
        vis = f.copy()
        h, w = vis.shape[:2]

        c0_cnt, c1_cnt = 0, 0
        if res.masks is not None and res.boxes is not None:
            m_data = res.masks.data.cpu().numpy()
            for i in range(len(res.boxes)):
                cls_id = int(res.boxes.cls[i].item())
                conf_v = float(res.boxes.conf[i].item())
                b = res.boxes.xyxy[i].cpu().numpy().astype(int)
                m_u8 = cv2.resize((m_data[i] > 0.5).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
                overlay = vis.copy()
                if cls_id == 0:
                    c0_cnt += 1
                    overlay[m_u8 > 0] = (255, 180, 0)
                    cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
                    cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
                    cv2.putText(vis, f"bat_loi_di {conf_v:.2f}", (b[0], max(30, b[1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                elif cls_id == 1:
                    c1_cnt += 1
                    overlay[m_u8 > 0] = (0, 255, 0)
                    cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)
                    cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
                    cv2.putText(vis, f"qr_code {conf_v:.2f}", (b[0], max(30, b[1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.putText(vis, f"F:{idx} | Path:{c0_cnt} QR:{c1_cnt}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        vis_tiles.append(cv2.resize(vis, (426, 240)))

    if len(vis_tiles) == 9:
        grid = np.vstack([
            np.hstack(vis_tiles[0:3]),
            np.hstack(vis_tiles[3:6]),
            np.hstack(vis_tiles[6:9]),
        ])
        out_preview = "/tmp/yolov8_multitask_133306_preview.jpg"
        cv2.imwrite(out_preview, grid)
        print(f"🖼 Đã lưu ảnh kiểm thử mask 9 khung hình tiêu biểu tại: {out_preview}")


if __name__ == "__main__":
    main()
