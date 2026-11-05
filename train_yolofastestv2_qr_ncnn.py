#!/usr/bin/env python3
"""
Tự động trích xuất dữ liệu Lối đi bạt đen (Channel 0) + Mã QR thực địa & khi đang quay (Channel 1)
từ robot_record_20261105_133306.mp4, 124600.mp4, 131851.mp4, 011534.mp4, 104137.mp4,
kết hợp với tập dữ liệu đa góc độ/ánh sáng (datasets/batden_multi_angle_v3),
huấn luyện mô hình 2-nhiệm-vụ YOLO-Fastest V2 (Channel 0: Bạt đen, Channel 1: Mã QR)
và lượng tử hóa tĩnh sang NCNN INT8 (models/yolo_fastestv2_int8.param & .bin).
"""

import glob
import os
import random
import re
import subprocess
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import zxingcpp

from prepare_record1_dataset import (
    de_overlay_outdoor_frame,
    get_133306_path_poly,
    get_124600_path_poly,
    main as build_base_dataset,
)

try:
    os.nice(15)
except Exception:
    pass

IMG_SIZE = 320
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(WORKSPACE, "datasets", "batden_multi_angle_v3")
QR_MASK_DIR_TRAIN = os.path.join(DATASET_DIR, "qr_masks", "train")
QR_MASK_DIR_VAL = os.path.join(DATASET_DIR, "qr_masks", "val")
MODELS_DIR = os.path.join(WORKSPACE, "models")
PNNX_BIN = os.path.join(WORKSPACE, ".venv-yolo", "bin", "pnnx")
NCNN_LIB = "/tmp/ncnn_tools/ncnn-20240820-ubuntu-2204-shared/lib"
NCNN_BIN = "/tmp/ncnn_tools/ncnn-20240820-ubuntu-2204-shared/bin"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def interp_box(keyframes, idx: int):
    """Nội suy tuyến tính bounding box (x1, y1, x2, y2) giữa các keyframe."""
    if idx < keyframes[0][0] or idx > keyframes[-1][0]:
        return None
    for i in range(len(keyframes) - 1):
        f0, x1_0, y1_0, x2_0, y2_0 = keyframes[i]
        f1, x1_1, y1_1, x2_1, y2_1 = keyframes[i + 1]
        if f0 <= idx <= f1:
            t = float(idx - f0) / max(1.0, float(f1 - f0))
            return (
                int(round((1 - t) * x1_0 + t * x1_1)),
                int(round((1 - t) * y1_0 + t * y1_1)),
                int(round((1 - t) * x2_0 + t * x2_1)),
                int(round((1 - t) * y2_0 + t * y2_1)),
            )
    return None


def get_133306_qr_box(idx: int):
    """
    Trả về tọa độ (x1, y1, x2, y2) chuẩn xác của bảng QR1 (F:180..436) và QR2 (F:480..842)
    trong video thực địa robot_record_20261105_133306.mp4 (từ tâm (cx, cy) và kích thước (w, h)).
    """
    qr1_kf = [
        (180, 485, 135, 45, 48),
        (220, 502, 142, 50, 54),
        (260, 522, 155, 56, 60),
        (300, 555, 165, 64, 68),
        (340, 595, 175, 72, 76),
        (355, 635, 205, 80, 84),
        (370, 638, 295, 92, 96),
        (385, 598, 335, 104, 108),
        (400, 610, 355, 118, 124),
        (415, 610, 465, 142, 150),
        (430, 600, 470, 165, 175),
        (436, 390, 475, 165, 175),
    ]
    qr2_kf = [
        (480, 458, 65, 38, 40),
        (505, 415, 71, 40, 42),
        (530, 515, 92, 44, 46),
        (560, 565, 66, 46, 48),
        (590, 506, 88, 50, 52),
        (620, 445, 72, 58, 62),
        (650, 380, 75, 72, 78),
        (660, 365, 75, 80, 88),
        (680, 315, 88, 92, 100),
        (690, 278, 150, 104, 110),
        (700, 510, 150, 110, 116),
        (705, 580, 152, 112, 118),
        (720, 568, 245, 118, 126),
        (735, 572, 375, 140, 150),
        (750, 545, 422, 175, 185),
        (760, 485, 480, 205, 215),
        (780, 445, 520, 250, 260),
        (800, 415, 530, 260, 270),
        (820, 445, 525, 260, 270),
        (830, 645, 525, 265, 275),
        (842, 1080, 535, 270, 280),
    ]
    for kf in (qr1_kf, qr2_kf):
        if kf[0][0] <= idx <= kf[-1][0]:
            for i in range(len(kf) - 1):
                if kf[i][0] <= idx <= kf[i + 1][0]:
                    t = float(idx - kf[i][0]) / max(1.0, float(kf[i + 1][0] - kf[i][0]))
                    cx = (1.0 - t) * kf[i][1] + t * kf[i + 1][1]
                    cy = (1.0 - t) * kf[i][2] + t * kf[i + 1][2]
                    w = (1.0 - t) * kf[i][3] + t * kf[i + 1][3]
                    h = (1.0 - t) * kf[i][4] + t * kf[i + 1][4]
                    return (
                        int(round(cx - w / 2.0)),
                        int(round(cy - h / 2.0)),
                        int(round(cx + w / 2.0)),
                        int(round(cy + h / 2.0)),
                    )
    return None


def get_124600_qr_box(idx: int):
    """Tọa độ bảng QR ngoài trời trong video robot_record_20261105_124600.mp4 (F:310..444)."""
    qr_kf = [
        (310, 535, 105, 615, 200),
        (320, 525, 118, 616, 224),
        (340, 496, 202, 622, 342),
        (360, 468, 286, 628, 462),
        (380, 546, 315, 712, 498),
        (400, 624, 344, 796, 535),
        (420, 622, 378, 804, 580),
        (444, 620, 412, 812, 625),
    ]
    return interp_box(qr_kf, idx)


def clean_hud_and_green_overlay(frame: np.ndarray) -> np.ndarray:
    """Khử thanh HUD phía trên (y < 44)."""
    h, w = frame.shape[:2]
    f = frame.copy()
    f[:44, :] = cv2.inpaint(
        f[:60, :],
        np.vstack([np.ones((44, w), np.uint8) * 255, np.zeros((16, w), np.uint8)]),
        5,
        cv2.INPAINT_TELEA,
    )[:44, :]
    return f


def write_poly_label(lbl_path: str, poly, w: int = 1280, h: int = 720):
    if poly is None or len(poly) < 3:
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("")
    else:
        coords = []
        for x, y in poly:
            nx = np.clip(float(x) / float(w), 0.0, 1.0)
            ny = np.clip(float(y) / float(h), 0.0, 1.0)
            coords.append(f"{nx:.6f} {ny:.6f}")
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("0 " + " ".join(coords) + "\n")


def extract_all_multitask_data():
    """
    Xây dựng bộ dữ liệu đa nhiệm (Path + QR) từ:
    1) robot_record_20261105_133306.mp4 (Thực địa: Luống 1, QR1, Luống 2, QR2, Luống 3 + Lá cây chói nắng âm tính)
    2) robot_record_20261105_124600.mp4 (Thực địa: Luống bạt nắng gắt + QR ngoài trời)
    3) robot_record_20261103_011534.mp4 (record-.mp4: QR nhòe chuyển động khi đang quay)
    4) robot_record_20261102_104137.mp4 (record_1.mp4: Bạt + QR1 + QR2 trong nhà)
    """
    if not os.path.exists(os.path.join(DATASET_DIR, "images", "train")):
        print("[INIT] Đang khởi tạo tập dữ liệu nền batden_multi_angle_v3...")
        build_base_dataset()

    os.makedirs(QR_MASK_DIR_TRAIN, exist_ok=True)
    os.makedirs(QR_MASK_DIR_VAL, exist_ok=True)

    qr_patches_bgr = []
    added_qr_train = 0
    added_qr_val = 0

    # =========================================================================
    # 1. TRÍCH XUẤT QR1 & QR2 THỰC ĐỊA TỪ robot_record_20261105_133306.mp4
    # =========================================================================
    v133306 = os.path.join(WORKSPACE, "output_qr/recordings/robot_record_20261105_133306.mp4")
    if os.path.exists(v133306):
        cap = cv2.VideoCapture(v133306)
        fr_133306 = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            fr_133306.append(f)
        cap.release()
        print(f"[DATA] Đã nạp {len(fr_133306)} frames từ robot_record_20261105_133306.mp4")

        # Lưu các mẫu patch QR thực địa ngoài trời (cả QR1 trên gạch đỏ và QR2 trên chân sắt)
        for p_idx in [260, 340, 380, 400, 415, 430, 680, 700, 720, 740, 760, 780, 800, 820]:
            box = get_133306_qr_box(p_idx)
            if box is not None and p_idx < len(fr_133306):
                x1, y1, x2, y2 = box
                clean_p = de_overlay_outdoor_frame(fr_133306[p_idx])
                patch = clean_p[max(0, y1 - 4):min(720, y2 + 4), max(0, x1 - 4):min(1280, x2 + 4)].copy()
                if patch.size > 0:
                    qr_patches_bgr.append(patch)

        # Lấy mẫu dày cho mọi khung hình có QR1 (180..436) và QR2 (480..842)
        for idx in range(180, 843, 3):
            if idx >= len(fr_133306):
                break
            box = get_133306_qr_box(idx)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            raw_f = fr_133306[idx]
            clean_f = de_overlay_outdoor_frame(raw_f)
            poly = get_133306_path_poly(idx)

            qr_mask = np.zeros((720, 1280), dtype=np.uint8)
            cv2.rectangle(qr_mask, (x1, y1), (x2, y2), 255, -1)
            qr_320 = cv2.resize(qr_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

            split = "val" if (idx % 15 == 0) else "train"
            for tag, frame_src in [("clean", clean_f), ("raw", raw_f)]:
                name = f"out133306_qr_{tag}_f{idx:04d}"
                img_320 = cv2.resize(frame_src, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(DATASET_DIR, "images", split, f"{name}.jpg"), img_320)
                write_poly_label(os.path.join(DATASET_DIR, "labels", split, f"{name}.txt"), poly, 1280, 720)
                cv2.imwrite(os.path.join(DATASET_DIR, "qr_masks", split, f"{name}.png"), qr_320)
                if split == "train":
                    added_qr_train += 1
                else:
                    added_qr_val += 1

        # Bổ sung mẫu âm tính (Hard Negative cho cả Path và QR) ở giai đoạn quay đầu xe trên cỏ/đất (F:300..440 & F:665..965)
        for idx in list(range(300, 440, 5)) + list(range(665, 965, 5)):
            if idx >= len(fr_133306):
                break
            raw_f = fr_133306[idx]
            clean_f = de_overlay_outdoor_frame(raw_f)
            box = get_133306_qr_box(idx)
            qr_mask = np.zeros((720, 1280), dtype=np.uint8)
            if box is not None:
                cv2.rectangle(qr_mask, (box[0], box[1]), (box[2], box[3]), 255, -1)
            qr_320 = cv2.resize(qr_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            split = "val" if (idx % 25 == 0) else "train"
            for tag, frame_src in [("clean", clean_f), ("raw", raw_f)]:
                name = f"out133306_hn_{tag}_f{idx:04d}"
                img_320 = cv2.resize(frame_src, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(DATASET_DIR, "images", split, f"{name}.jpg"), img_320)
                write_poly_label(os.path.join(DATASET_DIR, "labels", split, f"{name}.txt"), None, 1280, 720)
                cv2.imwrite(os.path.join(DATASET_DIR, "qr_masks", split, f"{name}.png"), qr_320)

    # =========================================================================
    # 2. TRÍCH XUẤT QR THỰC ĐỊA TỪ robot_record_20261105_124600.mp4
    # =========================================================================
    v124600 = os.path.join(WORKSPACE, "output_qr/recordings/robot_record_20261105_124600.mp4")
    if os.path.exists(v124600):
        cap = cv2.VideoCapture(v124600)
        fr_124600 = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            fr_124600.append(f)
        cap.release()
        for p_idx in [320, 360, 400, 440]:
            box = get_124600_qr_box(p_idx)
            if box is not None and p_idx < len(fr_124600):
                x1, y1, x2, y2 = box
                qr_patches_bgr.append(fr_124600[p_idx][y1:y2, x1:x2].copy())

        for idx in range(310, 445, 4):
            if idx >= len(fr_124600):
                break
            box = get_124600_qr_box(idx)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            clean_f = de_overlay_outdoor_frame(fr_124600[idx])
            poly = get_124600_path_poly(idx)
            qr_mask = np.zeros((720, 1280), dtype=np.uint8)
            cv2.rectangle(qr_mask, (x1, y1), (x2, y2), 255, -1)
            img_320 = cv2.resize(clean_f, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            qr_320 = cv2.resize(qr_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

            split = "val" if (idx % 20 == 0) else "train"
            name = f"out124600_qr_f{idx:04d}"
            cv2.imwrite(os.path.join(DATASET_DIR, "images", split, f"{name}.jpg"), img_320)
            write_poly_label(os.path.join(DATASET_DIR, "labels", split, f"{name}.txt"), poly, 1280, 720)
            cv2.imwrite(os.path.join(DATASET_DIR, "qr_masks", split, f"{name}.png"), qr_320)
            if split == "train":
                added_qr_train += 1
            else:
                added_qr_val += 1

    # =========================================================================
    # 3. TRÍCH XUẤT QR KHI ĐANG QUAY TỪ robot_record_20261103_011534.mp4 (record-.mp4)
    # =========================================================================
    recdash_path = os.path.join(WORKSPACE, "output_qr/recordings/robot_record_20261103_011534.mp4")
    if os.path.exists(recdash_path):
        cap = cv2.VideoCapture(recdash_path)
        frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
        cap.release()
        print(f"[DATA] Đã nạp {len(frames)} frames từ {recdash_path}")

        if len(frames) > 565:
            t1 = cv2.cvtColor(frames[67][378:462, 106:210], cv2.COLOR_BGR2GRAY)
            t2 = cv2.cvtColor(frames[91][352:442, 745:838], cv2.COLOR_BGR2GRAY)
            t3 = cv2.cvtColor(frames[99][355:450, 995:1092], cv2.COLOR_BGR2GRAY)
            t4 = cv2.cvtColor(frames[565][372:492, 1085:1202], cv2.COLOR_BGR2GRAY)
            templates = [t1, t2, t3, t4]

            qr_patches_bgr.extend([
                frames[67][375:465, 104:212].copy(),
                frames[91][350:444, 743:840].copy(),
                frames[99][353:452, 993:1094].copy(),
                frames[565][370:494, 1083:1204].copy(),
            ])

            for idx in range(0, len(frames), 3):
                f = frames[idx]
                roi_lower = f[300:, :]
                b, g, r = roi_lower[:, :, 0].astype(np.int16), roi_lower[:, :, 1].astype(np.int16), roi_lower[:, :, 2].astype(np.int16)
                has_cyan_overlay = np.sum((b - r > 55) & (g - r > 45) & (b > 95)) > 1500
                has_green_box = np.sum((g > 230) & (r < 30) & (b < 30)) > 200
                if has_cyan_overlay or has_green_box:
                    continue

                gray = cv2.cvtColor(f[270:570, :], cv2.COLOR_BGR2GRAY)
                best_val = 0.0
                best_box = None
                for tmpl in templates:
                    th, tw = tmpl.shape[:2]
                    for scale in [0.85, 1.0, 1.15]:
                        rw, rh = int(tw * scale), int(th * scale)
                        if rw > gray.shape[1] or rh > gray.shape[0]:
                            continue
                        rt = cv2.resize(tmpl, (rw, rh))
                        res = cv2.matchTemplate(gray, rt, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                        if max_val > best_val:
                            best_val = max_val
                            best_box = (max_loc[0], max_loc[1] + 270, rw, rh)

                if best_val >= 0.62 and best_box is not None:
                    clean_f = clean_hud_and_green_overlay(f)
                    h_f, w_f = clean_f.shape[:2]
                    bx, by, bw, bh = best_box
                    qr_mask = np.zeros((h_f, w_f), dtype=np.uint8)
                    cv2.rectangle(qr_mask, (bx, by), (bx + bw, by + bh), 255, -1)

                    img_320 = cv2.resize(clean_f, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                    qr_320 = cv2.resize(qr_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

                    split = "val" if (idx % 15 == 0) else "train"
                    name = f"recdash_qr_f{idx:03d}"
                    cv2.imwrite(os.path.join(DATASET_DIR, "images", split, f"{name}.jpg"), img_320)
                    write_poly_label(os.path.join(DATASET_DIR, "labels", split, f"{name}.txt"), None)
                    cv2.imwrite(os.path.join(DATASET_DIR, "qr_masks", split, f"{name}.png"), qr_320)
                    if split == "train":
                        added_qr_train += 1
                    else:
                        added_qr_val += 1

    # =========================================================================
    # 4. ĐỒNG BỘ MẶT NẠ QR CHO TẤT CẢ ẢNH TRONG DATASET
    # =========================================================================
    for split in ["train", "val"]:
        img_dir = os.path.join(DATASET_DIR, "images", split)
        qr_dir = os.path.join(DATASET_DIR, "qr_masks", split)
        for img_path in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
            base = os.path.splitext(os.path.basename(img_path))[0]
            qr_out = os.path.join(qr_dir, f"{base}.png")

            qr_m = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            # Nếu là ảnh từ 133306 (không biến đổi xoay/lật), luôn cập nhật đúng tọa độ QR thực địa
            m133306 = re.match(r"^out133306_(?:clean|raw|neg_clean|neg_raw|qr_clean|qr_raw)_f(\d+)$", base)
            if m133306:
                f_idx = int(m133306.group(1))
                box = get_133306_qr_box(f_idx)
                if box is not None:
                    x1, y1, x2, y2 = box
                    sx, sy = IMG_SIZE / 1280.0, IMG_SIZE / 720.0
                    cv2.rectangle(
                        qr_m,
                        (int(x1 * sx), int(y1 * sy)),
                        (int(x2 * sx), int(y2 * sy)),
                        255,
                        -1,
                    )
                cv2.imwrite(qr_out, qr_m)
                continue

            if os.path.exists(qr_out):
                continue

            # Nếu là ảnh biến thể aug từ 133306 hoặc 124600 ở vùng không có QR -> mask rỗng
            if base.startswith("out133306_") or base.startswith("out124600_"):
                cv2.imwrite(qr_out, qr_m)
                continue

            img = cv2.imread(img_path)
            if img is not None:
                try:
                    bs = zxingcpp.read_barcodes(img, try_rotate=True, try_downscale=True, return_errors=True)
                    for b in bs:
                        p = b.position
                        if p.top_left.x != 0 or p.bottom_right.x != 0:
                            pts = np.array(
                                [
                                    [p.top_left.x, p.top_left.y],
                                    [p.top_right.x, p.top_right.y],
                                    [p.bottom_right.x, p.bottom_right.y],
                                    [p.bottom_left.x, p.bottom_left.y],
                                ],
                                dtype=np.int32,
                            )
                            cv2.fillPoly(qr_m, [pts], 255)
                except Exception:
                    pass
            cv2.imwrite(qr_out, qr_m)

    print(f"[DATA] Đã tích hợp thêm {added_qr_train} ảnh Train QR & {added_qr_val} ảnh Val QR | Số mẫu QR patch: {len(qr_patches_bgr)}")
    return qr_patches_bgr


class ConvBNReLU(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DWConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_c, in_c, k, s, p, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ShuffleInvertedUnit(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.use_res = stride == 1 and in_c == out_c
        mid_c = out_c // 2
        self.branch = nn.Sequential(
            ConvBNReLU(in_c, mid_c, k=1, s=1, p=0),
            nn.Conv2d(mid_c, mid_c, 3, stride, 1, groups=mid_c, bias=False),
            nn.BatchNorm2d(mid_c),
            ConvBNReLU(mid_c, out_c, k=1, s=1, p=0),
        )

    def forward(self, x):
        out = self.branch(x)
        return x + out if self.use_res else out


class YOLOFastestV2MultiTaskSeg(nn.Module):
    """
    YOLO-Fastest V2 Multi-Task (ShuffleNetV2 Backbone + Lightweight FPN/PAN Head)
    Input : (1, 3, 320, 320)
    Output: (1, 2, 320, 320) -> Channel 0: bat_loi_di (Path), Channel 1: qr_code (QR)
    """

    def __init__(self, num_classes=2, export_mode=True):
        super().__init__()
        self.export_mode = export_mode
        self.stem = nn.Sequential(
            ConvBNReLU(3, 24, k=3, s=2, p=1),
            DWConvBlock(24, 24, k=3, s=2),
        )
        self.stage2 = nn.Sequential(
            ShuffleInvertedUnit(24, 48, stride=2),
            ShuffleInvertedUnit(48, 48, stride=1),
            ShuffleInvertedUnit(48, 48, stride=1),
        )
        self.stage3 = nn.Sequential(
            ShuffleInvertedUnit(48, 96, stride=2),
            ShuffleInvertedUnit(96, 96, stride=1),
            ShuffleInvertedUnit(96, 96, stride=1),
            ShuffleInvertedUnit(96, 96, stride=1),
        )
        self.stage4 = nn.Sequential(
            ShuffleInvertedUnit(96, 128, stride=2),
            ShuffleInvertedUnit(128, 128, stride=1),
        )
        self.lat4 = ConvBNReLU(128, 48, k=1, s=1, p=0)
        self.lat3 = ConvBNReLU(96, 48, k=1, s=1, p=0)
        self.lat2 = ConvBNReLU(48, 48, k=1, s=1, p=0)

        self.head = nn.Sequential(
            DWConvBlock(48, 32, k=3, s=1),
            nn.Conv2d(32, num_classes, 1, 1, 0),
        )

    def forward(self, x):
        s1 = self.stem(x)
        c2 = self.stage2(s1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)

        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=(20, 20), mode="bilinear", align_corners=False)
        p2 = self.lat2(c2) + F.interpolate(p3, size=(40, 40), mode="bilinear", align_corners=False)

        logits = self.head(p2)
        logits = F.interpolate(logits, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        return torch.sigmoid(logits) if self.export_mode else logits


class MultiTaskPathQRDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, qr_dir, qr_patches=None, augment=False):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.lbl_dir = lbl_dir
        self.qr_dir = qr_dir
        self.qr_patches = qr_patches or []
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def _load_path_mask(self, lbl_path, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        if not os.path.exists(lbl_path):
            return mask
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
                coords[:, 0] *= w
                coords[:, 1] *= h
                cv2.fillPoly(mask, [coords.astype(np.int32)], 1)
        return mask

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(self.lbl_dir, base + ".txt")
        qr_path = os.path.join(self.qr_dir, base + ".png")

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        path_mask = self._load_path_mask(lbl_path, IMG_SIZE, IMG_SIZE)

        if os.path.exists(qr_path):
            qr_raw = cv2.imread(qr_path, cv2.IMREAD_GRAYSCALE)
            qr_mask = (cv2.resize(qr_raw, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
        else:
            qr_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        if self.augment:
            # Copy-Paste bảng QR (cả ngoài trời lẫn trong nhà) ở nhiều độ cao (từ y=5%..78%)
            if qr_mask.sum() == 0 and len(self.qr_patches) > 0 and random.random() < 0.42:
                patch = random.choice(self.qr_patches).copy()
                pw = random.randint(14, 58)
                ph = int(pw * random.uniform(0.95, 1.25))
                patch_resized = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_LINEAR)

                if random.random() < 0.65:
                    ksize = random.choice([3, 5, 7, 9])
                    kernel_mb = np.zeros((ksize, ksize), dtype=np.float32)
                    kernel_mb[ksize // 2, :] = 1.0 / ksize
                    patch_resized = cv2.filter2D(patch_resized, -1, kernel_mb)

                px = random.randint(4, IMG_SIZE - pw - 4)
                py = random.randint(int(IMG_SIZE * 0.05), max(int(IMG_SIZE * 0.06), int(IMG_SIZE * 0.78) - ph))
                img[py : py + ph, px : px + pw] = patch_resized
                qr_mask[py : py + ph, px : px + pw] = 1
                path_mask[py : py + ph, px : px + pw] = 0

            elif qr_mask.sum() > 0 and random.random() < 0.45:
                ksize = random.choice([3, 5, 7])
                kernel_mb = np.zeros((ksize, ksize), dtype=np.float32)
                kernel_mb[ksize // 2, :] = 1.0 / ksize
                img = cv2.filter2D(img, -1, kernel_mb)

            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                path_mask = cv2.flip(path_mask, 1)
                qr_mask = cv2.flip(qr_mask, 1)

            if random.random() < 0.5:
                alpha = random.uniform(0.78, 1.25)
                beta = random.uniform(-22, 22)
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=np.float32) / 255.0
        t_img = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        target = np.stack([path_mask.astype(np.float32), qr_mask.astype(np.float32)], axis=0)
        t_mask = torch.from_numpy(np.ascontiguousarray(target)).contiguous()
        return t_img, t_mask


def multitask_loss(logits, targets, bce_path_fn, bce_qr_fn, eps=1e-6):
    bce_path = bce_path_fn(logits[:, 0:1], targets[:, 0:1])
    bce_qr = bce_qr_fn(logits[:, 1:2], targets[:, 1:2])

    probs = torch.sigmoid(logits)
    p0, t0 = probs[:, 0], targets[:, 0]
    dice_path = 1.0 - (2.0 * (p0 * t0).sum() + eps) / (p0.sum() + t0.sum() + eps)
    p1, t1 = probs[:, 1], targets[:, 1]
    dice_qr = 1.0 - (2.0 * (p1 * t1).sum() + eps) / (p1.sum() + t1.sum() + eps)

    loss_path = 0.5 * bce_path + 0.5 * dice_path
    loss_qr = 0.4 * bce_qr + 0.6 * dice_qr
    return loss_path + 1.3 * loss_qr


def evaluate(model, val_loader):
    model.eval()
    path_ious, qr_ious = [], []
    with torch.no_grad():
        for x, y in val_loader:
            probs = torch.sigmoid(model(x))
            preds = (probs > 0.5).float()
            for i in range(x.size(0)):
                if y[i, 0].sum() > 0:
                    inter = (preds[i, 0] * y[i, 0]).sum().item()
                    union = ((preds[i, 0] + y[i, 0]) > 0).float().sum().item()
                    path_ious.append((inter + 1e-6) / (union + 1e-6))
                if y[i, 1].sum() > 0:
                    inter = (preds[i, 1] * y[i, 1]).sum().item()
                    union = ((preds[i, 1] + y[i, 1]) > 0).float().sum().item()
                    qr_ious.append((inter + 1e-6) / (union + 1e-6))
    return (
        float(np.mean(path_ious)) if path_ious else 0.0,
        float(np.mean(qr_ious)) if qr_ious else 0.0,
    )


def main():
    try:
        os.nice(15)
    except Exception:
        pass
    torch.set_num_threads(2)

    qr_patches = extract_all_multitask_data()

    train_img_dir = os.path.join(DATASET_DIR, "images", "train")
    train_lbl_dir = os.path.join(DATASET_DIR, "labels", "train")
    val_img_dir = os.path.join(DATASET_DIR, "images", "val")
    val_lbl_dir = os.path.join(DATASET_DIR, "labels", "val")

    train_ds = MultiTaskPathQRDataset(train_img_dir, train_lbl_dir, QR_MASK_DIR_TRAIN, qr_patches=qr_patches, augment=True)
    val_ds = MultiTaskPathQRDataset(val_img_dir, val_lbl_dir, QR_MASK_DIR_VAL, qr_patches=None, augment=False)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = YOLOFastestV2MultiTaskSeg(num_classes=2, export_mode=False)

    prev_pt = "/tmp/yolo_fastestv2_mt_fp32.pt" if os.path.exists("/tmp/yolo_fastestv2_mt_fp32.pt") else "/tmp/yolo_fastestv2_fp32.pt"
    if os.path.exists(prev_pt):
        try:
            old_jit = torch.jit.load(prev_pt)
            old_sd = old_jit.state_dict()
            new_sd = model.state_dict()
            transferred = 0
            for k, v in old_sd.items():
                if k in new_sd and new_sd[k].shape == v.shape:
                    new_sd[k] = v.clone()
                    transferred += 1
            model.load_state_dict(new_sd)
            print(f"[TRANSFER] Đã kế thừa {transferred} tầng trọng số từ {prev_pt}!")
        except Exception as e:
            print("[WARN] Không nạp được checkpoint cũ:", e)

    epochs = 16 if os.path.exists(prev_pt) else 28
    lr_init = 1.5e-3 if os.path.exists(prev_pt) else 2.8e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_init, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=8e-5)
    bce_path_fn = nn.BCEWithLogitsLoss()
    bce_qr_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))

    best_score = 0.0
    best_path_iou = 0.0
    best_qr_iou = 0.0
    best_state = None

    print("=" * 70)
    print(f"🚀 HUẤN LUYỆN YOLO-Fastest V2 MULTI-TASK (PATH + QR THỰC ĐỊA & KHI QUAY)")
    print(f"   Train: {len(train_ds)} ảnh | Val: {len(val_ds)} ảnh | Epochs: {epochs} (Low-CPU Mode)")
    print("=" * 70)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = multitask_loss(logits, y, bce_path_fn, bce_qr_fn)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        p_iou, q_iou = evaluate(model, val_loader)
        score = 0.5 * p_iou + 0.5 * q_iou
        if score > best_score:
            best_score = score
            best_path_iou = p_iou
            best_qr_iou = q_iou
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 4 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"  Epoch {epoch:02d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | "
                f"Path mIoU: {p_iou*100:.2f}% | QR mIoU: {q_iou*100:.2f}% "
                f"(Best Path: {best_path_iou*100:.2f}%, Best QR: {best_qr_iou*100:.2f}%)"
            )

    print(f"✔ Hoàn tất sau {time.time() - t0:.1f}s | Best Path mIoU: {best_path_iou*100:.2f}% | Best QR mIoU: {best_qr_iou*100:.2f}%")

    export_m = YOLOFastestV2MultiTaskSeg(num_classes=2, export_mode=True)
    export_m.load_state_dict(best_state)
    export_m.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32).contiguous()
    pt_path = "/tmp/yolo_fastestv2_mt_fp32.pt"
    traced = torch.jit.trace(export_m, dummy)
    traced.save(pt_path)

    subprocess.run(
        [PNNX_BIN, pt_path, f"inputshape=[1,3,{IMG_SIZE},{IMG_SIZE}]", "fp16=0"],
        cwd="/tmp",
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    calib_list_path = "/tmp/calib_yolofastestv2_mt.txt"
    all_calib = sorted(glob.glob(os.path.join(train_img_dir, "*.jpg")) + glob.glob(os.path.join(val_img_dir, "*.jpg")))
    with open(calib_list_path, "w") as f:
        for p in all_calib:
            f.write(p + "\n")

    param_fp32 = "/tmp/yolo_fastestv2_mt_fp32.ncnn.param"
    bin_fp32 = "/tmp/yolo_fastestv2_mt_fp32.ncnn.bin"
    table_path = "/tmp/yolo_fastestv2_mt.table"
    out_param = os.path.join(MODELS_DIR, "yolo_fastestv2_int8.param")
    out_bin = os.path.join(MODELS_DIR, "yolo_fastestv2_int8.bin")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = NCNN_LIB

    subprocess.run(
        [
            f"{NCNN_BIN}/ncnn2table",
            param_fp32,
            bin_fp32,
            calib_list_path,
            table_path,
            "mean=[0.0,0.0,0.0]",
            "norm=[0.0039215686,0.0039215686,0.0039215686]",
            f"shape=[{IMG_SIZE},{IMG_SIZE},3]",
            "pixel=RGB",
            "thread=2",
            "method=kl",
        ],
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    subprocess.run(
        [f"{NCNN_BIN}/ncnn2int8", param_fp32, bin_fp32, out_param, out_bin, table_path],
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    size_kb = (os.path.getsize(out_param) + os.path.getsize(out_bin)) / 1024.0
    print("=" * 70)
    print(f"✅ XUẤT THÀNH CÔNG NCNN INT8 ĐA NHIỆM (PATH + QR):")
    print(f"   - File: {out_param} & {out_bin} ({size_kb:.1f} KB)")
    print(f"   - Path mIoU: {best_path_iou*100:.2f}% | QR mIoU: {best_qr_iou*100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
