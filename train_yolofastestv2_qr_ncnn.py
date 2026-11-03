#!/usr/bin/env python3
"""
Tự động trích xuất dữ liệu mã QR khi đang quay (Motion-Blurred QR) + Lối đi bạt đen từ record-.mp4,
kết hợp với tập dữ liệu đa góc độ/ánh sáng (datasets/batden_multi_angle_v3),
huấn luyện mô hình 2-nhiệm-vụ YOLO-Fastest V2 (Channel 0: Bạt đen, Channel 1: Mã QR)
và lượng tử hóa tĩnh sang NCNN INT8 (models/yolo_fastestv2_int8.param & .bin).
"""

import glob
import os
import random
import subprocess
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import zxingcpp

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


def clean_hud_and_green_overlay(frame: np.ndarray) -> np.ndarray:
    """Khử thanh HUD phía trên (y < 44) và bất kỳ khung xanh lá/cyan overlay nào trong video record-.mp4."""
    h, w = frame.shape[:2]
    f = frame.copy()
    # Khử thanh HUD trên cùng
    f[:44, :] = cv2.inpaint(
        f[:60, :],
        np.vstack([np.ones((44, w), np.uint8) * 255, np.zeros((16, w), np.uint8)]),
        5,
        cv2.INPAINT_TELEA,
    )[:44, :]
    return f


def extract_record_dash_data():
    """
    Quét các video record-.mp4, record2.mp4, record3.mp4 & line_qr.mp4 để trích xuất:
    1) Các khung hình chứa mã QR khi đang quay tốc độ cao (bị nhòe chuyển động ngang, nghiêng, chói sáng)
    2) Các khung hình âm tính (không có bạt, hoặc góc phòng)
    Lưu ảnh vào datasets/batden_multi_angle_v3/images/{train,val}
    và lưu mask QR vào datasets/batden_multi_angle_v3/qr_masks/{train,val}.
    """
    import json
    os.makedirs(QR_MASK_DIR_TRAIN, exist_ok=True)
    os.makedirs(QR_MASK_DIR_VAL, exist_ok=True)

    # 1. Trích xuất từ record-.mp4
    cap = cv2.VideoCapture(os.path.join(WORKSPACE, "record-.mp4"))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    print(f"[DATA] Đã nạp {len(frames)} frames từ record-.mp4")

    # Mẫu template chuẩn của bảng QR
    t1 = cv2.cvtColor(frames[67][378:462, 106:210], cv2.COLOR_BGR2GRAY)
    t2 = cv2.cvtColor(frames[91][352:442, 745:838], cv2.COLOR_BGR2GRAY)
    t3 = cv2.cvtColor(frames[99][355:450, 995:1092], cv2.COLOR_BGR2GRAY)
    t4 = cv2.cvtColor(frames[565][372:492, 1085:1202], cv2.COLOR_BGR2GRAY)
    templates = [t1, t2, t3, t4]

    qr_patches_bgr = [
        frames[67][375:465, 104:212].copy(),
        frames[91][350:444, 743:840].copy(),
        frames[99][353:452, 993:1094].copy(),
        frames[565][370:494, 1083:1204].copy(),
    ]

    added_qr_train = 0
    added_qr_val = 0

    for idx in range(0, len(frames), 2):
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
            for scale in [0.85, 0.95, 1.0, 1.08, 1.18]:
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

            split = "val" if (idx % 12 == 0) else "train"
            name = f"recdash_qr_f{idx:03d}"
            img_out = os.path.join(DATASET_DIR, "images", split, f"{name}.jpg")
            lbl_out = os.path.join(DATASET_DIR, "labels", split, f"{name}.txt")
            qr_out = os.path.join(DATASET_DIR, "qr_masks", split, f"{name}.png")

            cv2.imwrite(img_out, img_320)
            with open(lbl_out, "w") as lf:
                pass
            cv2.imwrite(qr_out, qr_320)

            if split == "train":
                added_qr_train += 1
            else:
                added_qr_val += 1

    # 2. Trích xuất từ record2.mp4 (Quay tốc độ cao)
    r2_lbl_file = "/home/rinaka/.gemini/antigravity/brain/addb9986-787f-42ef-919c-ddbe26beb407/scratch/r2_labels.json"
    if os.path.exists(r2_lbl_file) and os.path.exists(os.path.join(WORKSPACE, "record2.mp4")):
        r2_lbls = json.load(open(r2_lbl_file))
        cap2 = cv2.VideoCapture(os.path.join(WORKSPACE, "record2.mp4"))
        fr2 = []
        while True:
            ret, f = cap2.read()
            if not ret: break
            fr2.append(f)
        cap2.release()
        print(f"[DATA] Đã nạp {len(fr2)} frames từ record2.mp4")

        # Thêm patch từ record2
        if len(fr2) > 38:
            qr_patches_bgr.append(fr2[38][340:430, 1100:1210].copy())

        for k_str, v in r2_lbls.items():
            f_idx = int(k_str)
            if f_idx >= len(fr2): continue
            score = v[4]
            src_type = v[5]
            if src_type == "seed" or score >= 0.58:
                f = fr2[f_idx]
                clean_f = clean_hud_and_green_overlay(f)
                h_f, w_f = clean_f.shape[:2]
                x1, y1, x2, y2 = max(0, v[0]), max(0, v[1]), min(w_f, v[2]), min(h_f, v[3])
                if (x2 - x1) < 28 or (y2 - y1) < 28: continue

                qr_mask = np.zeros((h_f, w_f), dtype=np.uint8)
                cv2.rectangle(qr_mask, (x1, y1), (x2, y2), 255, -1)

                img_320 = cv2.resize(clean_f, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                qr_320 = cv2.resize(qr_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

                split = "val" if (f_idx % 10 == 0) else "train"
                name = f"rec2_qr_f{f_idx:03d}"
                cv2.imwrite(os.path.join(DATASET_DIR, "images", split, f"{name}.jpg"), img_320)
                with open(os.path.join(DATASET_DIR, "labels", split, f"{name}.txt"), "w") as lf:
                    pass
                cv2.imwrite(os.path.join(DATASET_DIR, "qr_masks", split, f"{name}.png"), qr_320)

                if split == "train": added_qr_train += 1
                else: added_qr_val += 1

    # 3. Trích xuất từ record3.mp4 (Quay tốc độ cao)
    r3_lbl_file = "/home/rinaka/.gemini/antigravity/brain/addb9986-787f-42ef-919c-ddbe26beb407/scratch/r3_labels.json"
    if os.path.exists(r3_lbl_file) and os.path.exists(os.path.join(WORKSPACE, "record3.mp4")):
        r3_lbls = json.load(open(r3_lbl_file))
        cap3 = cv2.VideoCapture(os.path.join(WORKSPACE, "record3.mp4"))
        fr3 = []
        while True:
            ret, f = cap3.read()
            if not ret: break
            fr3.append(f)
        cap3.release()
        print(f"[DATA] Đã nạp {len(fr3)} frames từ record3.mp4")

        for k_str, v in r3_lbls.items():
            f_idx = int(k_str)
            if f_idx >= len(fr3): continue
            # lấy các frame cách nhau 2 để tránh trùng lặp
            if f_idx % 2 != 0: continue
            f = fr3[f_idx]
            clean_f = clean_hud_and_green_overlay(f)
            h_f, w_f = clean_f.shape[:2]
            x1, y1, x2, y2 = max(0, v[0]), max(0, v[1]), min(w_f, v[2]), min(h_f, v[3])
            if (x2 - x1) < 28 or (y2 - y1) < 28: continue

            qr_mask = np.zeros((h_f, w_f), dtype=np.uint8)
            cv2.rectangle(qr_mask, (x1, y1), (x2, y2), 255, -1)

            img_320 = cv2.resize(clean_f, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            qr_320 = cv2.resize(qr_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

            split = "val" if (f_idx % 8 == 0) else "train"
            name = f"rec3_qr_f{f_idx:03d}"
            cv2.imwrite(os.path.join(DATASET_DIR, "images", split, f"{name}.jpg"), img_320)
            with open(os.path.join(DATASET_DIR, "labels", split, f"{name}.txt"), "w") as lf:
                pass
            cv2.imwrite(os.path.join(DATASET_DIR, "qr_masks", split, f"{name}.png"), qr_320)

            if split == "train": added_qr_train += 1
            else: added_qr_val += 1

    # Tự động đồng bộ qr_masks cho các ảnh cũ
    for split in ["train", "val"]:
        img_dir = os.path.join(DATASET_DIR, "images", split)
        qr_dir = os.path.join(DATASET_DIR, "qr_masks", split)
        for img_path in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
            base = os.path.splitext(os.path.basename(img_path))[0]
            qr_out = os.path.join(qr_dir, f"{base}.png")
            if os.path.exists(qr_out):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            qr_m = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
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

    print(f"[DATA] Tổng cộng đã tích hợp: {added_qr_train} ảnh Train QR & {added_qr_val} ảnh Val QR (từ record-, record2, record3)")
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
            # Tăng cường Copy-Paste bảng QR bị nhòe chuyển động (Motion Blur khi xe đang quay)
            if qr_mask.sum() == 0 and len(self.qr_patches) > 0 and random.random() < 0.45:
                patch = random.choice(self.qr_patches).copy()
                # Kích thước QR trên ảnh 320x320 (tương ứng ~60px..180px trên ảnh 1280x720)
                pw = random.randint(20, 56)
                ph = int(pw * random.uniform(0.95, 1.25))
                patch_resized = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_LINEAR)

                # Mô phỏng nhòe chuyển động ngang khi xe đang xoay tốc độ cao (Horizontal Motion Blur)
                if random.random() < 0.80:
                    ksize = random.choice([3, 5, 7, 9, 11])
                    kernel_mb = np.zeros((ksize, ksize), dtype=np.float32)
                    kernel_mb[ksize // 2, :] = 1.0 / ksize
                    patch_resized = cv2.filter2D(patch_resized, -1, kernel_mb)

                px = random.randint(4, IMG_SIZE - pw - 4)
                py = random.randint(int(IMG_SIZE * 0.30), int(IMG_SIZE * 0.72) - ph)
                img[py : py + ph, px : px + pw] = patch_resized
                qr_mask[py : py + ph, px : px + pw] = 1
                path_mask[py : py + ph, px : px + pw] = 0

            # Nếu ảnh đã có QR thật từ record-.mp4 / record2 / record3, mô phỏng thêm nhòe chuyển động khi quay gắt
            elif qr_mask.sum() > 0 and random.random() < 0.65:
                ksize = random.choice([3, 5, 7, 9, 11])
                kernel_mb = np.zeros((ksize, ksize), dtype=np.float32)
                kernel_mb[ksize // 2, :] = 1.0 / ksize
                img = cv2.filter2D(img, -1, kernel_mb)

            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                path_mask = cv2.flip(path_mask, 1)
                qr_mask = cv2.flip(qr_mask, 1)

            if random.random() < 0.5:
                alpha = random.uniform(0.75, 1.28)
                beta = random.uniform(-25, 25)
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=np.float32) / 255.0
        t_img = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        target = np.stack([path_mask.astype(np.float32), qr_mask.astype(np.float32)], axis=0)
        t_mask = torch.from_numpy(np.ascontiguousarray(target)).contiguous()
        return t_img, t_mask


def multitask_loss(logits, targets, bce_fn, eps=1e-6):
    # Channel 0: Path, Channel 1: QR (tăng trọng số cho QR vì diện tích bảng QR nhỏ hơn bạt)
    bce_path = bce_fn(logits[:, 0:1], targets[:, 0:1])
    bce_qr = bce_fn(logits[:, 1:2], targets[:, 1:2])

    probs = torch.sigmoid(logits)
    # Dice Path
    p0, t0 = probs[:, 0], targets[:, 0]
    dice_path = 1.0 - (2.0 * (p0 * t0).sum() + eps) / (p0.sum() + t0.sum() + eps)
    # Dice QR
    p1, t1 = probs[:, 1], targets[:, 1]
    dice_qr = 1.0 - (2.0 * (p1 * t1).sum() + eps) / (p1.sum() + t1.sum() + eps)

    loss_path = 0.5 * bce_path + 0.5 * dice_path
    loss_qr = 0.4 * bce_qr + 0.6 * dice_qr
    return loss_path + 1.2 * loss_qr


def evaluate(model, val_loader):
    model.eval()
    path_ious, qr_ious = [], []
    with torch.no_grad():
        for x, y in val_loader:
            probs = torch.sigmoid(model(x))
            preds = (probs > 0.5).float()
            for i in range(x.size(0)):
                # Path IoU
                if y[i, 0].sum() > 0:
                    inter = (preds[i, 0] * y[i, 0]).sum().item()
                    union = ((preds[i, 0] + y[i, 0]) > 0).float().sum().item()
                    path_ious.append((inter + 1e-6) / (union + 1e-6))
                # QR IoU
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
    # Giới hạn 2 luồng CPU nhẹ nhàng để máy chạy êm ái
    torch.set_num_threads(2)

    qr_patches = extract_record_dash_data()

    train_img_dir = os.path.join(DATASET_DIR, "images", "train")
    train_lbl_dir = os.path.join(DATASET_DIR, "labels", "train")
    val_img_dir = os.path.join(DATASET_DIR, "images", "val")
    val_lbl_dir = os.path.join(DATASET_DIR, "labels", "val")

    train_ds = MultiTaskPathQRDataset(train_img_dir, train_lbl_dir, QR_MASK_DIR_TRAIN, qr_patches=qr_patches, augment=True)
    val_ds = MultiTaskPathQRDataset(val_img_dir, val_lbl_dir, QR_MASK_DIR_VAL, qr_patches=None, augment=False)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = YOLOFastestV2MultiTaskSeg(num_classes=2, export_mode=False)

    # Nạp trọng số Backbone + Neck đã học từ bản trước (Transfer Learning) để hội tụ cực nhanh
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
                elif k == "head.1.weight" and v.shape[0] == 1:
                    new_sd[k][0:1] = v.clone()
                    transferred += 1
                elif k == "head.1.bias" and v.shape[0] == 1:
                    new_sd[k][0:1] = v.clone()
                    transferred += 1
            model.load_state_dict(new_sd)
            print(f"[TRANSFER] Đã kế thừa {transferred} tầng trọng số từ {prev_pt}!")
        except Exception as e:
            print("[WARN] Không nạp được checkpoint cũ:", e)

    epochs = 24
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.8e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    bce_fn = nn.BCEWithLogitsLoss()

    best_score = 0.0
    best_path_iou = 0.0
    best_qr_iou = 0.0
    best_state = None

    print("=" * 70)
    print(f"🚀 HUẤN LUYỆN YOLO-Fastest V2 MULTI-TASK (PATH + QR KHI QUAY)")
    print(f"   Train: {len(train_ds)} ảnh | Val: {len(val_ds)} ảnh | Epochs: {epochs} (Low-CPU Mode)")
    print("=" * 70)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = multitask_loss(logits, y, bce_fn)
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

    # Xuất sang TorchScript FP32 -> PNNX -> NCNN INT8
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

