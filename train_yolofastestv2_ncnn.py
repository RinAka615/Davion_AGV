#!/usr/bin/env python3
"""
Huấn luyện & Lượng tử hóa tĩnh NCNN INT8 cho mô hình YOLO-Fastest V2 (ShuffleNetV2 + Lightweight PAN)
trên tập dữ liệu đa góc độ & đa điều kiện ánh sáng (datasets/batden_multi_angle_v3).
Ghi đè trực tiếp lên models/yolo_fastestv2_int8.param & models/yolo_fastestv2_int8.bin.
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

IMG_SIZE = 320
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(WORKSPACE, "datasets", "batden_multi_angle_v3")
MODELS_DIR = os.path.join(WORKSPACE, "models")
PNNX_BIN = os.path.join(WORKSPACE, ".venv-yolo", "bin", "pnnx")
NCNN_LIB = "/tmp/ncnn_tools/ncnn-20240820-ubuntu-2204-shared/lib"
NCNN_BIN = "/tmp/ncnn_tools/ncnn-20240820-ubuntu-2204-shared/bin"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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
    """
    Khối cơ sở của YOLO-Fastest V2 dựa trên ShuffleNetV2 (tối ưu cho NCNN INT8).
    """
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.use_res = (stride == 1 and in_c == out_c)
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


class YOLOFastestV2Seg(nn.Module):
    """
    YOLO-Fastest V2 (ShuffleNetV2-based Backbone + Lightweight FPN/PAN Segmentation Head)
    Input: (1, 3, 320, 320) -> Output: (1, 1, 320, 320)
    """
    def __init__(self, export_mode=True):
        super().__init__()
        self.export_mode = export_mode
        # Stem: 320 -> 160 -> 80
        self.stem = nn.Sequential(
            ConvBNReLU(3, 24, k=3, s=2, p=1),
            DWConvBlock(24, 24, k=3, s=2),
        )
        # Stage 2 (80 -> 40, 48 ch)
        self.stage2 = nn.Sequential(
            ShuffleInvertedUnit(24, 48, stride=2),
            ShuffleInvertedUnit(48, 48, stride=1),
            ShuffleInvertedUnit(48, 48, stride=1),
        )
        # Stage 3 (40 -> 20, 96 ch)
        self.stage3 = nn.Sequential(
            ShuffleInvertedUnit(48, 96, stride=2),
            ShuffleInvertedUnit(96, 96, stride=1),
            ShuffleInvertedUnit(96, 96, stride=1),
            ShuffleInvertedUnit(96, 96, stride=1),
        )
        # Stage 4 (20 -> 10, 128 ch)
        self.stage4 = nn.Sequential(
            ShuffleInvertedUnit(96, 128, stride=2),
            ShuffleInvertedUnit(128, 128, stride=1),
        )
        # Lightweight FPN-PAN Neck
        self.lat4 = ConvBNReLU(128, 48, k=1, s=1, p=0)
        self.lat3 = ConvBNReLU(96, 48, k=1, s=1, p=0)
        self.lat2 = ConvBNReLU(48, 48, k=1, s=1, p=0)

        self.head = nn.Sequential(
            DWConvBlock(48, 32, k=3, s=1),
            nn.Conv2d(32, 1, 1, 1, 0),
        )

    def forward(self, x):
        s1 = self.stem(x)       # 80x80
        c2 = self.stage2(s1)    # 40x40
        c3 = self.stage3(c2)    # 20x20
        c4 = self.stage4(c3)    # 10x10

        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=(20, 20), mode="bilinear", align_corners=False)
        p2 = self.lat2(c2) + F.interpolate(p3, size=(40, 40), mode="bilinear", align_corners=False)

        logits = self.head(p2)
        logits = F.interpolate(logits, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        return torch.sigmoid(logits) if self.export_mode else logits


class MultiLightPathDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, augment=False):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        assert len(self.img_paths) > 0, f"Không tìm thấy ảnh trong {img_dir}"
        self.lbl_dir = lbl_dir
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def _load_mask(self, lbl_path, h, w):
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

        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        assert bgr is not None and bgr.size > 0
        img = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = self._load_mask(lbl_path, IMG_SIZE, IMG_SIZE)

        if self.augment:
            # 1. Lật ngang ngẫu nhiên
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)

            # 2. Xoay góc nhẹ & dịch chuyển mô phỏng nghiêng đầu xe
            if random.random() < 0.4:
                ang = random.uniform(-12.0, 12.0)
                scale = random.uniform(0.92, 1.08)
                M = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), ang, scale)
                img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                mask = cv2.warpAffine(mask, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # 3. Đổ bóng cục bộ / vệt nắng chói
            if random.random() < 0.45:
                sx1, sy1 = random.randint(0, IMG_SIZE - 80), random.randint(0, IMG_SIZE - 80)
                sx2, sy2 = min(IMG_SIZE, sx1 + random.randint(40, 140)), min(IMG_SIZE, sy1 + random.randint(40, 140))
                factor = random.choice([random.uniform(0.35, 0.70), random.uniform(1.25, 1.65)])
                img[sy1:sy2, sx1:sx2] = np.clip(img[sy1:sy2, sx1:sx2].astype(np.float32) * factor, 0, 255).astype(np.uint8)

            # 4. Biến thiên độ sáng, tương phản & nhiệt độ màu (Ấm / Lạnh)
            if random.random() < 0.6:
                alpha = random.uniform(0.70, 1.30)
                beta = random.uniform(-30, 30)
                img_f = img.astype(np.float32) * alpha + beta
                # Thay đổi cân bằng trắng nhẹ
                b_gain = random.uniform(0.88, 1.12)
                r_gain = random.uniform(0.88, 1.12)
                img_f[:, :, 0] *= b_gain
                img_f[:, :, 2] *= r_gain
                img = np.clip(img_f, 0, 255).astype(np.uint8)

        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=np.float32) / 255.0
        t_img = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        t_mask = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)).unsqueeze(0).contiguous()
        return t_img, t_mask


def combined_seg_loss(logits, targets, bce_fn, eps=1e-6):
    bce_loss = bce_fn(logits, targets)
    probs = torch.sigmoid(logits)
    # Tính Dice loss trên toàn batch để ổn định cả ảnh có bạt và ảnh âm tính (hard negative)
    num = 2.0 * (probs * targets).sum() + eps
    den = probs.sum() + targets.sum() + eps
    dice = 1.0 - (num / den)
    return 0.55 * bce_loss + 0.45 * dice


def evaluate_metrics(model, val_loader):
    model.eval()
    pos_ious = []
    neg_fp_rates = []
    all_ious = []
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5).float()
            for i in range(x.size(0)):
                p_i = preds[i]
                y_i = y[i]
                inter = (p_i * y_i).sum().item()
                union = ((p_i + y_i) > 0).float().sum().item()
                iou = (inter + 1e-6) / (union + 1e-6)
                all_ious.append(iou)
                if y_i.sum().item() > 0:
                    pos_ious.append(iou)
                else:
                    fp_ratio = p_i.mean().item()
                    neg_fp_rates.append(1.0 if fp_ratio > 0.005 else 0.0)
    return (
        float(np.mean(all_ious)) if all_ious else 0.0,
        float(np.mean(pos_ious)) if pos_ious else 0.0,
        float(np.mean(neg_fp_rates)) if neg_fp_rates else 0.0,
    )


def main():
    torch.set_num_threads(4)
    train_img_dir = os.path.join(DATASET_DIR, "images", "train")
    train_lbl_dir = os.path.join(DATASET_DIR, "labels", "train")
    val_img_dir = os.path.join(DATASET_DIR, "images", "val")
    val_lbl_dir = os.path.join(DATASET_DIR, "labels", "val")

    train_ds = MultiLightPathDataset(train_img_dir, train_lbl_dir, augment=True)
    val_ds = MultiLightPathDataset(val_img_dir, val_lbl_dir, augment=False)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # Tạo danh sách ảnh hiệu chuẩn INT8 (KL-divergence) từ tập dữ liệu mới
    calib_list_path = "/tmp/calib_yolofastestv2_v3.txt"
    all_calib_imgs = sorted(glob.glob(os.path.join(train_img_dir, "*.jpg")) + glob.glob(os.path.join(val_img_dir, "*.jpg")))
    with open(calib_list_path, "w") as f:
        for p in all_calib_imgs:
            f.write(p + "\n")

    model = YOLOFastestV2Seg(export_mode=False)

    # Nếu có thể, khởi tạo nhẹ hoặc train từ đầu với CosineAnnealingLR
    epochs = 35
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    bce_fn = nn.BCEWithLogitsLoss()

    best_score = 0.0
    best_pos_iou = 0.0
    best_state = None

    print("=" * 68)
    print(f"🚀 HUẤN LUYỆN YOLO-Fastest V2 ({IMG_SIZE}x{IMG_SIZE}) TRÊN TẬP DỮ LIỆU MỚI")
    print(f"   Train: {len(train_ds)} ảnh | Val: {len(val_ds)} ảnh | Epochs: {epochs}")
    print("=" * 68)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = combined_seg_loss(logits, y, bce_fn)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        all_iou, pos_iou, neg_fp = evaluate_metrics(model, val_loader)
        # Ưu tiên tối đa hóa mIoU trên ảnh có bạt và phạt nặng nếu có báo động giả (False Positive)
        score = pos_iou - 0.15 * neg_fp
        if score > best_score:
            best_score = score
            best_pos_iou = pos_iou
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"  Epoch {epoch:02d}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | "
                f"Val mIoU (Pos): {pos_iou*100:.2f}% | All mIoU: {all_iou*100:.2f}% | "
                f"FalsePos: {neg_fp*100:.1f}% (Best Pos mIoU: {best_pos_iou*100:.2f}%)"
            )

    train_sec = time.time() - t0
    print(f"✔ Hoàn tất huấn luyện sau {train_sec:.1f}s | Best Val mIoU: {best_pos_iou*100:.2f}%")

    # Lưu bản PyTorch FP32 dự phòng
    export_m = YOLOFastestV2Seg(export_mode=True)
    export_m.load_state_dict(best_state)
    export_m.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32).contiguous()
    pt_path = "/tmp/yolo_fastestv2_fp32.pt"
    traced = torch.jit.trace(export_m, dummy)
    traced.save(pt_path)

    print("▶ Đang chuyển đổi sang NCNN FP32 qua PNNX...")
    subprocess.run(
        [PNNX_BIN, pt_path, f"inputshape=[1,3,{IMG_SIZE},{IMG_SIZE}]", "fp16=0"],
        cwd="/tmp",
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    param_fp32 = "/tmp/yolo_fastestv2_fp32.ncnn.param"
    bin_fp32 = "/tmp/yolo_fastestv2_fp32.ncnn.bin"
    table_path = "/tmp/yolo_fastestv2.table"
    out_param = os.path.join(MODELS_DIR, "yolo_fastestv2_int8.param")
    out_bin = os.path.join(MODELS_DIR, "yolo_fastestv2_int8.bin")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = NCNN_LIB

    print(f"▶ Đang hiệu chuẩn phân phối KL-Divergence (INT8 Calibration) trên {len(all_calib_imgs)} ảnh...")
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
            "thread=4",
            "method=kl",
        ],
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print("▶ Đang lượng tử hóa tĩnh sang NCNN INT8 (ncnn2int8)...")
    subprocess.run(
        [f"{NCNN_BIN}/ncnn2int8", param_fp32, bin_fp32, out_param, out_bin, table_path],
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    size_kb = (os.path.getsize(out_param) + os.path.getsize(out_bin)) / 1024.0
    print("=" * 68)
    print(f"✅ XUẤT THÀNH CÔNG NCNN INT8 (GHI ĐÈ MODEL CŨ):")
    print(f"   - File cấu trúc : {out_param}")
    print(f"   - File trọng số : {out_bin}")
    print(f"   - Tổng dung lượng: {size_kb:.1f} KB")
    print(f"   - Độ chính xác  : {best_pos_iou*100:.2f}% mIoU")
    print("=" * 68)


if __name__ == "__main__":
    main()

