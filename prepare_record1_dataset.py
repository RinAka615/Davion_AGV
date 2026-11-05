#!/usr/bin/env python3
"""
Tự động trích xuất, khử lớp phủ (de-overlay), gán nhãn đa giác (Polygon Segmentation)
từ bộ video thực địa ngoài vườn (robot_record_20261105_133306.mp4, 124600.mp4, 131851.mp4)
kết hợp video trong nhà/nhà màng (robot_record_20261102_104137.mp4, line_qr.mp4, nhamang_20261025_030235.mp4, thamden.mp4)
và sinh dữ liệu đa góc độ (Perspective/Tilt/Rotation) + đa điều kiện ánh sáng (Glare/Warm/Dim/Cool).
"""

import os
import glob
import random
import shutil
import cv2
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATASET_DIR = "datasets/batden_multi_angle_v3"
IMG_SIZE = 320


def de_overlay_outdoor_frame(frame: np.ndarray) -> np.ndarray:
    """
    Khử lớp phủ HUD phía trên, mặt nạ xanh cyan (alpha=0.35), khung vàng,
    đường tâm đỏ/tím và nhãn xanh lá [OK] TURN_2 trên các video ghi từ thực địa.
    """
    h, w = frame.shape[:2]
    f = frame.copy()

    # 1. Xóa thanh HUD trên cùng (y < 72 nếu có banner, hoặc y < 44)
    f[:44, :] = cv2.inpaint(
        f[:60, :],
        np.vstack([np.ones((44, w), np.uint8) * 255, np.zeros((16, w), np.uint8)]),
        5,
        cv2.INPAINT_TELEA,
    )[:44, :]

    roi_y0 = 120
    roi = f[roi_y0:, :].copy()
    b = roi[:, :, 0].astype(np.float32)
    g = roi[:, :, 1].astype(np.float32)
    r = roi[:, :, 2].astype(np.float32)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 2. Khử lớp phủ xanh cyan cộng tính (+89.25 B, +63.0 G, +0 R)
    is_cyan = (b - r > 14) & (g - r > 8) & (b > 65) & (hsv[:, :, 0] >= 75) & (hsv[:, :, 0] <= 115)
    cyan_u8 = (is_cyan.astype(np.uint8)) * 255
    cyan_closed = cv2.morphologyEx(cyan_u8, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    cnts, _ = cv2.findContours(cyan_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_cyan = np.zeros_like(cyan_closed)
    for c in cnts:
        if cv2.contourArea(c) > 600:
            cv2.drawContours(clean_cyan, [c], -1, 255, -1)

    if np.any(clean_cyan > 0):
        alpha = cv2.GaussianBlur(clean_cyan, (7, 7), 0).astype(np.float32) / 255.0
        roi[:, :, 0] = np.clip(b - 89.25 * alpha, 0, 255).astype(np.uint8)
        roi[:, :, 1] = np.clip(g - 63.0 * alpha, 0, 255).astype(np.uint8)

    # 3. Xóa khung vàng, nhãn xanh lá, đường đỏ/tím
    hsv2 = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    b2, g2, r2 = roi[:, :, 0].astype(np.int16), roi[:, :, 1].astype(np.int16), roi[:, :, 2].astype(np.int16)
    is_yellow = (hsv2[:, :, 0] >= 18) & (hsv2[:, :, 0] <= 34) & (hsv2[:, :, 1] >= 170) & (hsv2[:, :, 2] >= 165)
    is_green_banner = (g2 >= 145) & (r2 < 55) & (b2 < 55)
    is_green_closed = cv2.morphologyEx((is_green_banner.astype(np.uint8)) * 255, cv2.MORPH_CLOSE, np.ones((11, 25), np.uint8)) > 0
    is_red = (r2 > 185) & (r2 > g2 + 75) & (r2 > b2 + 75)
    is_magenta = (r2 > 175) & (b2 > 175) & (g2 < 95)
    grad = cv2.morphologyEx(clean_cyan, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0

    mask_inp = ((is_yellow | is_green_closed | is_red | is_magenta | grad).astype(np.uint8)) * 255
    mask_inp = cv2.dilate(mask_inp, np.ones((3, 3), np.uint8), iterations=1)
    roi = cv2.inpaint(roi, mask_inp, 4, cv2.INPAINT_TELEA)
    f[roi_y0:, :] = roi
    return f


def interp_poly(keyframes, idx: int) -> np.ndarray:
    """Nội suy tuyến tính tọa độ đa giác 4 đỉnh giữa các keyframe."""
    if idx <= keyframes[0][0]:
        return np.array(keyframes[0][1], dtype=np.float32)
    if idx >= keyframes[-1][0]:
        return np.array(keyframes[-1][1], dtype=np.float32)
    for i in range(len(keyframes) - 1):
        f0, p0 = keyframes[i]
        f1, p1 = keyframes[i + 1]
        if f0 <= idx <= f1:
            t = float(idx - f0) / max(1.0, float(f1 - f0))
            arr0 = np.array(p0, dtype=np.float32)
            arr1 = np.array(p1, dtype=np.float32)
            return (1.0 - t) * arr0 + t * arr1
    return np.array(keyframes[-1][1], dtype=np.float32)


def get_133306_path_poly(idx: int):
    """
    Trả về đa giác (4, 2) của tấm bạt đen trên video thực địa robot_record_20261105_133306.mp4,
    hoặc None nếu khung hình đang ở vùng cỏ/đất/xoay đầu xe (Hard Negative cho Path).
    """
    row1_kf = [
        (0, [(420, 240), (555, 240), (630, 719), (160, 719)]),
        (40, [(425, 245), (560, 245), (635, 719), (160, 719)]),
        (80, [(425, 250), (565, 250), (640, 719), (165, 719)]),
        (120, [(430, 265), (575, 265), (650, 719), (165, 719)]),
        (160, [(460, 310), (605, 310), (665, 719), (170, 719)]),
        (200, [(420, 365), (585, 365), (610, 719), (160, 719)]),
        (240, [(410, 420), (565, 420), (595, 719), (180, 719)]),
        (270, [(410, 485), (565, 485), (600, 719), (240, 719)]),
        (286, [(395, 575), (590, 575), (620, 719), (330, 719)]),
    ]
    row2_kf = [
        (448, [(760, 230), (950, 230), (1180, 719), (440, 719)]),
        (470, [(590, 240), (735, 240), (980, 719), (220, 719)]),
        (500, [(575, 250), (720, 250), (960, 719), (210, 719)]),
        (530, [(475, 260), (635, 260), (880, 719), (150, 719)]),
        (560, [(465, 280), (625, 280), (850, 719), (170, 719)]),
        (590, [(525, 375), (685, 375), (860, 719), (240, 719)]),
        (610, [(430, 420), (620, 420), (790, 719), (210, 719)]),
        (625, [(420, 460), (615, 460), (760, 719), (210, 719)]),
        (642, [(380, 550), (600, 550), (720, 719), (240, 719)]),
    ]
    row3_kf = [
        (980, [(440, 220), (570, 220), (620, 719), (350, 719)]),
        (1000, [(470, 230), (610, 230), (690, 719), (380, 719)]),
        (1015, [(530, 280), (690, 280), (790, 719), (440, 719)]),
        (1030, [(530, 380), (730, 380), (830, 719), (420, 719)]),
        (1050, [(560, 390), (820, 390), (940, 719), (450, 719)]),
        (1080, [(580, 430), (860, 430), (980, 719), (490, 719)]),
        (1110, [(580, 435), (865, 435), (985, 719), (495, 719)]),
        (1157, [(580, 435), (865, 435), (985, 719), (495, 719)]),
    ]

    if 0 <= idx <= 286:
        return interp_poly(row1_kf, idx)
    if 448 <= idx <= 642:
        return interp_poly(row2_kf, idx)
    if 980 <= idx <= 1157:
        return interp_poly(row3_kf, idx)
    return None


def get_124600_path_poly(idx: int):
    """Trả về đa giác bạt đen trên video thực địa robot_record_20261105_124600.mp4."""
    row1_kf = [
        (0, [(410, 235), (555, 235), (630, 719), (160, 719)]),
        (120, [(410, 240), (555, 240), (630, 719), (160, 719)]),
        (200, [(395, 290), (555, 290), (600, 719), (150, 719)]),
        (280, [(390, 370), (555, 370), (590, 719), (160, 719)]),
        (320, [(360, 470), (545, 470), (575, 719), (200, 719)]),
    ]
    row2_kf = [
        (520, [(790, 95), (1050, 95), (1275, 719), (380, 719)]),
        (560, [(670, 110), (960, 110), (1250, 719), (290, 719)]),
        (600, [(310, 140), (740, 140), (1200, 719), (210, 719)]),
        (680, [(340, 140), (820, 140), (1150, 719), (170, 719)]),
        (760, [(350, 140), (830, 140), (1120, 719), (160, 719)]),
        (960, [(350, 140), (830, 140), (1120, 719), (160, 719)]),
    ]
    if 0 <= idx <= 320:
        return interp_poly(row1_kf, idx)
    if 520 <= idx <= 960:
        return interp_poly(row2_kf, idx)
    return None


def clean_record_frame_and_extract_poly(frame: np.ndarray, frame_idx: int):
    """
    Khôi phục khung hình gốc từ robot_record_20261102_104137.mp4 (record_1.mp4)
    và trích xuất đa giác (polygon) chuẩn xác của tấm bạt đen (frame 0..73).
    """
    h, w = frame.shape[:2]
    f = frame.copy()

    f[:44, :] = cv2.inpaint(
        f[:60, :],
        np.vstack([np.ones((44, w), np.uint8) * 255, np.zeros((16, w), np.uint8)]),
        5,
        cv2.INPAINT_TELEA
    )[:44, :]

    if frame_idx > 73:
        return f, None

    roi_y0 = 350
    roi = f[roi_y0:, :].copy()
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    b, g, r = roi[:, :, 0].astype(np.int16), roi[:, :, 1].astype(np.int16), roi[:, :, 2].astype(np.int16)

    is_blue = (b > r + 15) & (g > r + 25) & (hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 115)
    is_blue[:, :70] = False
    is_blue[:, 1130:] = False
    blue_mask = cv2.morphologyEx((is_blue.astype(np.uint8)) * 255, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    hull_mask = np.zeros_like(blue_mask)
    cnts, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) > 1500:
            hull = cv2.convexHull(c)
            cv2.drawContours(hull_mask, [hull], -1, 255, -1)

    is_dark = (hsv[:, :, 2] < 65) & (hsv[:, :, 1] < 165)
    is_dark[:, :75] = False
    is_dark[:, 1125:] = False
    if frame_idx >= 61:
        is_dark[:(660 - roi_y0), :] = False

    combined = cv2.morphologyEx(
        hull_mask | ((is_dark.astype(np.uint8)) * 255),
        cv2.MORPH_CLOSE,
        np.ones((9, 25), np.uint8)
    )
    cnts2, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    poly_pts = None
    poly_mask = np.zeros_like(blue_mask)
    if cnts2:
        c2 = max(cnts2, key=cv2.contourArea)
        if cv2.contourArea(c2) > 2500:
            hull2 = cv2.convexHull(c2)
            eps = 0.008 * cv2.arcLength(hull2, True)
            approx = cv2.approxPolyDP(hull2, eps, True)
            cv2.drawContours(poly_mask, [approx], -1, 255, -1)
            poly_pts = approx.reshape(-1, 2).copy()
            poly_pts[:, 1] += roi_y0

    if np.any(hull_mask > 0):
        offset = np.array([78.35, 90.66, 43.69], dtype=np.float32)
        roi_f = roi.astype(np.float32)
        restored = np.clip((roi_f - offset) / 0.55, 0, 255).astype(np.uint8)
        blue_actual = (hull_mask > 0) & ((b > r + 10) | (g > r + 15))
        alpha = cv2.GaussianBlur((blue_actual.astype(np.uint8)) * 255, (5, 5), 0)[:, :, None].astype(np.float32) / 255.0
        roi_clean = (restored * alpha + roi * (1.0 - alpha)).astype(np.uint8)

        is_yellow_box = (hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 30) & (hsv[:, :, 1] >= 195) & (hsv[:, :, 2] >= 175)
        is_green_circle = (hsv[:, :, 0] >= 50) & (hsv[:, :, 0] <= 72) & (hsv[:, :, 1] >= 180) & (hsv[:, :, 2] >= 150)
        is_red_line = ((hsv[:, :, 0] <= 8) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] >= 175) & (hsv[:, :, 2] >= 140)
        is_white_dot = (poly_mask > 0) & (hsv[:, :, 1] < 35) & (hsv[:, :, 2] > 210)
        grad = cv2.morphologyEx(hull_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0

        inpaint_mask = ((is_yellow_box | is_green_circle | is_red_line | is_white_dot | grad).astype(np.uint8)) * 255
        inpaint_mask = cv2.dilate(inpaint_mask, np.ones((3, 3), np.uint8), iterations=1)
        roi_clean = cv2.inpaint(roi_clean, inpaint_mask, 4, cv2.INPAINT_TELEA)
        f[roi_y0:, :] = roi_clean

    return f, poly_pts


def extract_poly_with_teacher(model, frame: np.ndarray, min_area: float = 4000.0):
    """Sử dụng model teacher kết hợp lọc hình học Convex Hull để lấy nhãn polygon sạch."""
    h, w = frame.shape[:2]
    res = model.predict(frame, imgsz=416, conf=0.25, verbose=False)[0]
    if res.masks is None or len(res.masks.data) == 0:
        return None
    m = res.masks.data[0].cpu().numpy()
    m_u8 = cv2.resize((m > 0.5).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    hull = cv2.convexHull(c)
    eps = 0.008 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, eps, True)
    if len(approx) < 3:
        return None
    return approx.reshape(-1, 2)


def transform_frame_and_poly(frame: np.ndarray, poly: np.ndarray, aug_type: str):
    """
    Biến đổi hình học (góc nhìn nghiêng, xoay, lật, phối cảnh) và ánh sáng
    (chói lóa, vàng ấm, tối, xanh mát) cho cả ảnh và tọa độ đa giác.
    """
    h, w = frame.shape[:2]
    img = frame.copy()
    pts = poly.astype(np.float32).copy() if poly is not None else None

    if "flip" in aug_type:
        img = cv2.flip(img, 1)
        if pts is not None:
            pts[:, 0] = (w - 1) - pts[:, 0]

    if "rot_left" in aug_type or "rot_right" in aug_type:
        angle = random.uniform(10.0, 18.0) if "rot_left" in aug_type else random.uniform(-18.0, -10.0)
        scale = random.uniform(0.95, 1.08)
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        if pts is not None:
            ones = np.ones((len(pts), 1), dtype=np.float32)
            pts = (np.hstack([pts, ones]) @ M.T)

    if "persp_steep" in aug_type or "persp_oblique" in aug_type:
        src_quad = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        if "persp_steep" in aug_type:
            dy = random.uniform(0.08, 0.16) * h
            dx = random.uniform(0.05, 0.12) * w
            dst_quad = np.float32([[dx, dy], [w - dx, dy], [w, h], [0, h]])
        else:
            shift = random.choice([-1, 1]) * random.uniform(0.08, 0.15) * w
            dst_quad = np.float32([
                [max(0, shift), h * 0.05],
                [min(w, w + shift), h * 0.05],
                [w, h],
                [0, h]
            ])
        H_mat = cv2.getPerspectiveTransform(src_quad, dst_quad)
        img = cv2.warpPerspective(img, H_mat, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        if pts is not None:
            pts_reshaped = pts.reshape(-1, 1, 2)
            pts = cv2.perspectiveTransform(pts_reshaped, H_mat).reshape(-1, 2)

    img_f = img.astype(np.float32)
    if "warm" in aug_type:
        img_f[:, :, 0] *= random.uniform(0.72, 0.85)
        img_f[:, :, 1] *= random.uniform(1.02, 1.12)
        img_f[:, :, 2] *= random.uniform(1.05, 1.18)
    elif "cool" in aug_type:
        img_f[:, :, 0] *= random.uniform(1.10, 1.25)
        img_f[:, :, 2] *= random.uniform(0.82, 0.92)
    elif "dim" in aug_type:
        gamma = random.uniform(1.35, 1.65)
        img_f = 255.0 * np.power(np.clip(img_f / 255.0, 0, 1), gamma)
    elif "bright_glare" in aug_type:
        gamma = random.uniform(0.68, 0.82)
        img_f = 255.0 * np.power(np.clip(img_f / 255.0, 0, 1), gamma)
        if pts is not None and len(pts) >= 3:
            cx = int(np.mean(pts[:, 0]) + random.uniform(-60, 60))
            glare_mask = np.zeros((h, w), dtype=np.float32)
            cv2.line(glare_mask, (cx, int(np.min(pts[:, 1]))), (cx + random.randint(-40, 40), int(np.max(pts[:, 1]))), 1.0, thickness=random.randint(25, 55))
            glare_mask = cv2.GaussianBlur(glare_mask, (51, 51), 0)
            img_f += glare_mask[:, :, None] * random.uniform(55.0, 95.0)

    img = np.clip(img_f, 0, 255).astype(np.uint8)

    if pts is not None:
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, [pts.astype(np.int32)], 255)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return img, None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 1500:
            return img, None
        hull = cv2.convexHull(c)
        eps = 0.008 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, eps, True)
        pts = approx.reshape(-1, 2).astype(np.float32)

    return img, pts


def save_yolo_sample(img: np.ndarray, poly: np.ndarray, split: str, sample_name: str):
    """Lưu ảnh (resize về IMG_SIZE x IMG_SIZE) và file label .txt chuẩn YOLO-seg."""
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img_path = os.path.join(DATASET_DIR, "images", split, f"{sample_name}.jpg")
    lbl_path = os.path.join(DATASET_DIR, "labels", split, f"{sample_name}.txt")

    cv2.imwrite(img_path, img_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 94])

    if poly is None or len(poly) < 3:
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("")
    else:
        norm_coords = []
        for x, y in poly:
            nx = np.clip(float(x) / float(w), 0.0, 1.0)
            ny = np.clip(float(y) / float(h), 0.0, 1.0)
            norm_coords.append(f"{nx:.6f} {ny:.6f}")
        line = "0 " + " ".join(norm_coords) + "\n"
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write(line)


def main():
    from ultralytics import YOLO

    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)

    for split in ["train", "val"]:
        os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)

    teacher = YOLO("models/nhamang_batden_seg.pt")
    counts = {"train_pos": 0, "train_neg": 0, "val_pos": 0, "val_neg": 0}

    # =========================================================================
    # 1. XỬ LÝ VIDEO THỰC ĐỊA CHÍNH: robot_record_20261105_133306.mp4 (1158 frames)
    # =========================================================================
    v133306_path = "output_qr/recordings/robot_record_20261105_133306.mp4"
    if os.path.exists(v133306_path):
        cap = cv2.VideoCapture(v133306_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[DATA] Đang trích xuất video thực địa chính: {v133306_path} ({n_frames} frames)...")
        for idx in range(0, n_frames):
            ret, raw_f = cap.read()
            if not ret:
                break
            poly = get_133306_path_poly(idx)
            # Lấy mẫu dày (mỗi 4 frame) cho vùng có bạt, mỗi 8 frame cho vùng âm tính (cỏ/đất/bóng cây)
            if poly is not None and idx % 4 == 0:
                clean_f = de_overlay_outdoor_frame(raw_f)
                split = "val" if (idx % 20 == 0) else "train"
                save_yolo_sample(clean_f, poly, split, f"out133306_clean_f{idx:04d}")
                counts[f"{split}_pos"] += 1
                # Lưu cả bản raw (mỗi 8 frame) để chạy replay trực tiếp trên video ghi sẵn đạt độ chính xác tối đa
                if idx % 8 == 0:
                    save_yolo_sample(raw_f, poly, split, f"out133306_raw_f{idx:04d}")
                    counts[f"{split}_pos"] += 1
                if split == "train" and idx % 8 == 0:
                    for aug_t in random.sample(
                        ["flip_cool", "rot_left_bright_glare", "rot_right_dim", "persp_oblique_warm"],
                        2,
                    ):
                        aug_img, aug_poly = transform_frame_and_poly(clean_f, poly, aug_t)
                        if aug_poly is not None:
                            save_yolo_sample(aug_img, aug_poly, "train", f"out133306_{aug_t}_f{idx:04d}")
                            counts["train_pos"] += 1
            elif poly is None and (295 <= idx <= 440 or 655 <= idx <= 965) and idx % 6 == 0:
                clean_f = de_overlay_outdoor_frame(raw_f)
                split = "val" if (idx % 24 == 0) else "train"
                save_yolo_sample(clean_f, None, split, f"out133306_neg_clean_f{idx:04d}")
                counts[f"{split}_neg"] += 1
                if idx % 12 == 0:
                    save_yolo_sample(raw_f, None, split, f"out133306_neg_raw_f{idx:04d}")
                    counts[f"{split}_neg"] += 1
        cap.release()

    # =========================================================================
    # 2. XỬ LÝ VIDEO THỰC ĐỊA PHỤ: robot_record_20261105_124600.mp4 (965 frames)
    # =========================================================================
    v124600_path = "output_qr/recordings/robot_record_20261105_124600.mp4"
    if os.path.exists(v124600_path):
        cap = cv2.VideoCapture(v124600_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[DATA] Đang trích xuất video thực địa: {v124600_path} ({n_frames} frames)...")
        for idx in range(0, n_frames):
            ret, raw_f = cap.read()
            if not ret:
                break
            poly = get_124600_path_poly(idx)
            if poly is not None and idx % 8 == 0:
                clean_f = de_overlay_outdoor_frame(raw_f)
                split = "val" if (idx % 40 == 0) else "train"
                save_yolo_sample(clean_f, poly, split, f"out124600_clean_f{idx:04d}")
                counts[f"{split}_pos"] += 1
                if split == "train" and idx % 16 == 0:
                    aug_img, aug_poly = transform_frame_and_poly(clean_f, poly, "flip_bright_glare")
                    if aug_poly is not None:
                        save_yolo_sample(aug_img, aug_poly, "train", f"out124600_aug_f{idx:04d}")
                        counts["train_pos"] += 1
            elif poly is None and (345 <= idx <= 500) and idx % 10 == 0:
                clean_f = de_overlay_outdoor_frame(raw_f)
                split = "val" if (idx % 40 == 0) else "train"
                save_yolo_sample(clean_f, None, split, f"out124600_neg_f{idx:04d}")
                counts[f"{split}_neg"] += 1
        cap.release()

    # =========================================================================
    # 3. XỬ LÝ VIDEO TRONG NHÀ: robot_record_20261102_104137.mp4 (record_1.mp4)
    # =========================================================================
    rec1_path = "output_qr/recordings/robot_record_20261102_104137.mp4"
    if os.path.exists(rec1_path):
        cap = cv2.VideoCapture(rec1_path)
        for idx in range(0, 74, 2):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, raw_f = cap.read()
            if not ret:
                continue
            clean_f, poly = clean_record_frame_and_extract_poly(raw_f, idx)
            if poly is None:
                continue
            split = "val" if (idx % 10 == 0) else "train"
            save_yolo_sample(clean_f, poly, split, f"rec1_clean_f{idx:03d}")
            counts[f"{split}_pos"] += 1
            if idx <= 60 and idx % 4 == 0:
                save_yolo_sample(raw_f, poly, split, f"rec1_orig_f{idx:03d}")
                counts[f"{split}_pos"] += 1
            if split == "train":
                for a_idx, aug_t in enumerate(random.sample(["flip_cool", "rot_left_bright_glare", "rot_right_dim", "persp_oblique_warm"], 2)):
                    aug_img, aug_poly = transform_frame_and_poly(clean_f, poly, aug_t)
                    if aug_poly is not None:
                        save_yolo_sample(aug_img, aug_poly, "train", f"rec1_aug{a_idx}_{aug_t}_f{idx:03d}")
                        counts["train_pos"] += 1

        for idx in range(85, 844, 24):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, raw_f = cap.read()
            if not ret:
                continue
            clean_f, _ = clean_record_frame_and_extract_poly(raw_f, idx)
            split = "val" if (idx % 5 == 0) else "train"
            save_yolo_sample(clean_f, None, split, f"rec1_neg_f{idx:03d}")
            counts[f"{split}_neg"] += 1
        cap.release()

    # =========================================================================
    # 4. XỬ LÝ VIDEO line_qr.mp4, nhamang_20261025_030235.mp4, thamden.mp4
    # =========================================================================
    if os.path.exists("line_qr.mp4"):
        cap = cv2.VideoCapture("line_qr.mp4")
        for idx in range(0, 455, 16):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if not ret:
                continue
            poly = extract_poly_with_teacher(teacher, f)
            if poly is not None:
                split = "val" if (idx % 80 == 0) else "train"
                save_yolo_sample(f, poly, split, f"lineqr_pos_f{idx:04d}")
                counts[f"{split}_pos"] += 1
        for idx in range(485, 1005, 50):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if not ret:
                continue
            split = "val" if (idx % 100 == 0) else "train"
            save_yolo_sample(f, None, split, f"lineqr_neg_f{idx:04d}")
            counts[f"{split}_neg"] += 1
        cap.release()

    if os.path.exists("nhamang_20261025_030235.mp4"):
        cap = cv2.VideoCapture("nhamang_20261025_030235.mp4")
        for idx in range(0, 440, 20):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if not ret:
                continue
            poly = extract_poly_with_teacher(teacher, f)
            if poly is not None:
                split = "val" if (idx % 80 == 0) else "train"
                save_yolo_sample(f, poly, split, f"nhamang_pos_f{idx:03d}")
                counts[f"{split}_pos"] += 1
        for idx in range(465, 605, 25):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if not ret:
                continue
            split = "val" if (idx % 50 == 0) else "train"
            save_yolo_sample(f, None, split, f"nhamang_neg_f{idx:03d}")
            counts[f"{split}_neg"] += 1
        cap.release()

    thamden_list = glob.glob("Quỳnh Robotics*/thamden.mp4")
    if thamden_list:
        cap = cv2.VideoCapture(thamden_list[0])
        for idx in range(0, 640, 40):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if not ret:
                continue
            poly = extract_poly_with_teacher(teacher, f)
            if poly is not None:
                split = "val" if (idx % 160 == 0) else "train"
                save_yolo_sample(f, poly, split, f"thamden_pos_f{idx:03d}")
                counts[f"{split}_pos"] += 1
        cap.release()

    yaml_path = os.path.join(DATASET_DIR, "dataset.yaml")
    abs_ds_dir = os.path.abspath(DATASET_DIR)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {abs_ds_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        f.write("  0: bat_loi_di\n")

    print("============================================================")
    print("HOÀN TẤT AUTO-LABEL & TĂNG CƯỜNG DỮ LIỆU ĐA GÓC ĐỘ / ÁNH SÁNG")
    print(f"Thư mục dataset: {abs_ds_dir}")
    print(f"Thống kê mẫu: {counts}")
    print(f"Tổng Train: {counts['train_pos'] + counts['train_neg']} ảnh | Tổng Val: {counts['val_pos'] + counts['val_neg']} ảnh")
    print("============================================================")


if __name__ == "__main__":
    main()
