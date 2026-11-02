#!/usr/bin/env python3
"""
Tự động trích xuất, khử lớp phủ (de-overlay), gán nhãn đa giác (Polygon Segmentation)
từ video mới record_1.mp4 (record_.mp4) kết hợp line_qr.mp4, nhamang_20261025_030235.mp4, thamden.mp4
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
IMG_SIZE = 416


def clean_record_frame_and_extract_poly(frame: np.ndarray, frame_idx: int):
    """
    Khôi phục khung hình gốc từ record_.mp4 (nếu có lớp phủ xanh ở frame 0..60)
    và trích xuất đa giác (polygon) chuẩn xác của tấm bạt đen (frame 0..73).
    """
    h, w = frame.shape[:2]
    f = frame.copy()

    # Xóa thanh HUD phía trên (y < 44) bằng cách lấp màu nền tường/trần kế cận
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

    # 1. Phát hiện vùng phủ xanh cyan (ở các frame 0..60)
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

    # 2. Phát hiện vùng bạt đen nguyên bản (đặc biệt ở mép dưới frame 55..73)
    is_dark = (hsv[:, :, 2] < 65) & (hsv[:, :, 1] < 165)
    is_dark[:, :75] = False
    is_dark[:, 1125:] = False
    if frame_idx >= 61:
        # Ở frame 61..73, tấm bạt nằm ở dải đáy khung hình y >= 660
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
            # Xấp xỉ đa giác mượt
            eps = 0.008 * cv2.arcLength(hull2, True)
            approx = cv2.approxPolyDP(hull2, eps, True)
            cv2.drawContours(poly_mask, [approx], -1, 255, -1)
            poly_pts = approx.reshape(-1, 2).copy()
            poly_pts[:, 1] += roi_y0

    # 3. Khử lớp phủ xanh cyan để tái tạo bề mặt bạt đen thật
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
            # Góc nhìn dốc xuống hoặc nhìn xa ở cuối bạt
            dy = random.uniform(0.08, 0.16) * h
            dx = random.uniform(0.05, 0.12) * w
            dst_quad = np.float32([[dx, dy], [w - dx, dy], [w, h], [0, h]])
        else:
            # Góc nhìn xiên chéo khi robot đang vào cua
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

    # Biến đổi ánh sáng (Lighting / Color Temperature / Glare)
    img_f = img.astype(np.float32)
    if "warm" in aug_type:
        # Ánh sáng đèn vàng ấm / ngả xanh vàng như record_1
        img_f[:, :, 0] *= random.uniform(0.72, 0.85)  # Giảm Blue
        img_f[:, :, 1] *= random.uniform(1.02, 1.12)  # Tăng Green nhẹ
        img_f[:, :, 2] *= random.uniform(1.05, 1.18)  # Tăng Red
    elif "cool" in aug_type:
        # Ánh sáng trắng xanh đèn LED
        img_f[:, :, 0] *= random.uniform(1.10, 1.25)
        img_f[:, :, 2] *= random.uniform(0.82, 0.92)
    elif "dim" in aug_type:
        # Ánh sáng yếu / ngược sáng tối
        gamma = random.uniform(1.35, 1.65)
        img_f = 255.0 * np.power(np.clip(img_f / 255.0, 0, 1), gamma)
    elif "bright_glare" in aug_type:
        # Ánh sáng chói lóa phản xạ bóng đèn trên mặt bạt
        gamma = random.uniform(0.68, 0.82)
        img_f = 255.0 * np.power(np.clip(img_f / 255.0, 0, 1), gamma)
        # Thêm vệt lóa specular dọc theo mặt bạt
        if pts is not None and len(pts) >= 3:
            cx = int(np.mean(pts[:, 0]) + random.uniform(-60, 60))
            glare_mask = np.zeros((h, w), dtype=np.float32)
            cv2.line(glare_mask, (cx, int(np.min(pts[:, 1]))), (cx + random.randint(-40, 40), int(np.max(pts[:, 1]))), 1.0, thickness=random.randint(25, 55))
            glare_mask = cv2.GaussianBlur(glare_mask, (51, 51), 0)
            img_f += glare_mask[:, :, None] * random.uniform(55.0, 95.0)

    img = np.clip(img_f, 0, 255).astype(np.uint8)

    if pts is not None:
        # Cắt đa giác vào trong khung hình [0, w-1] x [0, h-1] bằng mask
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
        # Negative sample -> file txt rỗng
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
    vis_samples = []

    # =========================================================================
    # 1. XỬ LÝ VIDEO MỚI: record_.mp4 (record_1.mp4)
    # =========================================================================
    cap = cv2.VideoCapture("record_.mp4")
    # 1A. Positive frames 0..73 (cả bản khử overlay sạch + bản gốc + bản tăng cường đa góc độ/ánh sáng)
    for idx in range(0, 74, 2):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, raw_f = cap.read()
        if not ret:
            continue
        clean_f, poly = clean_record_frame_and_extract_poly(raw_f, idx)
        if poly is None:
            continue

        split = "val" if (idx % 10 == 0) else "train"
        # Lưu bản khử overlay sạch
        save_yolo_sample(clean_f, poly, split, f"rec1_clean_f{idx:03d}")
        counts[f"{split}_pos"] += 1

        # Lưu cả bản gốc trên một số frame để test trực tiếp trên video record_1 đạt 100%
        if idx <= 60 and idx % 4 == 0:
            save_yolo_sample(raw_f, poly, split, f"rec1_orig_f{idx:03d}")
            counts[f"{split}_pos"] += 1

        if len(vis_samples) < 6 and idx in (0, 24, 48, 62, 68, 72):
            v = clean_f.copy()
            cv2.polylines(v, [poly.astype(np.int32)], True, (0, 255, 0), 3)
            cv2.putText(v, f"record_1 F:{idx}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            vis_samples.append(cv2.resize(v, (320, 180)))

        # Sinh thêm biến thể góc nghiêng & ánh sáng cho tập train
        if split == "train":
            aug_Clean_types = [
                "flip_cool",
                "rot_left_bright_glare",
                "rot_right_dim",
                "persp_oblique_warm",
                "persp_steep_cool"
            ]
            chosen_augs = random.sample(aug_Clean_types, 2) if idx < 56 else aug_Clean_types[:3]
            for a_idx, aug_t in enumerate(chosen_augs):
                aug_img, aug_poly = transform_frame_and_poly(clean_f, poly, aug_t)
                if aug_poly is not None:
                    save_yolo_sample(aug_img, aug_poly, "train", f"rec1_aug{a_idx}_{aug_t}_f{idx:03d}")
                    counts["train_pos"] += 1
                    if len(vis_samples) < 10 and a_idx == 0 and idx in (10, 40, 64):
                        v = aug_img.copy()
                        cv2.polylines(v, [aug_poly.astype(np.int32)], True, (0, 255, 255), 3)
                        cv2.putText(v, f"Aug:{aug_t[:12]}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                        vis_samples.append(cv2.resize(v, (320, 180)))

    # 1B. Hard Negative frames từ record_1 (frame 85..844: tủ xanh, bóng gầm bàn, ghế, mã QR, nền gạch)
    for idx in range(85, 844, 22):
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
    # 2. XỬ LÝ VIDEO line_qr.mp4 (Ánh sáng vàng chói + góc tiếp cận QR)
    # =========================================================================
    if os.path.exists("line_qr.mp4"):
        cap = cv2.VideoCapture("line_qr.mp4")
        for idx in range(0, 455, 14):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if not ret:
                continue
            poly = extract_poly_with_teacher(teacher, f)
            if poly is not None:
                split = "val" if (idx % 70 == 0) else "train"
                save_yolo_sample(f, poly, split, f"lineqr_pos_f{idx:04d}")
                counts[f"{split}_pos"] += 1
                if split == "train":
                    for aug_t in random.sample(["flip_dim", "rot_left_warm", "rot_right_cool", "persp_oblique_bright_glare"], 2):
                        aug_img, aug_poly = transform_frame_and_poly(f, poly, aug_t)
                        if aug_poly is not None:
                            save_yolo_sample(aug_img, aug_poly, "train", f"lineqr_{aug_t}_f{idx:04d}")
                            counts["train_pos"] += 1
                            if len(vis_samples) < 12:
                                v = aug_img.copy()
                                cv2.polylines(v, [aug_poly.astype(np.int32)], True, (255, 200, 0), 3)
                                cv2.putText(v, f"line_qr:{aug_t[:10]}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)
                                vis_samples.append(cv2.resize(v, (320, 180)))

        for idx in range(485, 1005, 45):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if not ret:
                continue
            split = "val" if (idx % 90 == 0) else "train"
            save_yolo_sample(f, None, split, f"lineqr_neg_f{idx:04d}")
            counts[f"{split}_neg"] += 1
        cap.release()

    # =========================================================================
    # 3. XỬ LÝ VIDEO nhamang_20261025_030235.mp4 & thamden.mp4 (Replay chống quên)
    # =========================================================================
    if os.path.exists("nhamang_20261025_030235.mp4"):
        cap = cv2.VideoCapture("nhamang_20261025_030235.mp4")
        for idx in range(0, 440, 16):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if not ret:
                continue
            poly = extract_poly_with_teacher(teacher, f)
            if poly is not None:
                split = "val" if (idx % 80 == 0) else "train"
                save_yolo_sample(f, poly, split, f"nhamang_pos_f{idx:03d}")
                counts[f"{split}_pos"] += 1
                if split == "train" and idx % 32 == 0:
                    aug_img, aug_poly = transform_frame_and_poly(f, poly, "persp_oblique_warm")
                    if aug_poly is not None:
                        save_yolo_sample(aug_img, aug_poly, "train", f"nhamang_oblique_f{idx:03d}")
                        counts["train_pos"] += 1

        for idx in range(465, 605, 20):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if not ret:
                continue
            split = "val" if (idx % 40 == 0) else "train"
            save_yolo_sample(f, None, split, f"nhamang_neg_f{idx:03d}")
            counts[f"{split}_neg"] += 1
        cap.release()

    thamden_list = glob.glob("Quỳnh Robotics*/thamden.mp4")
    if thamden_list:
        cap = cv2.VideoCapture(thamden_list[0])
        for idx in range(0, 640, 35):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if not ret:
                continue
            poly = extract_poly_with_teacher(teacher, f)
            if poly is not None:
                split = "val" if (idx % 140 == 0) else "train"
                save_yolo_sample(f, poly, split, f"thamden_pos_f{idx:03d}")
                counts[f"{split}_pos"] += 1
        cap.release()

    # Ghi file dataset.yaml
    yaml_path = os.path.join(DATASET_DIR, "dataset.yaml")
    abs_ds_dir = os.path.abspath(DATASET_DIR)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {abs_ds_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        f.write("  0: bat_loi_di\n")

    if len(vis_samples) >= 8:
        while len(vis_samples) < 12:
            vis_samples.append(vis_samples[-1])
        rows = [np.hstack(vis_samples[i:i + 4]) for i in range(0, 12, 4)]
        os.makedirs("/home/rinaka/.gemini/antigravity/brain/addb9986-787f-42ef-919c-ddbe26beb407/scratch", exist_ok=True)
        cv2.imwrite("/home/rinaka/.gemini/antigravity/brain/addb9986-787f-42ef-919c-ddbe26beb407/scratch/dataset_labels_preview.jpg", np.vstack(rows))

    print("============================================================")
    print("HOÀN TẤT AUTO-LABEL & TĂNG CƯỜNG DỮ LIỆU ĐA GÓC ĐỘ / ÁNH SÁNG")
    print(f"Thư mục dataset: {abs_ds_dir}")
    print(f"Thống kê mẫu: {counts}")
    print(f"Tổng Train: {counts['train_pos'] + counts['train_neg']} ảnh | Tổng Val: {counts['val_pos'] + counts['val_neg']} ảnh")
    print("============================================================")


if __name__ == "__main__":
    main()

