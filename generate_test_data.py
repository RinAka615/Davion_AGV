#!/usr/bin/env python3
"""
Tạo bộ dữ liệu kiểm thử QR độc lập và chuẩn xác (Ground Truth Test Dataset):
1. Mã QR tiêu chuẩn (Standard QR)
2. Mã QR nghiêng phối cảnh & xoay góc (Perspective Tilted QR)
3. Đa mã QR trong cùng một khung hình (Multi-QR)
4. Khung hình âm tính từ nhà màng thực tế (Negative Sample - No QR)
5. Video kiểm thử tổng hợp (Multi-scene Test Video 720x1280 @ 30fps)
"""

import os
import cv2
import numpy as np
import qrcode
from PIL import Image

OUT_DIR = "test_qr_dataset"
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_VID_PATH = "Quỳnh Robotics [13-10-2026 20_33]/2aobor4lgjkrognqzs1jz63kldqrlxveqvfef1rw4.mp4"


def make_qr_image(text: str, box_size: int = 10, border: int = 4) -> np.ndarray:
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=border,
    )
    qr.add_data(text)
    qr.make(fit=True)
    img_pil = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def generate_standard_sample():
    text = "https://github.com/warlord/RobotNongNghiep"
    qr_img = make_qr_image(text, box_size=8, border=4)
    # Đặt vào nền văn phòng/bàn làm việc 720x1280
    bg = np.full((1280, 720, 3), 235, dtype=np.uint8)
    # Thêm vân xám nhẹ
    cv2.rectangle(bg, (40, 200), (680, 1080), (220, 220, 220), -1)
    
    qh, qw = qr_img.shape[:2]
    y_off = (1280 - qh) // 2
    x_off = (720 - qw) // 2
    bg[y_off:y_off+qh, x_off:x_off+qw] = qr_img

    cv2.putText(bg, "TEST 01: STANDARD QR CODE", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
    path = os.path.join(OUT_DIR, "01_standard_qr.png")
    cv2.imwrite(path, bg)
    print(f"[OK] Generated: {path} -> Content: {text}")
    return text


def generate_tilted_sample():
    text = "QR-MELON-A3-TILTED"
    qr_img = make_qr_image(text, box_size=8, border=4)
    qh, qw = qr_img.shape[:2]

    # Điểm gốc
    src_pts = np.float32([
        [0, 0],
        [qw, 0],
        [qw, qh],
        [0, qh]
    ])
    # Biến dạng phối cảnh thực tế (mã dán nghiêng 25 độ, góc nhìn máy ảnh từ trên xuống)
    dst_pts = np.float32([
        [80, 50],
        [qw + 40, 90],
        [qw + 10, qh + 30],
        [40, qh - 10]
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_qr = cv2.warpPerspective(qr_img, M, (qw + 120, qh + 80), borderValue=(255, 255, 255))

    bg = np.full((1280, 720, 3), 220, dtype=np.uint8)
    wh, ww = warped_qr.shape[:2]
    y_off = (1280 - wh) // 2
    x_off = (720 - ww) // 2
    bg[y_off:y_off+wh, x_off:x_off+ww] = warped_qr

    # Thêm bóng bàn làm việc nhẹ
    cv2.rectangle(bg, (x_off - 20, y_off - 20), (x_off + ww + 20, y_off + wh + 20), (200, 200, 200), 2)

    cv2.putText(bg, "TEST 02: TILTED PERSPECTIVE QR", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (40, 40, 40), 2)
    path = os.path.join(OUT_DIR, "02_tilted_perspective_qr.png")
    cv2.imwrite(path, bg)
    print(f"[OK] Generated: {path} -> Content: {text}")
    return text


def generate_multi_qr_sample():
    codes = ["QR-BED-ROW-01", "QR-BED-ROW-02", "QR-BED-ROW-03"]
    bg = np.full((1280, 720, 3), 245, dtype=np.uint8)

    positions = [
        (60, 260, codes[0]),
        (220, 620, codes[1]),
        (380, 960, codes[2])
    ]

    for x, y, code in positions:
        qr = make_qr_image(code, box_size=5, border=3)
        qh, qw = qr.shape[:2]
        bg[y:y+qh, x:x+qw] = qr
        cv2.putText(bg, f"ID: {code}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

    cv2.putText(bg, "TEST 03: MULTI-QR IN SINGLE FRAME", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    path = os.path.join(OUT_DIR, "03_multi_qr.png")
    cv2.imwrite(path, bg)
    print(f"[OK] Generated: {path} -> Contents: {codes}")
    return codes


def extract_negative_sample():
    cap = cv2.VideoCapture(SAMPLE_VID_PATH)
    if not cap.isOpened():
        print("[WARN] Could not open sample video to extract negative frame, creating synthetic negative")
        bg = np.full((1280, 720, 3), (40, 120, 40), dtype=np.uint8)
    else:
        # Lấy frame 300 (ở giữa nhà màng)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
        ret, frame = cap.read()
        cap.release()
        if ret:
            bg = frame
        else:
            bg = np.full((1280, 720, 3), (40, 120, 40), dtype=np.uint8)

    path = os.path.join(OUT_DIR, "04_negative_greenhouse.png")
    cv2.imwrite(path, bg)
    print(f"[OK] Extracted Negative Sample (No QR): {path}")
    return path


def generate_test_video():
    """Tạo video 300 frames (10 giây @ 30 FPS, 720x1280) tổng hợp 4 kịch bản."""
    video_out = os.path.join(OUT_DIR, "test_qr_multiscene.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_out, fourcc, 30.0, (720, 1280))

    img_neg = cv2.imread(os.path.join(OUT_DIR, "04_negative_greenhouse.png"))
    img_std = cv2.imread(os.path.join(OUT_DIR, "01_standard_qr.png"))
    img_tilt = cv2.imread(os.path.join(OUT_DIR, "02_tilted_perspective_qr.png"))
    img_multi = cv2.imread(os.path.join(OUT_DIR, "03_multi_qr.png"))

    scenes = [
        (img_neg, 75, "SCENE 1: GreenHouse Negative (0-2.5s) - NO QR"),
        (img_std, 75, "SCENE 2: Standard High-Contrast QR (2.5-5s)"),
        (img_tilt, 75, "SCENE 3: Tilted Perspective QR (5-7.5s)"),
        (img_multi, 75, "SCENE 4: Multi-QR 3 Codes (7.5-10s)")
    ]

    frame_count = 0
    for img, duration_frames, label in scenes:
        resized = cv2.resize(img, (720, 1280))
        for _ in range(duration_frames):
            frame = resized.copy()
            # Thêm watermark thời gian
            t_sec = frame_count / 30.0
            cv2.putText(frame, f"{label} | t={t_sec:.2f}s", (30, 1240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            out.write(frame)
            frame_count += 1

    out.release()
    print(f"[OK] Generated Test Video: {video_out} (300 frames, 10.0s, 720x1280 @ 30FPS)")


if __name__ == "__main__":
    generate_standard_sample()
    generate_tilted_sample()
    generate_multi_qr_sample()
    extract_negative_sample()
    generate_test_video()
