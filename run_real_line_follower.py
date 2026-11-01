#!/usr/bin/env python3
"""
run_real_line_follower.py
Chạy thực tế bài toán Robot bám vạch / bạt đen nhà màng siêu nhẹ.
Tự động ưu tiên NCNN engine (50+ FPS) trên Raspberry Pi 5 / PC CPU.
"""

import os
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    video_source = "nhamang_20261025_030235.mp4"
    if len(sys.argv) > 1:
        video_source = sys.argv[1]

    # Kiểm tra model NCNN trước, fallback về .pt
    ncnn_dir = "models/batden_fast_256_ncnn_model"
    pt_file = "models/batden_fast_256.pt"

    if os.path.exists(ncnn_dir):
        model_path = ncnn_dir
        engine_type = "NCNN (C++ ARM NEON Fast Engine)"
    elif os.path.exists(pt_file):
        model_path = pt_file
        engine_type = "PyTorch CPU"
    else:
        print("[!] Không tìm thấy model fast trong models/")
        return

    print("=" * 65)
    print("   ROBOT BÁM VẠCH NHÀ MÀNG - CHẠY THỰC TẾ HIGH-SPEED")
    print(f"   Engine: {engine_type}")
    print(f"   Model : {model_path}")
    print(f"   Nguồn : {video_source}")
    print("=" * 65)

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"[!] Không thể mở video/camera: {video_source}")
        return

    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs("output", exist_ok=True)
    out_video_path = "output/real_run_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, 25.0, (w_orig, h_orig))

    print(f"[*] Kích thước gốc: {w_orig}x{h_orig} | Tổng frame: {total_frames}")
    print("[*] Đang khởi động suy luận thời gian thực...")

    times = []
    frame_idx = 0
    cam_cx = int(w_orig / 2)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= 150:  # Chạy 150 frames thực tế
                break

            t0 = time.perf_counter()
            # 1. Suy luận AI (NCNN / PyTorch)
            results = model.predict(source=frame, imgsz=256, conf=0.25, verbose=False)[0]
            dt = (time.perf_counter() - t0) * 1000.0
            times.append(dt)

            lateral_offset = None
            offset_text = "N/A"

            # 2. Xử lý hình học đường bạt & centerline
            if results.masks is not None and len(results.masks.data) > 0:
                mask = results.masks.data[0].cpu().numpy().astype(np.uint8)
                mask_full = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

                # Vẽ overlay vùng bạt (màu xanh cyan bán trong suốt)
                overlay = frame.copy()
                overlay[mask_full > 0] = [200, 150, 0]
                frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

                # Tìm bounding box và tâm đường
                y_indices, x_indices = np.where(mask_full > 0)
                if len(x_indices) > 0:
                    x1, x2 = np.min(x_indices), np.max(x_indices)
                    y1, y2 = np.min(y_indices), np.max(y_indices)
                    box_cx = int((x1 + x2) / 2)
                    box_cy = int((y1 + y2) / 2)

                    lateral_offset = float(box_cx - cam_cx)
                    if lateral_offset < -3.0:
                        offset_text = f"Lệch trái {abs(lateral_offset):.1f} px"
                    elif lateral_offset > 3.0:
                        offset_text = f"Lệch phải {lateral_offset:.1f} px"
                    else:
                        offset_text = "Đúng tâm (0 px)"

                    # Vẽ đường trục tim (Centerline - Đường đỏ)
                    cv2.line(frame, (box_cx, y1), (box_cx, y2), (0, 0, 255), 3, cv2.LINE_AA)
                    # Vẽ tâm mục tiêu (Chấm vàng)
                    cv2.circle(frame, (box_cx, box_cy), 8, (0, 255, 255), -1, cv2.LINE_AA)
                    # Vẽ khung bao bạt (Hộp xanh lá)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Vẽ trục giữa camera (Đường tham chiếu xám)
            cv2.line(frame, (cam_cx, 0), (cam_cx, h_orig), (100, 100, 100), 1, cv2.LINE_AA)

            # Hiển thị thông số trên ảnh
            current_fps = 1000.0 / max(dt, 0.1)
            cv2.putText(frame, f"FPS: {current_fps:.1f} | Latency: {dt:.1f}ms [{engine_type}]", 
                        (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Huong lai: {offset_text}", 
                        (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            writer.write(frame)

            if frame_idx % 15 == 0:
                print(f"▶ Frame {frame_idx:03d} | Độ trễ: {dt:4.1f}ms ({current_fps:4.1f} FPS) | Trạng thái: {offset_text}")

            frame_idx += 1

    finally:
        cap.release()
        writer.release()

    avg_latency = np.mean(times[5:]) if len(times) > 5 else np.mean(times)
    avg_fps = 1000.0 / avg_latency

    print("\n" + "=" * 65)
    print("   KẾT QUẢ CHẠY THỰC TẾ TRÊN VIDEO NHÀ MÀNG:")
    print(f"   Tổng khung hình đã xử lý : {frame_idx}")
    print(f"   Độ trễ trung bình        : {avg_latency:.2f} ms / khung hình")
    print(f"   Tốc độ khung hình (FPS)  : {avg_fps:.1f} FPS")
    print(f"   Video kết quả lưu tại    : {out_video_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
