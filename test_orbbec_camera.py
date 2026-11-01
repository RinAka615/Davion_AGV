#!/usr/bin/env python3
"""
Kiểm tra và hiển thị trực tiếp camera Orbbec Astra Pro HD.
Sử dụng OpenCV với định dạng MJPG để đạt độ phân giải HD 1280x720 @ 30 FPS.
"""

import argparse
import os
import sys
import time
import cv2

def main():
    parser = argparse.ArgumentParser(description="Kiểm tra Camera Orbbec Astra Pro HD")
    parser.add_argument("device", nargs="?", default="0", help="Camera device index hoặc path (mặc định: 0 hoặc /dev/video0)")
    parser.add_argument("--no-gui", action="store_true", help="Chạy không mở cửa sổ GUI")
    args = parser.parse_args()

    device_id = int(args.device) if args.device.isdigit() else args.device

    print(f"\n========================================================")
    print(f"  Kiểm tra Camera Orbbec Astra Pro HD (Device: {device_id})")
    print(f"========================================================")

    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"❌ Không thể mở thiết bị camera: {device_id}")
        print("Gợi ý: Kiểm tra `ls -l /dev/video*` hoặc quyền truy cập người dùng.")
        sys.exit(1)

    # Cấu hình định dạng tối ưu cho Orbbec Astra Pro: MJPG 1280x720 @ 30 FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"✓ Kết nối thành công!")
    print(f"  Độ phân giải: {w}x{h}")
    print(f"  Tốc độ khung hình: {fps:.1f} FPS")

    # Đọc frame đầu tiên
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Không đọc được dữ liệu frame từ camera.")
        cap.release()
        sys.exit(1)

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    snapshot_path = os.path.join(out_dir, "orbbec_live_snapshot.jpg")
    cv2.imwrite(snapshot_path, frame)
    print(f"✓ Đã lưu ảnh chụp thử mẫu tại: {snapshot_path}")

    # Nếu đang chạy trong môi trường có màn hình GUI (DISPLAY)
    has_display = bool(os.environ.get("DISPLAY"))
    if not has_display or "--no-gui" in sys.argv:
        print("▶ Chạy ở chế độ không mở cửa sổ (headless). Đọc 30 frame để kiểm tra FPS...")
        frame_count = 0
        t0 = time.time()
        for _ in range(30):
            ret, _ = cap.read()
            if ret:
                frame_count += 1
        elapsed = time.time() - t0
        print(f"✓ Đã đọc {frame_count} frame liên tục trong {elapsed:.2f}s (~{frame_count/elapsed:.1f} FPS)")
        cap.release()
        print("✓ Hoàn thành kiểm tra camera.")
        return

    print("\n▶ Đang hiển thị trực tiếp luồng camera...")
    print("  Nhấn phím 'q' hoặc 'ESC' trên cửa sổ hình ảnh (hoặc Ctrl+C) để thoát.\n")

    frame_count = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Mất tín hiệu camera...")
                break

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - t0
                real_fps = frame_count / elapsed
                print(f"[Live] Đã nhận {frame_count} frames | FPS thực tế: {real_fps:.1f}")

            # Vẽ thông số lên ảnh
            cv2.putText(frame, f"Orbbec Astra Pro HD | {w}x{h} | Frame #{frame_count}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.imshow("Orbbec Astra Pro - Live Stream (Nhan 'q' de thoat)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
    except KeyboardInterrupt:
        print("\nNgười dùng dừng chương trình.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Đã đóng camera an toàn.")


if __name__ == "__main__":
    main()
