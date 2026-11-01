#!/usr/bin/env python3
"""
Ứng dụng Phát hiện, Nắn thẳng (Perspective Dewarp) và Đọc mã QR Thời gian thực.
Triển khai trên Ubuntu/Linux bằng Python + OpenCV + ZXing-C++.
Hỗ trợ Webcam, Video File, RTSP/HTTP Stream và chế độ Headless.
"""

import os
import sys
import time
import signal
import argparse
from typing import Optional, Dict, Any, List, Tuple
import cv2
import numpy as np

# Thêm thư mục hiện tại vào sys.path để import qr_scanner
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from qr_scanner import QRDetector, QRJsonLogger, QRVisualizer, QRDetectionResult


class QRScannerApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.running = True
        self.paused = False

        # Khởi tạo detector
        self.detector = QRDetector(
            backend=args.backend,
            enable_dewarp=not args.no_dewarp
        )

        # Khởi tạo JSONL logger
        self.logger = QRJsonLogger(
            log_path=args.log_file,
            dedup_cooldown_sec=args.dedup_cooldown
        )

        # Khởi tạo visualizer
        self.visualizer = QRVisualizer(
            max_display_w=args.max_display_width,
            max_display_h=args.max_display_height
        )

        # Cấu hình snapshot
        self.save_snapshots = args.save_snapshots
        self.snapshot_dir = args.snapshot_dir
        if self.save_snapshots:
            os.makedirs(self.snapshot_dir, exist_ok=True)

        # Biến thống kê
        self.total_frames = 0
        self.total_decoded_count = 0
        self.total_candidate_count = 0
        self.fps_smoothed = 0.0
        self.start_time = 0.0

        # Đăng ký signal handler để thoát sạch khi nhận Ctrl+C
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        print("\n\n[INFO] Nhận tín hiệu ngắt (Ctrl+C). Đang dừng chương trình và giải phóng tài nguyên...")
        self.running = False

    def _open_source(self) -> Tuple[cv2.VideoCapture, bool, str, float]:
        """
        Mở nguồn đầu vào: Webcam index, Video file path, hoặc Stream URL.
        :return: (cap, is_live_stream, source_display_name, original_fps)
        """
        source_str = self.args.source.strip()
        is_live_stream = False
        original_fps = 30.0

        # Kiểm tra xem source có phải là số nguyên (camera index)
        if source_str.isdigit():
            cam_idx = int(source_str)
            cap = cv2.VideoCapture(cam_idx)
            is_live_stream = True
            source_name = f"Webcam #{cam_idx}"
            if not cap.isOpened():
                print(f"\n[LỖI NGHIÊM TRỌNG] Không thể mở camera thiết bị: /dev/video{cam_idx}")
                print("  Gợi ý khắc phục:")
                print("  1. Kiểm tra xem camera đã được cắm và nhận diện chưa: 'ls -l /dev/video*'")
                print("  2. Kiểm tra quyền truy cập user: 'groups | grep video'")
                print("  3. Thử chạy với file video: ./run.sh --source \"/duong/dan/video.mp4\"\n")
                sys.exit(2)
            # Tối ưu buffer cho webcam tránh tích lũy frame cũ gây lag
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        elif source_str.startswith(("rtsp://", "http://", "https://")):
            cap = cv2.VideoCapture(source_str)
            is_live_stream = True
            source_name = f"Stream ({source_str.split('?')[0]})"
            if not cap.isOpened():
                print(f"\n[LỖI] Không thể kết nối tới luồng stream URL: {source_str}")
                sys.exit(2)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        else:
            # File video cục bộ
            video_path = os.path.abspath(source_str)
            if not os.path.exists(video_path):
                print(f"\n[LỖI] File video không tồn tại: {video_path}")
                sys.exit(2)

            cap = cv2.VideoCapture(video_path)
            is_live_stream = False
            source_name = os.path.basename(video_path)
            if not cap.isOpened():
                print(f"\n[LỖI] OpenCV không thể mở định dạng file video: {video_path}")
                sys.exit(2)

            vid_fps = cap.get(cv2.CAP_PROP_FPS)
            if vid_fps and vid_fps > 1.0:
                original_fps = vid_fps

        return cap, is_live_stream, source_name, original_fps

    def run(self):
        cap, is_live, source_name, orig_fps = self._open_source()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live else -1
        duration_sec = (total_vid_frames / orig_fps) if total_vid_frames > 0 else 0.0

        print("=" * 68)
        print(f"Khởi động QR Scanner Pipeline:")
        print(f"  ▶ Nguồn đầu vào  : {source_name} ({width}x{height})")
        print(f"  ▶ FPS gốc        : {orig_fps:.2f} | Tổng số frame: {total_vid_frames if total_vid_frames > 0 else 'N/A'}")
        print(f"  ▶ Backend        : {self.args.backend} | Perspective Dewarp: {'BẬT' if not self.args.no_dewarp else 'TẮT'}")
        print(f"  ▶ Ghi log JSONL  : {self.args.log_file} (Cooldown chống lặp: {self.args.dedup_cooldown}s)")
        print(f"  ▶ Chế độ hiển thị: {'HEADLESS (Không mở cửa sổ)' if self.args.headless else 'Cửa sổ OpenCV HighGUI'}")
        if self.args.benchmark:
            print(f"  ▶ Chế độ BENCHMARK: BẬT (Quét toàn bộ với tốc độ tối đa không delay)")
        print("=" * 68)

        window_name = "QR Code Real-time Scanner (OpenCV + ZXing-C++)"
        if not self.args.headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Tính toán delay giữa các frame cho video file
        # Nếu benchmark hoặc no-delay: delay = 1ms
        target_frame_time = 1.0 / orig_fps if (not is_live and not self.args.benchmark and not self.args.no_delay) else 0.0

        self.start_time = time.time()
        fps_timer = time.time()
        fps_frame_count = 0

        current_frame_idx = 0

        try:
            while self.running:
                if not self.paused:
                    loop_start = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        if not is_live:
                            print(f"\n[INFO] Đã đọc hết toàn bộ file video ({current_frame_idx} frames).")
                            if self.args.loop:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                current_frame_idx = 0
                                continue
                            else:
                                break
                        else:
                            print("\n[CẢNH BÁO] Không nhận được frame từ camera/stream. Đang thử lại...")
                            time.sleep(0.1)
                            continue

                    current_frame_idx += 1
                    self.total_frames += 1
                    fps_frame_count += 1

                    # Tính toán video timestamp (giây)
                    if not is_live:
                        video_timestamp = current_frame_idx / orig_fps
                    else:
                        video_timestamp = None

                    # --- BƯỚC PHÁT HIỆN VÀ GIẢI MÃ QR ---
                    t_detect_start = time.time()
                    results = self.detector.detect_and_decode(frame)
                    t_detect_elapsed = time.time() - t_detect_start

                    # Xử lý kết quả
                    decoded_in_frame = 0
                    candidate_in_frame = 0

                    for res in results:
                        if res.is_decoded:
                            decoded_in_frame += 1
                            self.total_decoded_count += 1
                            # Ghi log ra JSONL (có chống ghi trùng lặp)
                            logged = self.logger.log(
                                text=res.text,
                                corners=res.points,
                                backend=res.backend,
                                video_timestamp_sec=video_timestamp,
                                barcode_format=res.format_name,
                                geometry=res.geometry
                            )
                            if logged:
                                t_info = f"[{video_timestamp:.2f}s]" if video_timestamp is not None else f"[{time.strftime('%H:%M:%S')}]"
                                cx, cy = res.center
                                print(f"  ✔ {t_info} [ĐÃ GIẢI MÃ] ({res.backend}): \"{res.text}\" | Tâm: ({cx:.1f}, {cy:.1f}) | {res.width_px:.0f}x{res.height_px:.0f}px")

                                # Lưu snapshot nếu được bật
                                if self.save_snapshots:
                                    self._save_detection_snapshot(frame, res, current_frame_idx)
                        else:
                            candidate_in_frame += 1
                            self.total_candidate_count += 1

                    # Cập nhật FPS mượt
                    now = time.time()
                    if (now - fps_timer) >= 0.5:
                        self.fps_smoothed = fps_frame_count / (now - fps_timer)
                        fps_frame_count = 0
                        fps_timer = now

                    # Chuẩn bị thông tin Telemetry HUD
                    if not is_live:
                        time_str = f"FRAME: {current_frame_idx}/{total_vid_frames} ({video_timestamp:.1f}s/{duration_sec:.1f}s)"
                    else:
                        time_str = f"FRAME: {current_frame_idx} ({time.strftime('%H:%M:%S')})"

                    hud_info = {
                        "fps": self.fps_smoothed,
                        "source": source_name,
                        "time_str": time_str,
                        "decoded_count": self.total_decoded_count,
                        "candidate_count": self.total_candidate_count,
                        "backend": self.detector.backend
                    }

                    # Hiển thị hoặc Headless
                    if not self.args.headless:
                        vis_frame = self.visualizer.draw_results(frame, results, hud_info)
                        display_frame, _ = self.visualizer.fit_to_screen(vis_frame)
                        cv2.imshow(window_name, display_frame)

                        # Tính waitKey delay để giữ đúng tốc độ phát của video
                        if target_frame_time > 0:
                            elapsed = time.time() - loop_start
                            wait_ms = max(1, int((target_frame_time - elapsed) * 1000))
                        else:
                            wait_ms = 1

                        key = cv2.waitKey(wait_ms) & 0xFF
                        if key in (ord('q'), ord('Q'), 27):  # Q hoặc ESC
                            print("\n[INFO] Người dùng nhấn phím Q/ESC để thoát.")
                            break
                        elif key == ord(' '):  # SPACE tạm dừng
                            self.paused = True
                            print("\n[INFO] Đã tạm dừng video. Nhấn SPACE để tiếp tục.")
                        elif key in (ord('s'), ord('S')):  # S lưu ảnh
                            snap_path = os.path.join(self.snapshot_dir, f"manual_snap_f{current_frame_idx}.jpg")
                            cv2.imwrite(snap_path, vis_frame)
                            print(f"[SNAPSHOT] Đã lưu ảnh: {snap_path}")

                    else:
                        # Chế độ Headless
                        if current_frame_idx % 250 == 0 or current_frame_idx == total_vid_frames:
                            print(f"  ... Đã xử lý {current_frame_idx}/{total_vid_frames} frames ({self.fps_smoothed:.1f} FPS) | Giải mã: {self.total_decoded_count}")

                else:
                    # Đang ở trạng thái Tạm dừng (PAUSED)
                    key = cv2.waitKey(30) & 0xFF
                    if key in (ord('q'), ord('Q'), 27):
                        break
                    elif key == ord(' '):
                        self.paused = False
                        print("[INFO] Tiếp tục phát video...")

        finally:
            self._cleanup(cap)

    def _save_detection_snapshot(self, frame: np.ndarray, res: QRDetectionResult, frame_idx: int):
        """Lưu ảnh vùng mã QR đã nắn thẳng và ảnh frame có nhãn đánh dấu."""
        timestamp_slug = int(time.time() * 1000)
        # Lưu patch nắn thẳng
        if res.dewarped_patch is not None:
            patch_name = f"qr_dewarped_f{frame_idx}_{timestamp_slug}.png"
            patch_path = os.path.join(self.snapshot_dir, patch_name)
            cv2.imwrite(patch_path, res.dewarped_patch)

    def _cleanup(self, cap: cv2.VideoCapture):
        """Dọn dẹp và in báo cáo kết thúc."""
        elapsed = time.time() - self.start_time
        avg_fps = self.total_frames / elapsed if elapsed > 0 else 0.0

        if cap is not None and cap.isOpened():
            cap.release()

        cv2.destroyAllWindows()
        self.logger.close()

        print("\n" + "=" * 68)
        print("TỔNG KẾT QUÁ TRÌNH QUÉT MÃ QR:")
        print(f"  ✔ Tổng số frames đã xử lý : {self.total_frames}")
        print(f"  ✔ Thời gian chạy          : {elapsed:.2f} giây")
        print(f"  ✔ Tốc độ xử lý trung bình : {avg_fps:.1f} FPS")
        print(f"  ✔ Tổng số sự kiện giải mã : {self.total_decoded_count}")
        print(f"  ✔ Số mã QR duy nhất đọc được: {len(self.logger.unique_codes_seen)}")
        if self.logger.unique_codes_seen:
            print("  ✔ Danh sách mã đọc được:")
            for code in self.logger.unique_codes_seen:
                print(f"      - {code}")
        print(f"  ✔ File nhật ký JSONL      : {self.args.log_file}")
        print("=" * 68 + "\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ứng dụng phát hiện, nắn thẳng phối cảnh và đọc mã QR thời gian thực (Ubuntu/Linux)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-s", "--source",
        type=str,
        default="0",
        help="Nguồn đầu vào: Camera index (e.g. 0), đường dẫn file video (.mp4/.avi/...), hoặc URL stream (rtsp://, http://)"
    )

    parser.add_argument(
        "-b", "--backend",
        type=str,
        choices=["auto", "zxing", "opencv"],
        default="auto",
        help="Backend nhận diện mã: 'auto' (ưu tiên ZXing-C++, dự phòng OpenCV), 'zxing', hoặc 'opencv'"
    )

    parser.add_argument(
        "--dedup-cooldown",
        type=float,
        default=2.0,
        help="Thời gian chống ghi log lặp (giây) khi cùng một mã QR xuất hiện liên tục trong khung hình"
    )

    parser.add_argument(
        "-l", "--log-file",
        type=str,
        default="output_qr/qr_events.jsonl",
        help="Đường dẫn file JSONL lưu sự kiện quét mã thành công"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Chạy ở chế độ không giao diện (dành cho server, ssh, hoặc đo benchmark)"
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Quét toàn bộ file video với tốc độ tối đa không chờ delay khung hình để đo hiệu năng FPS"
    )

    parser.add_argument(
        "--no-delay",
        action="store_true",
        help="Bỏ qua delay đồng bộ FPS của video gốc, chạy nhanh nhất có thể"
    )

    parser.add_argument(
        "--loop",
        action="store_true",
        help="Tự động lặp lại video khi phát hết"
    )

    parser.add_argument(
        "--no-dewarp",
        action="store_true",
        help="Tắt tính năng nắn phối cảnh (Perspective Dewarp) trên các mã nghiêng"
    )

    parser.add_argument(
        "--save-snapshots",
        action="store_true",
        help="Lưu ảnh crop nắn thẳng các mã QR giải mã thành công"
    )

    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default="output_qr/snapshots",
        help="Thư mục lưu ảnh snapshot"
    )

    parser.add_argument(
        "--max-display-width",
        type=int,
        default=1280,
        help="Chiều rộng tối đa của cửa sổ hiển thị (tránh tràn màn hình)"
    )

    parser.add_argument(
        "--max-display-height",
        type=int,
        default=800,
        help="Chiều cao tối đa của cửa sổ hiển thị (đặc biệt hữu ích với video dọc 720x1280)"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    app = QRScannerApp(args)
    app.run()


if __name__ == "__main__":
    main()
