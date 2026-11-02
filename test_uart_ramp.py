#!/usr/bin/env python3
"""
Script kiểm tra truyền thông UART với STM32 theo chuẩn khung truyền (Frame)
của hệ thống RobotNongNghiep (start_combined.sh / combined_app.py).

Giao thức khung truyền 5-Byte Union (chuẩn STM32 DMA Idle Line):
- Byte 0..3: Float 32-bit (IEEE-754 Little-Endian) đại diện cho độ lệch target_path / error_path
- Byte 4   : Byte cờ điều khiển (0: Bình thường / Chạy; 1: Trạm dừng QR >200x200)

Kịch bản test:
- Gửi giá trị tăng dần từ 1 -> 100 (theo từng frame)
- Gửi giá trị giảm dần từ 100 -> 1 (theo từng frame)
- Lặp lại liên tục (hoặc theo số vòng chỉ định) với tần số gửi tương đương tốc độ frame.
"""

import argparse
import os
import struct
import sys
import time
from typing import List, Optional

# Thêm thư mục hiện tại vào sys.path để import UartController nếu cần
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    import serial
except ImportError:
    print("❌ Lỗi: Chưa cài đặt thư viện 'pyserial' trong môi trường Python.")
    print("Vui lòng chạy trong môi trường .venv-yolo hoặc cài đặt: pip install pyserial")
    sys.exit(1)


def find_serial_port(preferred_port: str = "/dev/ttySTM32") -> str:
    """Tự động tìm kiếm cổng serial STM32 có sẵn."""
    candidate_ports = [preferred_port, "/dev/stm32", "/dev/ttyUSB0", "/dev/ttyACM0"]
    for p in candidate_ports:
        if os.path.exists(p):
            return p
    return preferred_port


def format_union_5b_packet(val_float: float, flag_byte: int = 0) -> bytes:
    """Đóng gói 5-byte theo chuẩn IEEE-754 Little-Endian + 1 Byte cờ."""
    offset_bytes = struct.pack('<f', float(val_float))
    flag_b = bytes([int(flag_byte) & 0xFF])
    return offset_bytes + flag_b


def main():
    parser = argparse.ArgumentParser(
        description="Test UART STM32: Gửi giá trị 1-100 và 100-1 theo frame của start_combined.sh"
    )
    parser.add_argument(
        "--port", "-p", type=str, default="/dev/ttySTM32",
        help="Cổng Serial kết nối STM32 (mặc định: /dev/ttySTM32, tự fallback sang /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--baud", "-b", type=int, default=115200,
        help="Baudrate kết nối (mặc định: 115200)"
    )
    parser.add_argument(
        "--fps", "-f", type=float, default=10.0,
        help="Tần số gửi gói tin theo frame / giây (mặc định: 10.0 Hz = 0.1s/frame như start_combined.sh)"
    )
    parser.add_argument(
        "--min", type=float, default=1.0,
        help="Giá trị bắt đầu (mặc định: 1.0)"
    )
    parser.add_argument(
        "--max", type=float, default=100.0,
        help="Giá trị cực đại (mặc định: 100.0)"
    )
    parser.add_argument(
        "--step", type=float, default=1.0,
        help="Bước nhảy giá trị mỗi frame (mặc định: 1.0)"
    )
    parser.add_argument(
        "--flag", type=int, default=0, choices=[0, 1],
        help="Giá trị Byte thứ 5 (0: Chạy bình thường, 1: Cờ dừng trạm QR)"
    )
    parser.add_argument(
        "--loops", "-l", type=int, default=0,
        help="Số chu kỳ lặp lại (0: lặp vô tận cho đến khi bấm Ctrl+C, mặc định: 0)"
    )

    args = parser.parse_args()

    port = find_serial_port(args.port)
    baud = args.baud
    target_fps = max(0.5, args.fps)
    frame_period_sec = 1.0 / target_fps
    flag_val = args.flag

    print("=" * 72)
    print("  RobotNongNghiep - TEST UART GỬI GIÁ TRỊ 1 -> 100 VÀ 100 -> 1")
    print(f"  (Đồng bộ chuẩn Frame 5-Byte của start_combined.sh)")
    print("=" * 72)
    print(f"• Cổng Serial       : {port} (Baudrate: {baud})")
    print(f"• Tần số truyền     : {target_fps:.1f} FPS (Chu kỳ mỗi frame: {frame_period_sec*1000:.1f} ms)")
    print(f"• Dải giá trị       : {args.min:.1f} -> {args.max:.1f} -> {args.min:.1f} (Bước: {args.step:.1f})")
    print(f"• Byte 4 (Cờ lệnh)  : {flag_val} ({'STOP TRẠM QR' if flag_val == 1 else 'CHẠY BÌNH THƯỜNG'})")
    print(f"• Cấu trúc Frame    : [Byte 0..3: Float IEEE-754 LE] + [Byte 4: Flag]")
    print(f"• Chế độ lặp        : {'Vô tận (Nhấn Ctrl+C để dừng)' if args.loops <= 0 else f'{args.loops} chu kỳ'}")
    print("-" * 72)

    # Khởi tạo kết nối Serial
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.05,
            write_timeout=0.05
        )
    except Exception as e:
        print(f"\n❌ LỖI: Không thể mở cổng Serial '{port}'!")
        print(f"  Chi tiết: {e}")
        print("\n💡 Gợi ý khắc phục:")
        print("  1. Kiểm tra cáp USB-UART đã cắm vào máy chưa (`lsusb`).")
        print("  2. Kiểm tra quyền truy cập: `sudo chmod 666 /dev/ttyUSB0`")
        print("  3. Đảm bảo chương trình start_combined.sh hoặc phần mềm khác không đang chiếm cổng.")
        sys.exit(1)

    print(f"✓ Đã kết nối thành công tới cổng {port}!\n")
    print(f"{'FRAME':^8} | {'HƯỚNG':^12} | {'GIÁ TRỊ (Float)':^16} | {'GÓI TIN 5-BYTE (HEX)':^22} | {'TRẠNG THÁI'}")
    print("-" * 72)

    total_frames_sent = 0
    current_loop = 0
    t_start_total = time.perf_counter()

    try:
        while True:
            current_loop += 1
            if args.loops > 0 and current_loop > args.loops:
                break

            # -------------------------------------------------------------
            # PHA 1: TĂNG DẦN TỪ MIN -> MAX (1 -> 100)
            # -------------------------------------------------------------
            cur_val = args.min
            while cur_val <= args.max + 1e-6:
                t_frame_start = time.perf_counter()

                pkt = format_union_5b_packet(cur_val, flag_val)
                ser.write(pkt)
                ser.flush()
                total_frames_sent += 1

                hex_str = " ".join(f"{b:02X}" for b in pkt)
                print(f"#{total_frames_sent:06d} | {'TĂNG ↗':^12} | {cur_val:^16.2f} | [{hex_str}] | STM32 RX OK")

                cur_val += args.step

                # Căn thời gian đúng theo tốc độ Frame (pacing)
                elapsed = time.perf_counter() - t_frame_start
                sleep_time = frame_period_sec - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # -------------------------------------------------------------
            # PHA 2: GIẢM DẦN TỪ MAX -> MIN (100 -> 1)
            # -------------------------------------------------------------
            cur_val = args.max - args.step
            while cur_val >= args.min - 1e-6:
                t_frame_start = time.perf_counter()

                pkt = format_union_5b_packet(cur_val, flag_val)
                ser.write(pkt)
                ser.flush()
                total_frames_sent += 1

                hex_str = " ".join(f"{b:02X}" for b in pkt)
                print(f"#{total_frames_sent:06d} | {'GIẢM ↘':^12} | {cur_val:^16.2f} | [{hex_str}] | STM32 RX OK")

                cur_val -= args.step

                # Căn thời gian đúng theo tốc độ Frame (pacing)
                elapsed = time.perf_counter() - t_frame_start
                sleep_time = frame_period_sec - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n⏹ Nhận tín hiệu dừng từ người dùng (Ctrl+C).")
    finally:
        total_time = time.perf_counter() - t_start_total
        actual_fps = total_frames_sent / total_time if total_time > 0 else 0
        try:
            ser.close()
        except Exception:
            pass

        print("=" * 72)
        print("  KẾT THÚC QUÁ TRÌNH TEST UART")
        print("=" * 72)
        print(f"• Tổng số frame đã gửi : {total_frames_sent} frames")
        print(f"• Số chu kỳ hoàn thành  : {current_loop - 1} chu kỳ (1->100->1)")
        print(f"• Thời gian chạy thực tế: {total_time:.2f} giây")
        print(f"• Tốc độ phát trung bình: {actual_fps:.2f} FPS")
        print("=" * 72)


if __name__ == "__main__":
    main()

