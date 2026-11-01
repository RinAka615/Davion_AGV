"""
Automated Unit Tests for Vision Navigation Fusion (Path Centerline vs QR Code Center).
Verifies the automatic transition:
When QR code bounding area > 60x60, lateral offset is calculated based on QR center instead of path center.
"""

import os
import sys
import numpy as np

# Thêm root vào sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qr_scanner.detector import QRDetectionResult
from src.path_detector import DetectionResult, TrackingState
from src.navigation_fusion import (
    FusedNavigationTarget,
    check_qr_exceeds_threshold,
    fuse_navigation_target
)
from src.uart_controller import UartController


def create_mock_qr(
    cx: float,
    cy: float,
    width: float,
    height: float,
    text: str = "TEST-QR-CHECKPOINT"
) -> QRDetectionResult:
    """Tạo đối tượng QRDetectionResult giả lập với kích thước và tâm xác định."""
    half_w = width / 2.0
    half_h = height / 2.0
    pts = np.array([
        [cx - half_w, cy - half_h],  # TL
        [cx + half_w, cy - half_h],  # TR
        [cx + half_w, cy + half_h],  # BR
        [cx - half_w, cy + half_h]   # BL
    ], dtype=np.float32)

    return QRDetectionResult(
        points=pts,
        text=text,
        is_decoded=True,
        center=(cx, cy),
        center_int=(int(round(cx)), int(round(cy))),
        width_px=width,
        height_px=height,
        area_px=width * height,
        aspect_ratio=width / max(1.0, height),
        orientation_deg=0.0
    )


def create_mock_path_result(
    path_center_x: int = 320,
    ref_y: int = 400,
    offset_px: float = 0.0
) -> DetectionResult:
    """Tạo đối tượng DetectionResult giả lập cho bám đường (Path Tracking)."""
    return DetectionResult(
        state=TrackingState.TRACKING,
        confidence=85.0,
        lateral_offset_px=offset_px,
        offset_direction="Đúng tâm (0 px)" if abs(offset_px) <= 3 else f"Lệch {'phải' if offset_px > 0 else 'trái'} {abs(offset_px):.1f} px",
        ref_y_global=ref_y,
        path_center=(path_center_x, ref_y),
        box_center=(path_center_x, ref_y),
        center_method="path"
    )


def test_qr_size_threshold_check():
    """Kiểm tra điều kiện kích thước và diện tích đường bao > 60x60."""
    # 1. Mã QR nhỏ (40x40 = 1600 px^2 < 3600 px^2) -> False
    small_qr = create_mock_qr(cx=300, cy=300, width=40, height=40)
    is_over, bw, bh, area = check_qr_exceeds_threshold(small_qr, min_dim=60.0)
    assert is_over is False, f"Expected False for 40x40 QR, got {is_over}"
    assert area == 1600.0

    # 2. Mã QR lớn (80x80 = 6400 px^2 > 3600 px^2) -> True
    large_qr = create_mock_qr(cx=300, cy=300, width=80, height=80)
    is_over, bw, bh, area = check_qr_exceeds_threshold(large_qr, min_dim=60.0)
    assert is_over is True, f"Expected True for 80x80 QR, got {is_over}"
    assert area == 6400.0

    # 3. Mã QR cận ngưỡng chuẩn (60x60 = 3600 px^2) -> True
    boundary_qr = create_mock_qr(cx=300, cy=300, width=60, height=60)
    is_over, bw, bh, area = check_qr_exceeds_threshold(boundary_qr, min_dim=60.0)
    assert is_over is True, f"Expected True for 60x60 QR, got {is_over}"

    print("[PASS] test_qr_size_threshold_check: Size & area threshold logic (> 60x60) 100% verified")


def test_fusion_normal_path_when_qr_small():
    """Khi QR còn ở xa (< 60x60), hệ thống phải bám theo tâm của lối đi."""
    frame_shape = (720, 1280)  # w = 1280, cam_cx = 640
    path_res = create_mock_path_result(path_center_x=620, ref_y=500, offset_px=-20.0)
    small_qr = create_mock_qr(cx=750, cy=200, width=45, height=45)

    fused = fuse_navigation_target(
        path_res=path_res,
        qr_results=[small_qr],
        frame_shape=frame_shape,
        min_qr_size=60.0
    )

    assert fused.is_qr_mode is False, "Should NOT switch to QR mode for small QR (< 60x60)"
    assert fused.target_pt == (620, 500), f"Expected path target (620, 500), got {fused.target_pt}"
    assert fused.lateral_offset_px == -20.0, f"Expected offset -20.0, got {fused.lateral_offset_px}"
    print("[PASS] test_fusion_normal_path_when_qr_small: Correctly maintained path tracking")


def test_fusion_switches_to_qr_center_when_qr_large():
    """Khi QR đến gần (> 60x60), hệ thống phải chuyển sang tính sai số theo tâm QR."""
    frame_shape = (720, 1280)  # w = 1280, cam_cx = 640
    # Lối đi đang có tâm tại x = 600 (lệch -40px)
    path_res = create_mock_path_result(path_center_x=600, ref_y=500, offset_px=-40.0)

    # Mã QR trạm dừng có tâm tại x = 700, y = 350, kích thước 75x75 (> 60x60)
    # Độ lệch so với tâm camera (640) phải là: 700 - 640 = +60.0 px (Lệch phải)
    large_qr = create_mock_qr(cx=700.0, cy=350.0, width=75.0, height=75.0)

    fused = fuse_navigation_target(
        path_res=path_res,
        qr_results=[large_qr],
        frame_shape=frame_shape,
        min_qr_size=60.0
    )

    assert fused.is_qr_mode is True, "Must switch to QR mode when QR size > 60x60"
    assert fused.target_pt == (700, 350), f"Expected target (700, 350), got {fused.target_pt}"
    assert abs(fused.lateral_offset_px - 60.0) < 1e-3, f"Expected offset +60.0, got {fused.lateral_offset_px}"
    assert "Lệch phải 60.0 px (Theo tâm QR)" in fused.offset_direction
    print(f"[PASS] test_fusion_switches_to_qr_center_when_qr_large: Successfully switched to QR center (> 60x60) (Offset={fused.lateral_offset_px:+.1f}px)")


def test_fusion_handles_no_path_with_large_qr():
    """Ngay cả khi lối đi bị mất (lost), nếu thấy QR lớn (> 60x60) thì vẫn điều hướng chuẩn theo tâm QR."""
    frame_shape = (720, 1280)
    path_res = None  # Không có lối đi
    large_qr = create_mock_qr(cx=640.0, cy=400.0, width=80.0, height=80.0)

    fused = fuse_navigation_target(
        path_res=path_res,
        qr_results=[large_qr],
        frame_shape=frame_shape,
        min_qr_size=60.0
    )

    assert fused.is_qr_mode is True
    assert fused.target_pt == (640, 400)
    assert abs(fused.lateral_offset_px - 0.0) < 1e-3
    assert "Đúng tâm mã QR (0 px)" in fused.offset_direction
    print("[PASS] test_fusion_handles_no_path_with_large_qr: Successfully navigated on QR without path")


def test_fusion_respects_override_toggle():
    """Kiểm tra khi người dùng tắt checkbox ghi đè QR (enable_qr_override=False)."""
    frame_shape = (720, 1280)
    path_res = create_mock_path_result(path_center_x=600, ref_y=500, offset_px=-40.0)
    large_qr = create_mock_qr(cx=700.0, cy=350.0, width=80.0, height=80.0)

    fused = fuse_navigation_target(
        path_res=path_res,
        qr_results=[large_qr],
        frame_shape=frame_shape,
        min_qr_size=60.0,
        enable_qr_override=False  # Người dùng tắt tính năng
    )

    assert fused.is_qr_mode is False, "Should remain in path mode when override is disabled"
    assert fused.lateral_offset_px == -40.0
    print("[PASS] test_fusion_respects_override_toggle: Successfully verified user toggle")


def test_uart_flag_byte_qr_over_200x200():
    """
    Quy tắc nghiệp vụ: Khi diện tích đường bao QR > 200x200 (S >= 40,000 px^2):
    - uart_flag_byte = 1 (Byte 4 gửi STM32 = 0x01).
    - qr_stop_triggered = True.
    - Sai số lái vẫn được gửi bình thường qua Bytes 0..3 (theo tâm QR).
    """
    frame_shape = (720, 1280)
    # QR lớn tại x = 670, kích thước 210x210 (> 200x200)
    huge_qr = create_mock_qr(cx=670.0, cy=360.0, width=210.0, height=210.0)
    path_res = create_mock_path_result(path_center_x=620, ref_y=500, offset_px=-20.0)

    fused = fuse_navigation_target(
        path_res=path_res,
        qr_results=[huge_qr],
        frame_shape=frame_shape,
        min_qr_size=60.0,
        qr_stop_thresh=200.0
    )

    assert fused.qr_stop_triggered is True, "qr_stop_triggered must be True for QR > 200x200"
    assert fused.uart_flag_byte == 1, "uart_flag_byte must be 1 for QR > 200x200"
    assert fused.is_qr_mode is True, "Must be in QR mode"
    assert abs(fused.lateral_offset_px - 30.0) < 1e-3, f"Expected offset +30.0, got {fused.lateral_offset_px}"

    # Kiểm tra gói tin UART 5-byte
    uart = UartController()
    pkt, disp = uart.format_union_5b_packet(
        offset_px=fused.lateral_offset_px,
        is_detected=bool(fused.uart_flag_byte)
    )
    assert len(pkt) == 5, f"Expected 5 bytes packet, got {len(pkt)}"
    assert pkt[4] == 1, f"Byte 4 must be 1 (0x01), got {pkt[4]}"
    print(f"[PASS] test_uart_flag_byte_qr_over_200x200: Successfully triggered STOP station (Byte 4 = {pkt[4]}, Packet: {disp})")


def test_uart_flag_byte_qr_between_60_and_200():
    """
    Khi QR ở giữa 60x60 và 200x200 (ví dụ 100x100):
    - Đã chuyển sang lái theo tâm QR (> 60x60).
    - Nhưng CHƯA đạt ngưỡng trạm dừng (<= 200x200) -> uart_flag_byte = 0 (Byte 4 = 0x00).
    - Sai số lái vẫn được gửi bình thường qua Bytes 0..3 (theo tâm QR).
    """
    frame_shape = (720, 1280)
    mid_qr = create_mock_qr(cx=680.0, cy=360.0, width=100.0, height=100.0)
    path_res = create_mock_path_result(path_center_x=620, ref_y=500, offset_px=-20.0)

    fused = fuse_navigation_target(
        path_res=path_res,
        qr_results=[mid_qr],
        frame_shape=frame_shape,
        min_qr_size=60.0,
        qr_stop_thresh=200.0
    )

    assert fused.is_qr_mode is True, "Must be in QR mode since 100x100 > 60x60"
    assert fused.qr_stop_triggered is False, "qr_stop_triggered must be False since 100x100 <= 200x200"
    assert fused.uart_flag_byte == 0, "uart_flag_byte must be 0 since QR <= 200x200"
    assert abs(fused.lateral_offset_px - 40.0) < 1e-3

    # Kiểm tra gói tin UART 5-byte
    uart = UartController()
    pkt, disp = uart.format_union_5b_packet(
        offset_px=fused.lateral_offset_px,
        is_detected=bool(fused.uart_flag_byte)
    )
    assert pkt[4] == 0, f"Byte 4 must be 0 (0x00), got {pkt[4]}"
    # Bytes 0..3 vẫn phải giữ sai số lái +40.0 px (không bị gán về 0)
    import struct
    unpacked_offset = struct.unpack('<f', pkt[0:4])[0]
    assert abs(unpacked_offset - 40.0) < 1e-3, f"Expected offset in Bytes 0..3 = 40.0, got {unpacked_offset}"
    print(f"[PASS] test_uart_flag_byte_qr_between_60_and_200: Byte 4 = 0, Steering preserved ({unpacked_offset:+.1f}px) (Packet: {disp})")


def test_uart_flag_byte_ignores_path_condition():
    """
    Quy tắc then chốt: KHÔNG xem điều kiện có đường hay mất đường để gửi Byte 4.
    Dù có đường (TRACKING) nhưng QR <= 200x200 (hoặc không có QR) thì Byte 4 BẮT BUỘC là 0.
    """
    frame_shape = (720, 1280)
    # Lối đi rõ ràng, Tracking chuẩn
    path_res = create_mock_path_result(path_center_x=600, ref_y=500, offset_px=-40.0)

    # 1. Trường hợp có đường nhưng hoàn toàn không có QR
    fused_no_qr = fuse_navigation_target(
        path_res=path_res,
        qr_results=[],
        frame_shape=frame_shape,
        min_qr_size=60.0,
        qr_stop_thresh=200.0
    )
    assert fused_no_qr.uart_flag_byte == 0, "Byte 4 must be 0 even if path is tracking!"
    assert fused_no_qr.qr_stop_triggered is False

    uart = UartController()
    pkt, disp = uart.format_union_5b_packet(
        offset_px=fused_no_qr.lateral_offset_px,
        is_detected=bool(fused_no_qr.uart_flag_byte)
    )
    assert pkt[4] == 0, f"Byte 4 must be 0, got {pkt[4]}"
    import struct
    unpacked_offset = struct.unpack('<f', pkt[0:4])[0]
    assert abs(unpacked_offset - (-40.0)) < 1e-3, f"Path steering offset -40.0px preserved"

    # 2. Trường hợp mất đường và không có QR
    fused_lost = fuse_navigation_target(
        path_res=None,
        qr_results=[],
        frame_shape=frame_shape,
        min_qr_size=60.0,
        qr_stop_thresh=200.0
    )
    assert fused_lost.uart_flag_byte == 0
    assert fused_lost.qr_stop_triggered is False
    pkt_lost, _ = uart.format_union_5b_packet(
        offset_px=fused_lost.lateral_offset_px,
        is_detected=bool(fused_lost.uart_flag_byte)
    )
    assert pkt_lost[4] == 0

    print("[PASS] test_uart_flag_byte_ignores_path_condition: Verified Byte 4 is strictly 0 when QR <= 200x200 regardless of path state")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING AUTOMATED TEST SUITE FOR NAVIGATION FUSION")
    print("=" * 60)
    test_qr_size_threshold_check()
    test_fusion_normal_path_when_qr_small()
    test_fusion_switches_to_qr_center_when_qr_large()
    test_fusion_handles_no_path_with_large_qr()
    test_fusion_respects_override_toggle()
    test_uart_flag_byte_qr_over_200x200()
    test_uart_flag_byte_qr_between_60_and_200()
    test_uart_flag_byte_ignores_path_condition()
    print("=" * 60)
    print("ALL 8 NAVIGATION FUSION TESTS PASSED 100%!")
    print("=" * 60)

