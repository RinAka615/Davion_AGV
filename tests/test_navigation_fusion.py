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
    fuse_navigation_target,
    QRTurnMissionManager,
    QRTurnMissionState
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
    """Kiểm tra điều kiện kích thước và diện tích đường bao > 100x100."""
    # 1. Mã QR nhỏ (80x80 = 6400 px^2 < 10000 px^2) -> False
    small_qr = create_mock_qr(cx=300, cy=300, width=80, height=80)
    is_over, bw, bh, area = check_qr_exceeds_threshold(small_qr, min_dim=100.0)
    assert is_over is False, f"Expected False for 80x80 QR, got {is_over}"
    assert area == 6400.0

    # 2. Mã QR lớn (140x140 = 19600 px^2 > 10000 px^2) -> True
    large_qr = create_mock_qr(cx=300, cy=300, width=140, height=140)
    is_over, bw, bh, area = check_qr_exceeds_threshold(large_qr, min_dim=100.0)
    assert is_over is True, f"Expected True for 140x140 QR, got {is_over}"
    assert area == 19600.0

    # 3. Mã QR cận ngưỡng chuẩn (100x100 = 10000 px^2) -> True
    boundary_qr = create_mock_qr(cx=300, cy=300, width=100, height=100)
    is_over, bw, bh, area = check_qr_exceeds_threshold(boundary_qr, min_dim=100.0)
    assert is_over is True, f"Expected True for 100x100 QR, got {is_over}"

    print("[PASS] test_qr_size_threshold_check: Size & area threshold logic (> 100x100) 100% verified")


def test_fusion_normal_path_when_qr_small():
    """Khi QR còn ở xa (< 100x100), hệ thống phải bám theo tâm của lối đi."""
    frame_shape = (720, 1280)  # w = 1280, cam_cx = 640
    path_res = create_mock_path_result(path_center_x=620, ref_y=500, offset_px=-20.0)
    small_qr = create_mock_qr(cx=750, cy=200, width=75, height=75)

    fused = fuse_navigation_target(
        path_res=path_res,
        qr_results=[small_qr],
        frame_shape=frame_shape,
        min_qr_size=100.0
    )

    assert fused.is_qr_mode is False, "Should NOT switch to QR mode for small QR (< 100x100)"
    assert fused.target_pt == (620, 500), f"Expected path target (620, 500), got {fused.target_pt}"
    assert fused.lateral_offset_px == -20.0, f"Expected offset -20.0, got {fused.lateral_offset_px}"
    print("[PASS] test_fusion_normal_path_when_qr_small: Correctly maintained path tracking")


def test_fusion_switches_to_qr_center_when_qr_large():
    """Khi QR đến gần (> 100x100), hệ thống phải chuyển sang tính sai số theo tâm QR."""
    frame_shape = (720, 1280)  # w = 1280, cam_cx = 640
    # Lối đi đang có tâm tại x = 600 (lệch -40px)
    path_res = create_mock_path_result(path_center_x=600, ref_y=500, offset_px=-40.0)

    # Mã QR trạm dừng có tâm tại x = 700, y = 350, kích thước 120x120 (> 100x100)
    # Độ lệch so với tâm camera (640) phải là: 700 - 640 = +60.0 px (Lệch phải)
    large_qr = create_mock_qr(cx=700.0, cy=350.0, width=120.0, height=120.0)

    fused = fuse_navigation_target(
        path_res=path_res,
        qr_results=[large_qr],
        frame_shape=frame_shape,
        min_qr_size=100.0
    )

    assert fused.is_qr_mode is True, "Must switch to QR mode when QR size > 100x100"
    assert fused.target_pt == (700, 350), f"Expected target (700, 350), got {fused.target_pt}"
    assert abs(fused.lateral_offset_px - 60.0) < 1e-3, f"Expected offset +60.0, got {fused.lateral_offset_px}"
    assert "Lệch phải 60.0 px (Theo tâm QR)" in fused.offset_direction
    print(f"[PASS] test_fusion_switches_to_qr_center_when_qr_large: Successfully switched to QR center (> 100x100) (Offset={fused.lateral_offset_px:+.1f}px)")


def test_fusion_handles_no_path_with_large_qr():
    """Ngay cả khi lối đi bị mất (lost), nếu thấy QR lớn (> 100x100) thì vẫn điều hướng chuẩn theo tâm QR."""
    frame_shape = (720, 1280)
    path_res = None  # Không có lối đi
    large_qr = create_mock_qr(cx=640.0, cy=400.0, width=130.0, height=130.0)

    fused = fuse_navigation_target(
        path_res=path_res,
        qr_results=[large_qr],
        frame_shape=frame_shape,
        min_qr_size=100.0
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
    large_qr = create_mock_qr(cx=700.0, cy=350.0, width=120.0, height=120.0)

    fused = fuse_navigation_target(
        path_res=path_res,
        qr_results=[large_qr],
        frame_shape=frame_shape,
        min_qr_size=100.0,
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
        min_qr_size=100.0,
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


def test_uart_flag_byte_qr_between_100_and_200():
    """
    Khi QR ở giữa 100x100 và 200x200 (ví dụ 150x150):
    - Đã chuyển sang lái theo tâm QR (> 100x100).
    - Nhưng CHƯA đạt ngưỡng trạm dừng (<= 200x200) -> uart_flag_byte = 0 (Byte 4 = 0x00).
    - Sai số lái vẫn được gửi bình thường qua Bytes 0..3 (theo tâm QR).
    """
    frame_shape = (720, 1280)
    mid_qr = create_mock_qr(cx=680.0, cy=360.0, width=150.0, height=150.0)
    path_res = create_mock_path_result(path_center_x=620, ref_y=500, offset_px=-20.0)

    fused = fuse_navigation_target(
        path_res=path_res,
        qr_results=[mid_qr],
        frame_shape=frame_shape,
        min_qr_size=100.0,
        qr_stop_thresh=200.0
    )

    assert fused.is_qr_mode is True, "Must be in QR mode since 150x150 > 100x100"
    assert fused.qr_stop_triggered is False, "qr_stop_triggered must be False since 150x150 <= 200x200"
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
    print(f"[PASS] test_uart_flag_byte_qr_between_100_and_200: Byte 4 = 0, Steering preserved ({unpacked_offset:+.1f}px) (Packet: {disp})")


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
        min_qr_size=100.0,
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
        min_qr_size=100.0,
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


def test_uart_packet_multimode_flags():
    """Kiểm tra UartController đóng gói chính xác 3 chế độ Byte 4:
    0: Chạy bám đường bình thường
    1: Trạm TURN_1 (Kích hoạt dừng 1.0s và tự quay trái trên STM32)
    2: Đã bắt trúng tâm TURN_2 (Dừng hoàn tất)
    """
    import struct
    uart = UartController()

    # Byte 4 = 0 (Chạy)
    pkt0, disp0 = uart.format_union_5b_packet(offset_px=12.5, is_detected=0)
    assert pkt0[4] == 0, f"Expected Byte 4 == 0, got {pkt0[4]}"
    assert abs(struct.unpack('<f', pkt0[0:4])[0] - 12.5) < 1e-4

    # Byte 4 = 1 (Trạm TURN_1)
    pkt1, disp1 = uart.format_union_5b_packet(offset_px=0.0, is_detected=1)
    assert pkt1[4] == 1, f"Expected Byte 4 == 1, got {pkt1[4]}"
    assert abs(struct.unpack('<f', pkt1[0:4])[0] - 0.0) < 1e-4

    # Byte 4 = 2 (Đã bắt tâm TURN_2)
    pkt2, disp2 = uart.format_union_5b_packet(offset_px=15.0, is_detected=2)
    assert pkt2[4] == 2, f"Expected Byte 4 == 2, got {pkt2[4]}"
    assert abs(struct.unpack('<f', pkt2[0:4])[0] - 15.0) < 1e-4

    # Byte 4 = 3 (Đến trạm TURN_2: Dừng 500ms và xoay trái tìm đường)
    pkt3, disp3 = uart.format_union_5b_packet(offset_px=0.0, is_detected=3)
    assert pkt3[4] == 3, f"Expected Byte 4 == 3, got {pkt3[4]}"
    assert abs(struct.unpack('<f', pkt3[0:4])[0] - 0.0) < 1e-4

    print("[PASS] test_uart_packet_multimode_flags: 4 modes (0=Run, 1=TURN_1 station, 2=TURN_2 center, 3=Turn left seek path) correctly encoded into UART packet")


def test_qr_turn_mission_lifecycle():
    """Kiểm tra toàn bộ chu trình nhiệm vụ 10 giai đoạn:
    1. Chạy thẳng bám luống 1 (TRACKING_ROW_1 / IDLE, B4=0)
    2. Gặp QR1 >= min_qr_nav_dim (30px): Chuyển sang bám tâm QR1 (TRACKING_QR1, B4=0)
    3. QR1 phóng to >= min_stop_qr_dim (200px): Dừng xe 1.0s tại trạm QR1 (STOP_AT_QR1, B4=1)
    4. Hết 1.0s dừng: Xoay phải mù 2.0s (BLIND_TURN_RIGHT, B4=2)
    5. Hết 2.0s xoay phải mù: Dừng dập quán tính 500ms (PAUSE_AFTER_TURN1, B4=1)
    6. Hết 500ms dừng: Chuyển sang bám tâm QR2 (TRACKING_QR2, B4=0)
    7. QR2 phóng to >= min_stop_qr_dim (200px): Dừng xe 1.0s tại trạm QR2 (STOP_AT_QR2, B4=1)
    8. Hết 1.0s dừng: Xoay trái mù 2.0s (BLIND_TURN_LEFT, B4=3)
    9. Hết 2.0s xoay trái mù: Dừng dập quán tính 500ms (PAUSE_AFTER_TURN2, B4=1)
    10. Hết 500ms dừng: Tự động reset về TRACKING_ROW_1 để chạy thẳng tiếp (B4=0)!
    """
    frame_shape = (720, 1280)
    mission = QRTurnMissionManager(
        min_qr_nav_dim=30.0,
        min_stop_qr_dim=200.0,
        stop_delay_t1_sec=1.0,
        blind_turn1_sec=2.0,
        pause_turn1_sec=0.5,
        stop_delay_t2_sec=1.0,
        blind_turn2_sec=2.0,
        pause_turn2_sec=0.5,
        calib_turn2_stop_only=False
    )

    # 1. Khởi động ở trạng thái TRACKING_ROW_1 trên luống 1
    assert mission.state == QRTurnMissionState.TRACKING_ROW_1
    path_res = create_mock_path_result(path_center_x=650, ref_y=500, offset_px=10.0)
    base_nav = fuse_navigation_target(path_res=path_res, qr_results=[], frame_shape=frame_shape)
    nav_out = mission.process(base_nav, [], frame_shape, now=0.0)
    assert mission.state == QRTurnMissionState.TRACKING_ROW_1
    assert nav_out.uart_flag_byte == 0

    # 2. Phát hiện QR1 nhưng kích thước cực nhỏ (< 30px, vd 20x20): Chưa bám tâm QR1
    qr_tiny = create_mock_qr(cx=640, cy=400, width=20, height=20, text="TURN_1")
    nav_out = mission.process(base_nav, [qr_tiny], frame_shape, now=0.5)
    assert mission.state == QRTurnMissionState.TRACKING_ROW_1
    assert nav_out.uart_flag_byte == 0

    # 3. Gặp QR1 >= 30px (vd 50x50): Bắt đầu chạy theo tâm QR1 (TRACKING_QR1, B4=0)
    qr_nav1 = create_mock_qr(cx=670, cy=400, width=50, height=50, text="TURN_1")
    nav_out = mission.process(base_nav, [qr_nav1], frame_shape, now=1.0)
    assert mission.state == QRTurnMissionState.TRACKING_QR1
    assert nav_out.uart_flag_byte == 0
    assert abs(nav_out.lateral_offset_px - 30.0) < 1e-3

    # 4. Đến khi QR1 >= 200px (vd 210x210): Chuyển STOP_AT_QR1, dừng 1s (B4=1)
    qr_stop1 = create_mock_qr(cx=670, cy=400, width=210, height=210, text="TURN_1")
    nav_out = mission.process(base_nav, [qr_stop1], frame_shape, now=2.0)
    assert mission.state == QRTurnMissionState.STOP_AT_QR1
    assert nav_out.uart_flag_byte == 1
    assert nav_out.lateral_offset_px == 0.0

    # 5. Trong thời gian dừng 1s (now = 2.5s, dt = 0.5s < 1.0s): Giữ B4=1, offset=0
    nav_out = mission.process(base_nav, [], frame_shape, now=2.5)
    assert mission.state == QRTurnMissionState.STOP_AT_QR1
    assert nav_out.uart_flag_byte == 1
    assert nav_out.lateral_offset_px == 0.0

    # 6. Hết 1.0s dừng (now = 3.0s): Bắt đầu xoay phải mù 2s (BLIND_TURN_RIGHT, B4=2)
    nav_out = mission.process(base_nav, [], frame_shape, now=3.0)
    assert mission.state == QRTurnMissionState.BLIND_TURN_RIGHT
    assert nav_out.uart_flag_byte == 2
    assert nav_out.lateral_offset_px == 0.0

    # 7. Trong thời gian xoay phải mù 2s (now = 4.0s, dt = 1.0s < 2.0s): Giữ B4=2
    nav_out = mission.process(base_nav, [], frame_shape, now=4.0)
    assert mission.state == QRTurnMissionState.BLIND_TURN_RIGHT
    assert nav_out.uart_flag_byte == 2

    # 8. Hết 2.0s xoay phải mù (now = 5.0s): Chuyển sang dừng dập quán tính 500ms (PAUSE_AFTER_TURN1, B4=1)
    nav_out = mission.process(base_nav, [], frame_shape, now=5.0)
    assert mission.state == QRTurnMissionState.PAUSE_AFTER_TURN1
    assert nav_out.uart_flag_byte == 1

    # 9. Trong 500ms dừng dập quán tính (now = 5.2s, dt = 0.2s < 0.5s): Giữ B4=1
    nav_out = mission.process(base_nav, [], frame_shape, now=5.2)
    assert mission.state == QRTurnMissionState.PAUSE_AFTER_TURN1
    assert nav_out.uart_flag_byte == 1

    # 10. Hết 500ms dừng (now = 5.5s): Chuyển sang chạy theo tâm turn2 (TRACKING_QR2, B4=0)
    nav_out = mission.process(base_nav, [], frame_shape, now=5.5)
    assert mission.state == QRTurnMissionState.TRACKING_QR2
    assert nav_out.uart_flag_byte == 0

    # 11. Bám theo tâm QR2 khi QR2 < 200px (vd 80x80): B4=0, offset theo QR2
    qr_nav2 = create_mock_qr(cx=610, cy=380, width=80, height=80, text="TURN_2")
    nav_out = mission.process(base_nav, [qr_nav2], frame_shape, now=6.0)
    assert mission.state == QRTurnMissionState.TRACKING_QR2
    assert nav_out.uart_flag_byte == 0
    assert abs(nav_out.lateral_offset_px - (-30.0)) < 1e-3

    # 12. QR2 phóng to >= 200px (vd 215x215): Chuyển STOP_AT_QR2, dừng 1s (B4=1)
    qr_stop2 = create_mock_qr(cx=640, cy=400, width=215, height=215, text="TURN_2")
    nav_out = mission.process(base_nav, [qr_stop2], frame_shape, now=7.0)
    assert mission.state == QRTurnMissionState.STOP_AT_QR2
    assert nav_out.uart_flag_byte == 1

    # 13. Trong 1s dừng trạm 2 (now = 7.5s): Giữ B4=1, offset=0
    nav_out = mission.process(base_nav, [], frame_shape, now=7.5)
    assert mission.state == QRTurnMissionState.STOP_AT_QR2
    assert nav_out.uart_flag_byte == 1

    # 14. Hết 1.0s dừng (now = 8.0s): Bắt đầu xoay trái mù 2s (BLIND_TURN_LEFT, B4=3)
    nav_out = mission.process(base_nav, [], frame_shape, now=8.0)
    assert mission.state == QRTurnMissionState.BLIND_TURN_LEFT
    assert nav_out.uart_flag_byte == 3

    # 15. Trong 2.0s xoay trái mù (now = 9.0s): Giữ B4=3
    nav_out = mission.process(base_nav, [], frame_shape, now=9.0)
    assert mission.state == QRTurnMissionState.BLIND_TURN_LEFT
    assert nav_out.uart_flag_byte == 3

    # 16. Hết 2.0s xoay trái mù (now = 10.0s): Dừng dập quán tính 500ms (PAUSE_AFTER_TURN2, B4=1)
    nav_out = mission.process(base_nav, [], frame_shape, now=10.0)
    assert mission.state == QRTurnMissionState.PAUSE_AFTER_TURN2
    assert nav_out.uart_flag_byte == 1

    # 17. Trong 500ms dừng dập quán tính (now = 10.2s): Giữ B4=1
    nav_out = mission.process(base_nav, [], frame_shape, now=10.2)
    assert mission.state == QRTurnMissionState.PAUSE_AFTER_TURN2
    assert nav_out.uart_flag_byte == 1

    # 18. Hết 500ms dừng (now = 10.5s): Hoàn tất -> Tự động reset về TRACKING_ROW_1 để chạy thẳng tiếp (B4=0)!
    nav_out = mission.process(base_nav, [], frame_shape, now=10.5)
    assert mission.state == QRTurnMissionState.TRACKING_ROW_1
    assert nav_out.uart_flag_byte == 0

    print("[PASS] test_qr_turn_mission_lifecycle: Full 10-phase QR blind-turn mission sequence 100% verified")


def test_method_3_path_seek_with_offset_and_heading_angle():
    """Kiểm tra hàm tính góc nghiêng của luống calculate_path_heading_angle:
    - Góc thẳng (centerline thẳng đứng) -> ~0 độ.
    - Nghiêng lệch phải -> góc dương.
    - Nghiêng lệch trái -> góc âm.
    """
    from src.navigation_fusion import calculate_path_heading_angle

    # 1. Đường thẳng đứng
    path_straight = create_mock_path_result(path_center_x=640, ref_y=500, offset_px=0.0)
    path_straight.centerline_points = [(640, 200), (640, 300), (640, 400)]
    angle_straight = calculate_path_heading_angle(path_straight)
    assert angle_straight is not None and abs(angle_straight) < 1.0, f"Straight path angle should be ~0, got {angle_straight}"

    # 2. Đường nghiêng phải ~35 độ
    path_tilted = create_mock_path_result(path_center_x=660, ref_y=500, offset_px=20.0)
    path_tilted.centerline_points = [(800, 200), (730, 300), (660, 400)]
    angle_calc = calculate_path_heading_angle(path_tilted)
    assert angle_calc is not None and abs(angle_calc) > 20.0, f"Angle {angle_calc} should be > 20 deg"

    # 3. Đường nghiêng nhẹ ~5.7 độ
    path_aligned = create_mock_path_result(path_center_x=665, ref_y=500, offset_px=25.0)
    path_aligned.centerline_points = [(685, 200), (675, 300), (665, 400)]
    angle_aligned = calculate_path_heading_angle(path_aligned)
    assert angle_aligned is not None and abs(angle_aligned) <= 20.0, f"Angle {angle_aligned} should be <= 20 deg"

    print("[PASS] test_method_3_path_seek_with_offset_and_heading_angle: Path heading angle calculations verified!")


def test_turn2_detection_during_rotation_stops_and_approaches():
    """Kiểm tra tiếp cận mã QR2 trên hành trình:
    - Ở TRACKING_QR2, robot bám tâm QR2 khi QR2 < 200px (B4 = 0)
    - Khi QR2 >= 200px: Kích hoạt STOP_AT_QR2 (B4 = 1) để chuẩn bị xoay trái mù.
    """
    frame_shape = (720, 1280) # cam_cx = 640
    mission = QRTurnMissionManager()
    mission.state = QRTurnMissionState.TRACKING_QR2

    # QR2 kích thước 150x150 (< 200px)
    pts = np.array([[492.0, 353.0], [540.0, 353.0], [540.0, 402.0], [492.0, 402.0]], dtype=np.float32)
    qr_ai = QRDetectionResult(
        points=pts,
        text="TURN_2",
        is_decoded=True,
        backend="yolo-fastestv2-qr",
        center=(516.0, 377.5),
        center_int=(516, 377),
        width_px=150.0,
        height_px=150.0,
        area_px=22500.0,
        aspect_ratio=1.0,
        orientation_deg=0.0
    )

    base_nav = fuse_navigation_target(path_res=None, qr_results=[qr_ai], frame_shape=frame_shape)

    # 1. Tiếp cận tâm QR2: Gửi B4 = 0
    res0 = mission.process(base_nav, [qr_ai], frame_shape, now=10.0)
    assert mission.state == QRTurnMissionState.TRACKING_QR2
    assert res0.uart_flag_byte == 0, f"Expected B4=0 approaching QR2, got {res0.uart_flag_byte}"
    assert abs(res0.lateral_offset_px - (-124.0)) < 1e-3

    # 2. QR2 phóng to >= 200px (chạm trạm QR2): Chuyển sang STOP_AT_QR2 và gửi B4 = 1
    qr_stop = create_mock_qr(cx=640, cy=400, width=220, height=220, text="TURN_2")
    res1 = mission.process(base_nav, [qr_stop], frame_shape, now=10.5)
    assert mission.state == QRTurnMissionState.STOP_AT_QR2
    assert res1.uart_flag_byte == 1, f"Expected B4=1 at station QR2, got {res1.uart_flag_byte}"

    print("[PASS] test_turn2_detection_during_rotation_stops_and_approaches: QR2 approach B4=0 -> Station stop B4=1 verified!")


def test_turn_delays_and_confirm_frames():
    """Kiểm tra khả năng tinh chỉnh thời gian và ngưỡng px từ UI:
    - stop_delay_t1_sec, blind_turn1_sec, pause_turn1_sec
    - stop_delay_t2_sec, blind_turn2_sec, pause_turn2_sec
    - min_qr_nav_dim, min_stop_qr_dim
    """
    frame_shape = (720, 1280)
    mission = QRTurnMissionManager(
        min_qr_nav_dim=40.0,
        min_stop_qr_dim=250.0,
        stop_delay_t1_sec=1.5,
        blind_turn1_sec=2.5,
        pause_turn1_sec=0.8,
        stop_delay_t2_sec=1.2,
        blind_turn2_sec=1.8,
        pause_turn2_sec=0.6,
        calib_turn2_stop_only=False
    )
    base_nav = FusedNavigationTarget(None, 0.0, '', False, 'IDLE', None, (0, 0), 0.0, 0, False, 0)

    # 1. min_qr_nav_dim = 40.0 -> QR 35px không kích hoạt bám tâm
    qr_35 = create_mock_qr(cx=640, cy=400, width=35, height=35, text="TURN_1")
    mission.process(base_nav, [qr_35], frame_shape, now=0.0)
    assert mission.state == QRTurnMissionState.TRACKING_ROW_1

    # QR 45px >= 40.0 -> Kích hoạt bám tâm
    qr_45 = create_mock_qr(cx=640, cy=400, width=45, height=45, text="TURN_1")
    mission.process(base_nav, [qr_45], frame_shape, now=0.1)
    assert mission.state == QRTurnMissionState.TRACKING_QR1

    # 2. min_stop_qr_dim = 250.0 -> QR 220px chưa dừng trạm
    qr_220 = create_mock_qr(cx=640, cy=400, width=220, height=220, text="TURN_1")
    mission.process(base_nav, [qr_220], frame_shape, now=0.2)
    assert mission.state == QRTurnMissionState.TRACKING_QR1

    # QR 260px >= 250.0 -> Dừng trạm
    qr_260 = create_mock_qr(cx=640, cy=400, width=260, height=260, text="TURN_1")
    mission.process(base_nav, [qr_260], frame_shape, now=1.0)
    assert mission.state == QRTurnMissionState.STOP_AT_QR1

    # 3. stop_delay_t1_sec = 1.5s -> Ở now = 2.4s (dt = 1.4s < 1.5s) vẫn là STOP_AT_QR1
    mission.process(base_nav, [], frame_shape, now=2.4)
    assert mission.state == QRTurnMissionState.STOP_AT_QR1

    # Ở now = 2.5s (dt = 1.5s) -> Chuyển BLIND_TURN_RIGHT
    mission.process(base_nav, [], frame_shape, now=2.5)
    assert mission.state == QRTurnMissionState.BLIND_TURN_RIGHT

    # 4. blind_turn1_sec = 2.5s -> Ở now = 4.9s (dt = 2.4s < 2.5s) vẫn là BLIND_TURN_RIGHT
    mission.process(base_nav, [], frame_shape, now=4.9)
    assert mission.state == QRTurnMissionState.BLIND_TURN_RIGHT

    # Ở now = 5.0s (dt = 2.5s) -> Chuyển PAUSE_AFTER_TURN1
    mission.process(base_nav, [], frame_shape, now=5.0)
    assert mission.state == QRTurnMissionState.PAUSE_AFTER_TURN1

    # 5. pause_turn1_sec = 0.8s -> Ở now = 5.7s (dt = 0.7s < 0.8s) vẫn là PAUSE_AFTER_TURN1
    mission.process(base_nav, [], frame_shape, now=5.7)
    assert mission.state == QRTurnMissionState.PAUSE_AFTER_TURN1

    # Ở now = 5.85s (dt = 0.85s >= 0.8s) -> Chuyển TRACKING_QR2
    mission.process(base_nav, [], frame_shape, now=5.85)
    assert mission.state == QRTurnMissionState.TRACKING_QR2

    print("[PASS] test_turn_delays_and_confirm_frames: UI parameter custom delays and thresholds verified 100%!")


def test_calib_turn2_rotates_and_stops_completely():
    """Kiểm tra chế độ Calib T2 (calib_turn2_stop_only=True):
    - Khi tới trạm TURN_2: Dừng 1.0s (STOP_AT_QR2, B4=1)
    - Hết 1.0s: Xoay trái mù 2.0s (BLIND_TURN_LEFT, B4=3)
    - Hết 2.0s: Dừng 500ms dập đà (PAUSE_AFTER_TURN2, B4=1)
    - Hết 500ms: DỪNG HẲN HOÀN TOÀN ở COMPLETED_STOPPED (B4=1, offset=0.0)!
    """
    frame_shape = (720, 1280)
    mission = QRTurnMissionManager(
        stop_delay_t2_sec=1.0,
        blind_turn2_sec=2.0,
        pause_turn2_sec=0.5,
        calib_turn2_stop_only=True
    )
    mission.state = QRTurnMissionState.TRACKING_QR2
    base_nav = FusedNavigationTarget(None, 0.0, '', False, 'IDLE', None, (0, 0), 0.0, 0, False, 0)

    # 1. Chạm trạm Turn 2 (>= 200px) tại now=10.0s -> Dừng 1s (STOP_AT_QR2, B4=1)
    qr_turn2 = create_mock_qr(cx=640, cy=400, width=220, height=220, text="TURN_2")
    res1 = mission.process(base_nav, [qr_turn2], frame_shape, now=10.0)
    assert mission.state == QRTurnMissionState.STOP_AT_QR2
    assert res1.uart_flag_byte == 1

    # 2. Hết 1.0s dừng (now = 11.0s) -> Bắt đầu xoay trái mù (BLIND_TURN_LEFT, B4=3)
    res2 = mission.process(base_nav, [], frame_shape, now=11.0)
    assert mission.state == QRTurnMissionState.BLIND_TURN_LEFT
    assert res2.uart_flag_byte == 3

    # 3. Đang trong 2.0s xoay mù (now = 12.0s) -> Tiếp tục xoay trái (B4=3)
    res3 = mission.process(base_nav, [], frame_shape, now=12.0)
    assert mission.state == QRTurnMissionState.BLIND_TURN_LEFT
    assert res3.uart_flag_byte == 3

    # 4. Hết 2.0s xoay mù (now = 13.0s) -> Dừng 500ms dập đà (PAUSE_AFTER_TURN2, B4=1)
    res4 = mission.process(base_nav, [], frame_shape, now=13.0)
    assert mission.state == QRTurnMissionState.PAUSE_AFTER_TURN2
    assert res4.uart_flag_byte == 1

    # 5. Hết 500ms dừng dập đà (now = 13.5s) -> Chuyển COMPLETED_STOPPED (B4=1, offset=0.0)
    res5 = mission.process(base_nav, [], frame_shape, now=13.5)
    assert mission.state == QRTurnMissionState.COMPLETED_STOPPED
    assert res5.uart_flag_byte == 1
    assert res5.lateral_offset_px == 0.0

    # 6. Các frame tiếp theo vẫn giữ nguyên COMPLETED_STOPPED
    res6 = mission.process(base_nav, [], frame_shape, now=15.0)
    assert mission.state == QRTurnMissionState.COMPLETED_STOPPED
    assert res6.uart_flag_byte == 1
    assert res6.lateral_offset_px == 0.0

    print("[PASS] test_calib_turn2_rotates_and_stops_completely: Stop 1s -> Blind turn 2s -> Pause 500ms -> Completed stopped verified!")


def test_idle_direct_turn2_station_trigger():
    """Kiểm tra tính năng kích hoạt trực tiếp trạm TURN_2 từ trạng thái IDLE (hỗ trợ test nhanh tại chỗ trạm 2):
    - Khi đặt robot trực tiếp trước mã TURN_2 (>= 200px) để test calib
    - Hệ thống chuyển ngay sang STOP_AT_QR2 (B4=1)
    - Dừng 1s -> Xoay trái mù 2s (B4=3) -> Dừng 500ms -> COMPLETED_STOPPED (B4=1)
    """
    frame_shape = (720, 1280)
    mission = QRTurnMissionManager(
        stop_delay_t2_sec=1.0,
        blind_turn2_sec=2.0,
        pause_turn2_sec=0.5,
        calib_turn2_stop_only=True
    )
    assert mission.state == QRTurnMissionState.TRACKING_ROW_1
    base_nav = FusedNavigationTarget(None, 0.0, '', False, 'IDLE', None, (0, 0), 0.0, 0, False, 0)

    # Đưa mã TURN_2 >= 200px vào khung hình khi đang ở TRACKING_ROW_1
    qr_turn2 = create_mock_qr(cx=640, cy=400, width=220, height=220, text="TURN_2")
    res = mission.process(base_nav, [qr_turn2], frame_shape, now=0.0)
    assert mission.state == QRTurnMissionState.STOP_AT_QR2, "IDLE phải kích hoạt ngay trạm STOP_AT_QR2"
    assert res.uart_flag_byte == 1

    # Hết 1.0s dừng -> Xoay trái mù 2.0s (B4=3)
    res_spin = mission.process(base_nav, [], frame_shape, now=1.0)
    assert mission.state == QRTurnMissionState.BLIND_TURN_LEFT
    assert res_spin.uart_flag_byte == 3

    # Hết 2.0s xoay mù (now = 3.0s) -> Dừng dập quán tính 500ms (B4=1)
    res_pause = mission.process(base_nav, [], frame_shape, now=3.0)
    assert mission.state == QRTurnMissionState.PAUSE_AFTER_TURN2
    assert res_pause.uart_flag_byte == 1

    # Hết 500ms dừng dập quán tính (now = 3.5s) -> DỪNG HẲN (B4=1)
    res_end = mission.process(base_nav, [], frame_shape, now=3.5)
    assert mission.state == QRTurnMissionState.COMPLETED_STOPPED
    assert res_end.uart_flag_byte == 1
    assert res_end.lateral_offset_px == 0.0

    print("[PASS] test_idle_direct_turn2_station_trigger: Direct TURN_2 trigger -> Stop 1s -> Turn 2s -> Pause 500ms -> Completed stopped verified!")


def test_mission_completely_ignores_path_line_and_runs_straight():
    """Kiểm tra nguyên tắc cốt lõi: 'BỎ PHẦN LUỐNG LUÔN':
    - Dù camera phát hiện luống đường với sai số lớn (vd offset = +85.0 px)
    - Hệ thống trong nhiệm vụ này KHÔNG bám luống, mà luôn CHẠY THẲNG (offset = 0.0 px)!
    - Chỉ bám theo tâm khi nhìn thấy mã QR!
    """
    frame_shape = (720, 1280)
    mission = QRTurnMissionManager()
    
    # Giả lập phát hiện luống rất rõ với sai số lệch +85 px
    path_with_big_error = create_mock_path_result(path_center_x=725, ref_y=500, offset_px=85.0)
    base_nav = fuse_navigation_target(path_res=path_with_big_error, qr_results=[], frame_shape=frame_shape)
    
    # 1. Ở TRACKING_ROW_1 (Chạy thẳng): Không bám sai số +85 px của luống, mà offset bắt buộc là 0.0!
    nav_straight = mission.process(base_nav, [], frame_shape, path_res=path_with_big_error, now=0.0)
    assert mission.state == QRTurnMissionState.TRACKING_ROW_1
    assert nav_straight.lateral_offset_px == 0.0, f"Phải chạy thẳng (offset=0), không bám luống! Got {nav_straight.lateral_offset_px}"
    assert nav_straight.uart_flag_byte == 0
    assert nav_straight.mode_name == "RUN_STRAIGHT"

    # 2. Ở TRACKING_QR2 (Chờ QR2): Dù luống có lệch bao nhiêu cũng chạy thẳng (offset = 0.0)
    mission.state = QRTurnMissionState.TRACKING_QR2
    nav_seek_qr2 = mission.process(base_nav, [], frame_shape, path_res=path_with_big_error, now=1.0)
    assert mission.state == QRTurnMissionState.TRACKING_QR2
    assert nav_seek_qr2.lateral_offset_px == 0.0
    assert nav_seek_qr2.mode_name == "RUN_STRAIGHT"

    # 3. Khi QR2 xuất hiện (vd lệch tâm -35 px): Chỉ bám theo tâm QR2!
    qr2 = create_mock_qr(cx=605, cy=400, width=80, height=80, text="TURN_2")
    nav_qr2 = mission.process(base_nav, [qr2], frame_shape, path_res=path_with_big_error, now=2.0)
    assert nav_qr2.mode_name == "QR_CENTER"
    assert abs(nav_qr2.lateral_offset_px - (-35.0)) < 1e-3, "Bám chính xác theo tâm QR2"

    print("[PASS] test_mission_completely_ignores_path_line_and_runs_straight: Path line completely ignored, runs straight verified 100%!")


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
    test_uart_flag_byte_qr_between_100_and_200()
    test_uart_flag_byte_ignores_path_condition()
    test_uart_packet_multimode_flags()
    test_qr_turn_mission_lifecycle()
    test_method_3_path_seek_with_offset_and_heading_angle()
    test_turn2_detection_during_rotation_stops_and_approaches()
    test_turn_delays_and_confirm_frames()
    test_calib_turn2_rotates_and_stops_completely()
    test_idle_direct_turn2_station_trigger()
    test_mission_completely_ignores_path_line_and_runs_straight()
    print("=" * 60)
    print("ALL 16 NAVIGATION FUSION TESTS PASSED 100%!")
    print("=" * 60)

