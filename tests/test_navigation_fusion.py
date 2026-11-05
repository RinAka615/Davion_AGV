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
    """Kiểm tra toàn bộ chu trình nhiệm vụ 2 QR Ziczac (Luống 1 -> QR1 -> Luống 2 -> QR2 -> Luống 1):
    1. IDLE: Bám luống 1
    2. Gặp QR1 > 100px: Bám tâm QR1 (Byte 4 = 0)
    3. Gặp QR1 >= 200px: Chuyển TURNING_RIGHT_SEEK_PATH (Byte 4 = 1) để STM32 dừng 1.0s và quay PHẢI
    4. Trong delay_turn1_sec: Giữ Byte 4 = 1, offset = 0.0
    5. Sau delay_turn1_sec: Camera bắt được Luống 2 (|dx| <= 40px) -> Chuyển sang TRACKING_ROW_2 (Byte 4 = 0)
    6. Đang bám Luống 2: Thấy QR2 > 100px -> Bám tâm QR2 (Byte 4 = 0)
    7. Gặp QR2 >= 200px: Chuyển TURNING_LEFT_SEEK_PATH (Byte 4 = 3) để STM32 dừng 1.0s và quay TRÁI
    8. Trong delay_turn2_sec: Giữ Byte 4 = 3, offset = 0.0
    9. Sau delay_turn2_sec: Camera bắt được Luống mới (|dx| <= 40px) -> Loop về IDLE (Byte 4 = 0)
    """
    frame_shape = (720, 1280)
    mission = QRTurnMissionManager(
        stop_duration_sec=1.0,
        pause_center_sec=0.5,
        align_tolerance_px=180.0,
        min_stop_qr_dim=200.0,
        path_seek_tolerance_px=40.0,
        path_seek_max_angle_deg=20.0,
        delay_turn1_sec=1.0,
        delay_turn2_sec=1.0,
        confirm_frames=2,
        calib_turn2_stop_only=False
    )

    # 1. Khởi động ở trạng thái IDLE trên luống 1
    assert mission.state == QRTurnMissionState.IDLE
    path_res = create_mock_path_result(path_center_x=650, ref_y=500, offset_px=10.0)
    base_nav = fuse_navigation_target(path_res=path_res, qr_results=[], frame_shape=frame_shape)
    nav_out = mission.process(base_nav, [], frame_shape, now=0.0)
    assert mission.state == QRTurnMissionState.IDLE
    assert nav_out.uart_flag_byte == 0

    # 2. Phát hiện QR1 nhưng kích thước nhỏ (120x120 < 200px)
    qr_turn1_small = create_mock_qr(cx=640, cy=400, width=120, height=120, text="TURN_1")
    base_nav = fuse_navigation_target(path_res=path_res, qr_results=[qr_turn1_small], frame_shape=frame_shape)
    nav_out = mission.process(base_nav, [qr_turn1_small], frame_shape, now=1.0)
    assert mission.state == QRTurnMissionState.IDLE
    assert nav_out.uart_flag_byte == 0, "Chưa dừng khi QR1 < 200px"

    # 3. Chạm trạm QR1 (>= 200px, vd 210x210): Kích hoạt trạm dừng QR1!
    qr_turn1_large = create_mock_qr(cx=640, cy=400, width=210, height=210, text="TURN_1")
    base_nav = fuse_navigation_target(path_res=path_res, qr_results=[qr_turn1_large], frame_shape=frame_shape)
    nav_out = mission.process(base_nav, [qr_turn1_large], frame_shape, now=2.0)
    assert mission.state == QRTurnMissionState.TURNING_RIGHT_SEEK_PATH
    assert nav_out.uart_flag_byte == 1, "Chạm trạm QR1 -> Gửi B4=1 để STM32 dừng 1.0s và quay phải"

    # 4. Trong 1.0s trễ đầu (now = 2.5s, dt = 0.5s < 1.0s): Duy trì B4 = 1, offset = 0.0
    nav_out = mission.process(base_nav, [], frame_shape, now=2.5)
    assert mission.state == QRTurnMissionState.TURNING_RIGHT_SEEK_PATH
    assert nav_out.uart_flag_byte == 1
    assert nav_out.lateral_offset_px == 0.0

    # 5. Hết 1.0s trễ (now = 3.2s): Bắt đầu quét tìm luống 2
    # Frame 1: Thấy luống 2 với sai số |offset| = 15px <= 40px
    path_row2 = create_mock_path_result(path_center_x=655, ref_y=500, offset_px=15.0)
    base_nav_row2 = fuse_navigation_target(path_res=path_row2, qr_results=[], frame_shape=frame_shape)
    nav_out = mission.process(base_nav_row2, [], frame_shape, path_res=path_row2, now=3.2)
    assert mission.state == QRTurnMissionState.TURNING_RIGHT_SEEK_PATH
    assert nav_out.uart_flag_byte == 1, "Frame 1: Đang xác nhận ổn định"

    # Frame 2: Xác nhận đủ 2 frame liên tiếp -> BẮT ĐƯỢC LUỐNG 2!
    nav_out = mission.process(base_nav_row2, [], frame_shape, path_res=path_row2, now=3.25)
    assert mission.state == QRTurnMissionState.TRACKING_ROW_2, "Đủ frame -> Chuyển sang TRACKING_ROW_2"
    assert nav_out.uart_flag_byte == 0, "Gửi B4=0 để STM32 dừng 500ms dập đà xoay rồi bám luống 2"
    assert abs(nav_out.lateral_offset_px - 15.0) < 1e-3

    # 6. Đang bám luống 2, phát hiện QR2 < 200px: Vẫn duy trì B4 = 0
    qr_turn2_small = create_mock_qr(cx=640, cy=400, width=150, height=150, text="TURN_2")
    base_nav_row2 = fuse_navigation_target(path_res=path_row2, qr_results=[qr_turn2_small], frame_shape=frame_shape)
    nav_out = mission.process(base_nav_row2, [qr_turn2_small], frame_shape, now=4.0)
    assert mission.state == QRTurnMissionState.TRACKING_ROW_2
    assert nav_out.uart_flag_byte == 0

    # 7. Chạm trạm QR2 (>= 200px, vd 215x215): Kích hoạt trạm dừng QR2!
    qr_turn2_large = create_mock_qr(cx=640, cy=400, width=215, height=215, text="TURN_2")
    nav_out = mission.process(base_nav_row2, [qr_turn2_large], frame_shape, now=5.0)
    assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_PATH
    assert nav_out.uart_flag_byte == 3, "Chạm trạm QR2 -> Gửi B4=3 để STM32 dừng 1.0s và quay trái"

    # 8. Trong 1.0s trễ đầu (now = 5.6s, dt = 0.6s < 1.0s): Duy trì B4 = 3, offset = 0.0
    nav_out = mission.process(base_nav_row2, [], frame_shape, now=5.6)
    assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_PATH
    assert nav_out.uart_flag_byte == 3
    assert nav_out.lateral_offset_px == 0.0

    # 9. Hết 1.0s trễ (now = 6.2s): Bắt đầu quét tìm luống mới
    # Frame 1: Thấy luống mới với sai số |offset| = 20px <= 40px
    path_new = create_mock_path_result(path_center_x=660, ref_y=500, offset_px=20.0)
    base_nav_new = fuse_navigation_target(path_res=path_new, qr_results=[], frame_shape=frame_shape)
    nav_out = mission.process(base_nav_new, [], frame_shape, path_res=path_new, now=6.2)
    assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_PATH
    assert nav_out.uart_flag_byte == 3, "Frame 1: Đang xác nhận ổn định"

    # Frame 2: Xác nhận đủ 2 frame liên tiếp -> LOOP VỀ IDLE!
    nav_out = mission.process(base_nav_new, [], frame_shape, path_res=path_new, now=6.25)
    assert mission.state == QRTurnMissionState.IDLE, "Đủ frame -> Tự động chuyển về IDLE để bám luống mới từ đầu!"
    assert nav_out.uart_flag_byte == 0, "Gửi B4=0 để STM32 dừng 500ms dập đà xoay rồi bám luống mới"
    assert abs(nav_out.lateral_offset_px - 20.0) < 1e-3

    # 10. Tiếp tục bám luống mới trong IDLE
    nav_out = mission.process(base_nav_new, [], frame_shape, path_res=path_new, now=7.0)
    assert mission.state == QRTurnMissionState.IDLE
    assert nav_out.uart_flag_byte == 0

    print("[PASS] test_qr_turn_mission_lifecycle: Full 2-QR Ziczac mission cycle (IDLE -> QR1 B4=1 -> SEEK ROW2 -> TRACK ROW2 B4=0 -> QR2 B4=3 -> SEEK NEW ROW -> IDLE B4=0) 100% verified")


def test_method_3_path_seek_with_offset_and_heading_angle():
    """Kiểm tra điều kiện chuyển bước bắt luống mới yêu cầu ĐỒNG THỜI:
    1) |lateral_offset_px| <= 40px
    2) |heading_angle| <= 20 độ
    Nếu không thỏa mãn một trong hai -> Tiếp tục xoay trái (B4=3).
    """
    from src.navigation_fusion import calculate_path_heading_angle

    frame_shape = (720, 1280)
    mission = QRTurnMissionManager(
        stop_duration_sec=1.0,
        pause_center_sec=0.5,
        align_tolerance_px=100.0,
        min_stop_qr_dim=200.0,
        path_seek_tolerance_px=40.0,
        path_seek_max_angle_deg=20.0,
        delay_turn2_sec=1.0,
        confirm_frames=2,
        calib_turn2_stop_only=False
    )

    # Đưa mission vào trạng thái TURNING_LEFT_SEEK_PATH (đang xoay trái tìm luống)
    mission.state = QRTurnMissionState.TURNING_LEFT_SEEK_PATH
    mission.turn_right_start_time = 100.0

    # 1. Luống lệch tâm quá xa (|offset| = 80px > 40px) dù góc thẳng (angle = 0)
    path_far_offset = create_mock_path_result(path_center_x=720, ref_y=500, offset_px=80.0)
    base_nav = fuse_navigation_target(path_res=path_far_offset, qr_results=[], frame_shape=frame_shape)
    nav_out = mission.process(base_nav, [], frame_shape, path_res=path_far_offset, now=101.5)
    assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_PATH, "Lệch tâm > 40px -> Phải tiếp tục xoay trái tìm luống"
    assert nav_out.uart_flag_byte == 3

    # 2. Luống ở gần tâm (|offset| = 20px <= 40px) nhưng góc nghiêng quá chéo (|angle| = ~35 độ > 20 độ)
    path_tilted = create_mock_path_result(path_center_x=660, ref_y=500, offset_px=20.0)
    path_tilted.centerline_points = [(800, 200), (730, 300), (660, 400)]
    angle_calc = calculate_path_heading_angle(path_tilted)
    assert angle_calc is not None and abs(angle_calc) > 20.0, f"Angle {angle_calc} should be > 20 deg"
    
    base_nav = fuse_navigation_target(path_res=path_tilted, qr_results=[], frame_shape=frame_shape)
    nav_out = mission.process(base_nav, [], frame_shape, path_res=path_tilted, now=101.8)
    assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_PATH, "Góc chéo > 20 độ -> Phải tiếp tục xoay trái để nắn thẳng"
    assert nav_out.uart_flag_byte == 3

    # 3. Luống thỏa mãn CẢ HAI (|offset| = 25px <= 40px VÀ |angle| = ~5.7 độ <= 20 độ)
    path_aligned = create_mock_path_result(path_center_x=665, ref_y=500, offset_px=25.0)
    path_aligned.centerline_points = [(685, 200), (675, 300), (665, 400)]
    angle_aligned = calculate_path_heading_angle(path_aligned)
    assert angle_aligned is not None and abs(angle_aligned) <= 20.0

    base_nav = fuse_navigation_target(path_res=path_aligned, qr_results=[], frame_shape=frame_shape)
    # Frame 1: Nhận diện thỏa mãn
    nav_out = mission.process(base_nav, [], frame_shape, path_res=path_aligned, now=102.0)
    assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_PATH
    assert nav_out.uart_flag_byte == 3
    # Frame 2: Xác nhận ổn định -> Chuyển sang IDLE và phát Byte 4 = 0!
    nav_out = mission.process(base_nav, [], frame_shape, path_res=path_aligned, now=102.1)
    assert mission.state == QRTurnMissionState.IDLE, "Thỏa mãn cả offset <= 40px và angle <= 20 độ -> Chuyển sang bám luống!"
    assert nav_out.uart_flag_byte == 0, "Gửi Byte 4 = 0 để STM32 phanh 500ms rồi bám luống chạy tiếp"

    print("[PASS] test_method_3_path_seek_with_offset_and_heading_angle: Dual condition (|offset|<=40px AND |angle|<=20deg) 100% verified!")


def test_turn2_detection_during_rotation_stops_and_approaches():
    """Kiểm tra tiếp cận mã QR2 trên Luống 2 (TRACKING_ROW_2):
    - Đang ở TRACKING_ROW_2, robot bám tâm QR2 khi QR2 > 100px (B4 = 0)
    - Khi QR2 >= 200px: Kích hoạt trạm dừng QR2 (B4 = 3)
    """
    frame_shape = (720, 1280) # cam_cx = 640
    mission = QRTurnMissionManager(align_tolerance_px=180.0, confirm_frames=2)
    mission.state = QRTurnMissionState.TRACKING_ROW_2

    # QR2 kích thước 150x150 (>100px nhưng <200px)
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
    assert mission.state == QRTurnMissionState.TRACKING_ROW_2
    assert res0.uart_flag_byte == 0, f"Expected B4=0 approaching QR2, got {res0.uart_flag_byte}"
    assert abs(res0.lateral_offset_px - (-124.0)) < 1e-3

    # 2. QR2 phóng to >= 200px (chạm trạm QR2): Chuyển sang TURNING_LEFT_SEEK_PATH và gửi B4 = 3
    qr_stop = create_mock_qr(cx=640, cy=400, width=220, height=220, text="TURN_2")
    res1 = mission.process(base_nav, [qr_stop], frame_shape, now=10.5)
    assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_PATH
    assert res1.uart_flag_byte == 3, f"Expected B4=3 at station QR2, got {res1.uart_flag_byte}"

    print("[PASS] test_turn2_detection_during_rotation_stops_and_approaches: QR2 approach B4=0 -> Station stop B4=3 verified!")


def test_turn_delays_and_confirm_frames():
    """Kiểm tra các thông số tùy chỉnh mới trong chu trình 2 QR ziczac:
    1. delay_turn1_sec: Trong thời gian trễ bắt đầu xoay tại Turn 1, gửi B4=1, offset=0, bỏ qua mọi nhiễu đường.
    2. delay_turn2_sec: Trong thời gian trễ bắt đầu xoay tại Turn 2, gửi B4=3, offset=0, bỏ qua mọi nhiễu đường.
    3. confirm_frames: Yêu cầu đủ N frames liên tiếp mới chốt khóa luống đường mới (Luống 2 hoặc Luống mới).
    """
    frame_shape = (720, 1280)
    mission = QRTurnMissionManager(
        delay_turn1_sec=1.5,
        delay_turn2_sec=1.2,
        confirm_frames=3,
        path_seek_tolerance_px=40.0,
        min_stop_qr_dim=200.0,
        calib_turn2_stop_only=False
    )

    base_nav = FusedNavigationTarget(None, 0.0, '', False, 'IDLE', None, (0, 0), 0.0, 0, False, 0)

    # --- KỊCH BẢN 1: TRỄ VÀ XÁC NHẬN FRAME TẠI TURN 1 (QUAY PHẢI TÌM LUỐNG 2) ---
    qr_turn1 = create_mock_qr(cx=640, cy=400, width=220, height=220, text="TURN_1")
    res1 = mission.process(base_nav, [qr_turn1], frame_shape, now=1.0)
    assert mission.state == QRTurnMissionState.TURNING_RIGHT_SEEK_PATH
    assert res1.uart_flag_byte == 1

    # Trong 1.5s trễ đầu (now = 1.8s, dt = 0.8s < 1.5s):
    # Dù có đường (nhiễu luống) lọt vào khung hình, vẫn phải bỏ qua và giữ B4=1, offset=0.0!
    mock_path_row2 = create_mock_path_result(path_center_x=640, ref_y=500, offset_px=0.0)
    res_blind = mission.process(base_nav, [], frame_shape, path_res=mock_path_row2, now=1.8)
    assert mission.state == QRTurnMissionState.TURNING_RIGHT_SEEK_PATH
    assert res_blind.uart_flag_byte == 1, "Trong 1.5s trễ Turn 1 -> Duy trì B4=1, không bắt luống tránh ủi"
    assert res_blind.lateral_offset_px == 0.0
    assert mission.consecutive_path_frames == 0

    # Hết 1.5s trễ (now = 2.6s, dt = 1.6s >= 1.5s): Bắt đầu tìm kiếm Luống 2
    # Với confirm_frames = 3: Cần đúng 3 frames liên tiếp
    # Frame 1:
    res_f1 = mission.process(base_nav, [], frame_shape, path_res=mock_path_row2, now=2.6)
    assert mission.state == QRTurnMissionState.TURNING_RIGHT_SEEK_PATH
    assert res_f1.uart_flag_byte == 1
    assert mission.consecutive_path_frames == 1

    # Giả lập frame 2 bị mất dấu do bụi / chớp nháy:
    res_miss = mission.process(base_nav, [], frame_shape, path_res=None, now=2.64)
    assert mission.consecutive_path_frames == 0, "Mất dấu phải reset counter về 0"
    assert res_miss.uart_flag_byte == 1

    # Lại thấy liên tiếp 3 frames:
    res_c1 = mission.process(base_nav, [], frame_shape, path_res=mock_path_row2, now=2.68)
    assert mission.consecutive_path_frames == 1
    assert mission.state == QRTurnMissionState.TURNING_RIGHT_SEEK_PATH
    assert res_c1.uart_flag_byte == 1

    res_c2 = mission.process(base_nav, [], frame_shape, path_res=mock_path_row2, now=2.72)
    assert mission.consecutive_path_frames == 2
    assert mission.state == QRTurnMissionState.TURNING_RIGHT_SEEK_PATH
    assert res_c2.uart_flag_byte == 1

    res_c3 = mission.process(base_nav, [], frame_shape, path_res=mock_path_row2, now=2.76)
    assert mission.consecutive_path_frames == 0  # Đã chuyển state nên counter được reset
    assert mission.state == QRTurnMissionState.TRACKING_ROW_2, "Đủ 3 frames liên tiếp -> Bắt Luống 2 thành công!"
    assert res_c3.uart_flag_byte == 0, "Gửi B4=0 để STM32 dừng 500ms dập đà rồi bám luống 2"

    # --- KỊCH BẢN 2: TRỄ VÀ XÁC NHẬN FRAME TẠI TURN 2 (QUAY TRÁI TÌM LUỐNG MỚI) ---
    # Tiến tới trạm Turn 2 (>= 200px)
    qr_turn2_station = create_mock_qr(cx=640, cy=400, width=210, height=210, text="TURN_2")
    res_reach = mission.process(base_nav, [qr_turn2_station], frame_shape, now=4.0)
    assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_PATH
    assert res_reach.uart_flag_byte == 3

    # Trong 1.2s trễ tại Turn 2 (now = 4.8s, dt = 0.8s < 1.2s):
    # Dù có đường (nhiễu mép luống cũ) thì KHÔNG được bắt để tránh ủi bậy!
    res_t2_blind = mission.process(base_nav, [], frame_shape, path_res=mock_path_row2, now=4.8)
    assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_PATH
    assert res_t2_blind.uart_flag_byte == 3, "Trong 1.2s trễ Turn 2 -> Duy trì B4=3, không bắt đường tránh ủi"
    assert mission.consecutive_path_frames == 0

    # Hết 1.2s (now = 5.3s, dt = 1.3s >= 1.2s): Bắt đầu kiểm tra confirm 3 frames đường mới
    for i in range(1, 3):
        res_p = mission.process(base_nav, [], frame_shape, path_res=mock_path_row2, now=5.3 + i*0.04)
        assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_PATH
        assert mission.consecutive_path_frames == i
        assert res_p.uart_flag_byte == 3

    # Frame thứ 3: Hoàn tất bắt đường mới!
    res_p3 = mission.process(base_nav, [], frame_shape, path_res=mock_path_row2, now=5.42)
    assert mission.state == QRTurnMissionState.IDLE, "Đủ 3 frames đường -> Bắt đầu bám luống mới (IDLE) thành công!"
    assert res_p3.uart_flag_byte == 0, "Gửi B4=0 để STM32 dừng 500ms dập đà rồi bám luống mới"

    print("[PASS] test_turn_delays_and_confirm_frames: Delay Turn 1 (1.5s), Delay Turn 2 (1.2s), Confirm Frames (3 frames) 100% verified!")


def test_calib_turn2_rotates_and_stops_completely():
    """Kiểm tra chế độ Calib T2 (calib_turn2_stop_only=True):
    - Khi tới trạm TURN_2: Bắt đầu xoay trái với B4=3.
    - Trong 3.0s: Liên tục xoay trái (B4=3, offset=0.0).
    - Hết 3.0s: DỪNG HẲN HOÀN TOÀN:
      + 0.5s đầu gửi B4=0 để STM32 ngắt lệnh xoay Step 7 và hãm phanh triệt tiêu quán tính.
      + Sau 0.5s gửi B4=1 (offset=0.0) để giữ xe đứng yên tuyệt đối, không dò đường, không bám luống!
    """
    frame_shape = (720, 1280)
    mission = QRTurnMissionManager(
        delay_turn2_sec=3.0,
        calib_turn2_stop_only=True
    )
    mission.state = QRTurnMissionState.TRACKING_ROW_2
    base_nav = FusedNavigationTarget(None, 0.0, '', False, 'IDLE', None, (0, 0), 0.0, 0, False, 0)

    # 1. Chạm trạm Turn 2 (>= 200px) tại now=10.0s
    qr_turn2 = create_mock_qr(cx=640, cy=400, width=220, height=220, text="TURN_2")
    res1 = mission.process(base_nav, [qr_turn2], frame_shape, now=10.0)
    assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_PATH
    assert res1.uart_flag_byte == 3

    # 2. Đang trong 3.0s xoay calib (now = 12.0s, dt = 2.0s < 3.0s): Tiếp tục xoay trái (B4=3)
    res_spinning = mission.process(base_nav, [], frame_shape, now=12.0)
    assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_PATH
    assert res_spinning.uart_flag_byte == 3

    # 3. Hết 3.0s xoay calib (now = 13.0s, dt = 3.0s >= 3.0s):
    # Dù có đường hay không, DỪNG HẲN: Gửi B4=0 để STM32 dừng xoay ngay lập tức!
    mock_path = create_mock_path_result(path_center_x=640, ref_y=500, offset_px=0.0)
    res_stop = mission.process(base_nav, [], frame_shape, path_res=mock_path, now=13.0)
    assert mission.state == QRTurnMissionState.COMPLETED_STOPPED, "Xoay xong 3s calib -> Phải chuyển sang COMPLETED_STOPPED!"
    assert res_stop.uart_flag_byte == 0, "Bắt buộc gửi B4=0 để STM32 ngắt cờ xoay ngay lập tức!"
    assert res_stop.lateral_offset_px == 0.0

    # 4. Sau 0.5s dừng triệt tiêu quán tính (now = 13.6s, dt_stop = 0.6s >= 0.5s):
    # Gửi B4=1 (Lệnh dừng tuyệt đối) với offset=0.0 để xe đứng yên hoàn toàn!
    res_hold = mission.process(base_nav, [], frame_shape, path_res=mock_path, now=13.6)
    assert mission.state == QRTurnMissionState.COMPLETED_STOPPED
    assert res_hold.uart_flag_byte == 1, "Sau 0.5s gửi B4=1 để STM32 khóa phanh đứng yên tuyệt đối"
    assert res_hold.lateral_offset_px == 0.0

    print("[PASS] test_calib_turn2_rotates_and_stops_completely: 3.0s rotation -> Stop completely verified!")


def test_idle_direct_turn2_station_trigger():
    """Kiểm tra tính năng kích hoạt trực tiếp trạm TURN_2 từ trạng thái IDLE:
    - Khi đặt robot trực tiếp trước mã TURN_2 (>= 200px) để test calib
    - Hệ thống chuyển ngay sang TURNING_LEFT_SEEK_PATH (B4=3)
    - Xoay đúng 3.0s rồi DỪNG HẲN HOÀN TOÀN (B4=0 rồi B4=1, offset=0.0)
    """
    frame_shape = (720, 1280)
    mission = QRTurnMissionManager(
        delay_turn2_sec=3.0,
        calib_turn2_stop_only=True
    )
    assert mission.state == QRTurnMissionState.IDLE
    base_nav = FusedNavigationTarget(None, 0.0, '', False, 'IDLE', None, (0, 0), 0.0, 0, False, 0)

    # Đưa mã TURN_2 >= 200px vào khung hình khi đang ở IDLE
    qr_turn2 = create_mock_qr(cx=640, cy=400, width=220, height=220, text="TURN_2")
    res = mission.process(base_nav, [qr_turn2], frame_shape, now=0.0)
    assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_PATH, "IDLE phải kích hoạt ngay trạm TURN_2"
    assert res.uart_flag_byte == 3

    # Đang xoay (now = 1.5s < 3.0s)
    res_mid = mission.process(base_nav, [], frame_shape, now=1.5)
    assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_PATH
    assert res_mid.uart_flag_byte == 3

    # Hết 3.0s (now = 3.0s) -> Chuyển sang COMPLETED_STOPPED, gửi B4=0 ngắt xoay
    res_end = mission.process(base_nav, [], frame_shape, now=3.0)
    assert mission.state == QRTurnMissionState.COMPLETED_STOPPED
    assert res_end.uart_flag_byte == 0, "Hết 3s phải gửi B4=0 để STM32 ngắt xoay"
    assert res_end.lateral_offset_px == 0.0

    # Sau 0.5s (now = 3.6s) -> Gửi B4=1 khóa phanh
    res_hold = mission.process(base_nav, [], frame_shape, now=3.6)
    assert res_hold.uart_flag_byte == 1, "Sau 0.5s gửi B4=1 để khóa phanh"
    assert res_hold.lateral_offset_px == 0.0

    print("[PASS] test_idle_direct_turn2_station_trigger: Direct TURN_2 trigger in IDLE -> 3.0s turn -> Stop completely verified!")


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
    print("=" * 60)
    print("ALL 15 NAVIGATION FUSION TESTS PASSED 100%!")
    print("=" * 60)

