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
    """Kiểm tra toàn bộ chu trình máy trạng thái rẽ QR phân tầng (QRTurnMissionManager):
    IDLE -> Đạt trạm TURN_1 (>=200px, Gửi B4=1) -> STM32 tự delay 1s & quay trái (B4=0)
    -> Bắt trúng tâm TURN_2 (<=40px, Gửi B4=2) -> Dừng 500ms triệt tiêu quán tính (B4=2)
    -> Tiến theo tâm TURN_2 (B4=0) -> Đạt trạm TURN_2 (>=200px, Gửi B4=1) -> DỪNG HOÀN TẤT -> RESET
    """
    frame_shape = (720, 1280)  # cam_cx = 640
    mission = QRTurnMissionManager(
        stop_duration_sec=1.0,
        pause_center_sec=0.5,
        align_tolerance_px=40.0,
        min_stop_qr_dim=200.0
    )

    # 1. Ban đầu: Trạng thái IDLE
    assert mission.state == QRTurnMissionState.IDLE
    path_res = create_mock_path_result(path_center_x=640, ref_y=500, offset_px=0.0)
    base_nav = fuse_navigation_target(path_res=path_res, qr_results=[], frame_shape=frame_shape)

    # 2. Tiếp cận TURN_1 nhưng kích thước còn nhỏ (< 200px, vd 120x120)
    qr_turn1_small = create_mock_qr(cx=640, cy=400, width=120, height=120, text="TURN_1")
    base_nav = fuse_navigation_target(path_res=path_res, qr_results=[qr_turn1_small], frame_shape=frame_shape)
    nav_out = mission.process(base_nav, [qr_turn1_small], frame_shape, now=1.0)
    assert mission.state == QRTurnMissionState.IDLE
    assert nav_out.uart_flag_byte == 0, "Chưa dừng khi TURN_1 < 200px"

    # 3. Tiếp cận TURN_1 kích thước lớn (>= 200px, vd 210x210): Phát hiện trạm TURN_1!
    qr_turn1_large = create_mock_qr(cx=640, cy=400, width=210, height=210, text="TURN_1")
    base_nav = fuse_navigation_target(path_res=path_res, qr_results=[qr_turn1_large], frame_shape=frame_shape)
    nav_out = mission.process(base_nav, [qr_turn1_large], frame_shape, now=2.0)
    assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_TURN2
    assert nav_out.uart_flag_byte == 1, "Kích hoạt trạm TURN_1: Gửi Byte 4 = 1 để STM32 tự delay 1s và quay trái"
    assert nav_out.qr_stop_triggered is True

    # 4. Robot đang quay trái trên STM32: Camera chưa thấy TURN_2 (frame rỗng)
    nav_out = mission.process(base_nav, [], frame_shape, now=3.0)
    assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_TURN2
    assert nav_out.uart_flag_byte == 0, "Chưa thấy TURN_2 -> Gửi Byte 4 = 0 để STM32 tiếp tục quay trái"

    # 5. Camera bắt đầu thấy TURN_2 nhưng chưa vào tâm (cx = 780, offset = +140px > 40px)
    qr_turn2_far = create_mock_qr(cx=780, cy=360, width=150, height=150, text="TURN_2")
    nav_out = mission.process(base_nav, [qr_turn2_far], frame_shape, now=3.5)
    assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_TURN2
    assert nav_out.uart_flag_byte == 0, "Thấy TURN_2 nhưng chưa vào tâm -> Vẫn gửi Byte 4 = 0 để STM32 quay tiếp"
    assert abs(nav_out.lateral_offset_px - 140.0) < 1e-3

    # 6. TURN_2 đã vào đúng tâm camera (cx = 655, offset = +15px <= 40px)
    # Frame 1: Bắt đầu xác nhận tâm TURN_2 (chống nhiễu chớp tắt)
    qr_turn2_centered = create_mock_qr(cx=655, cy=360, width=165, height=165, text="TURN_2")
    nav_out = mission.process(base_nav, [qr_turn2_centered], frame_shape, now=10.0)
    assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_TURN2
    assert nav_out.uart_flag_byte == 0, "Frame 1: Đang xác nhận tâm (chống nhiễu), xe tiếp tục quay trái (B4=0)"
    assert mission.consecutive_turn2_frames == 1

    # Frame 2: Xác nhận liên tiếp đủ confirm_frames=2 -> CHÍNH THỨC BẮT TÂM & DỪNG 500MS!
    nav_out = mission.process(base_nav, [qr_turn2_centered], frame_shape, now=10.05)
    assert mission.state == QRTurnMissionState.APPROACHING_TURN2
    assert nav_out.uart_flag_byte == 2, "Bắt trúng tâm TURN_2 đủ 2 frames -> Gửi Byte 4 = 2 để STM32 dừng 500ms"
    assert abs(nav_out.lateral_offset_px - 15.0) < 1e-3

    # 7. Trong khoảng 500ms dừng triệt tiêu quán tính (now = 10.2s, dt = 200ms < 500ms)
    nav_out = mission.process(base_nav, [qr_turn2_centered], frame_shape, now=10.2)
    assert mission.state == QRTurnMissionState.APPROACHING_TURN2
    assert nav_out.uart_flag_byte == 2, "Vẫn trong khoảng 500ms dừng quán tính -> Gửi Byte 4 = 2"

    # 8. Hết 500ms (now = 10.6s, dt = 600ms >= 500ms): Bắt đầu tiến theo tâm TURN_2!
    nav_out = mission.process(base_nav, [qr_turn2_centered], frame_shape, now=10.6)
    assert mission.state == QRTurnMissionState.APPROACHING_TURN2
    assert nav_out.uart_flag_byte == 0, "Hết 500ms -> Gửi Byte 4 = 0 để STM32 tiến lên theo tâm TURN_2 qua PID"
    assert abs(nav_out.lateral_offset_px - 15.0) < 1e-3

    # 9. Tiếp cận TURN_2 và kích thước đạt trạm dừng (>= 200px, vd 210x210):
    # KÍCH HOẠT BƯỚC MỚI: DỪNG 500MS VÀ XOAY TRÁI TÌM ĐƯỜNG (Byte 4 = 3)!
    qr_turn2_reached = create_mock_qr(cx=645, cy=400, width=210, height=210, text="TURN_2")
    nav_out = mission.process(base_nav, [qr_turn2_reached], frame_shape, now=12.0)
    assert mission.state == QRTurnMissionState.TURNING_RIGHT_SEEK_PATH
    assert nav_out.uart_flag_byte == 3, "Đã đến trạm TURN_2 (>=200px) -> Gửi Byte 4 = 3 để STM32 dừng 500ms rồi xoay trái tìm đường"

    # 10. Trong 500ms dừng triệt tiêu quán tính tại TURN_2 (now = 12.3s, dt = 300ms < 500ms)
    nav_out = mission.process(base_nav, [], frame_shape, now=12.3)
    assert mission.state == QRTurnMissionState.TURNING_RIGHT_SEEK_PATH
    assert nav_out.uart_flag_byte == 3, "Đang trong 500ms dừng tại trạm TURN_2 -> Gửi Byte 4 = 3"

    # 11. Hết 500ms: STM32 xoay trái tìm đường, camera chưa thấy đường (now = 12.8s)
    nav_out = mission.process(base_nav, [], frame_shape, now=12.8)
    assert mission.state == QRTurnMissionState.TURNING_RIGHT_SEEK_PATH
    assert nav_out.uart_flag_byte == 3, "Đang xoay trái tìm đường -> Tiếp tục gửi Byte 4 = 3"

    # 12. Frame 1: Camera bắt đầu phát hiện lối đi mới (now = 13.0s, frame 1)
    path_new = create_mock_path_result(path_center_x=650, ref_y=500, offset_px=10.0)
    base_nav_new = fuse_navigation_target(path_res=path_new, qr_results=[], frame_shape=frame_shape)
    nav_out = mission.process(base_nav_new, [], frame_shape, path_res=path_new, now=13.0)
    assert mission.state == QRTurnMissionState.TURNING_RIGHT_SEEK_PATH
    assert nav_out.uart_flag_byte == 3, "Khung hình 1 phát hiện đường -> Đang xác nhận ổn định (chống nhiễu)"

    # 13. Frame 2: Xác nhận phát hiện luống đường ổn định (now = 13.1s, frame 2)
    # -> TỰ ĐỘNG RESET VỀ IDLE & BẮT ĐẦU CHẠY BÁM LUỐNG MỚI (Byte 4 = 0, STM32 tự dừng 500ms rồi bám luống)!
    nav_out = mission.process(base_nav_new, [], frame_shape, path_res=path_new, now=13.1)
    assert mission.state == QRTurnMissionState.IDLE, "Đã thấy luống -> Tự động reset về IDLE để bám luống mới chạy tiếp!"
    assert nav_out.uart_flag_byte == 0, "Báo STM32 chuyển sang bám luống mới (Byte 4 = 0)"
    assert nav_out.mode_name == "PATH_LINE"
    assert nav_out.qr_stop_triggered is False
    assert abs(nav_out.lateral_offset_px - 10.0) < 1e-3

    # 14. Kiểm tra xe tiếp tục chạy bám luống mới ở IDLE (Chu trình 8 bước khép kín và quay lại bước 1)
    nav_out = mission.process(base_nav_new, [], frame_shape, path_res=path_new, now=14.0)
    assert mission.state == QRTurnMissionState.IDLE
    assert nav_out.uart_flag_byte == 0, "Duy trì bám luống mới (Byte 4 = 0)"
    assert nav_out.mode_name == "PATH_LINE"

    # 15. Test Reset thủ công (khi người dùng bấm Reset trên UI hoặc gạt switch)
    mission.reset()
    assert mission.state == QRTurnMissionState.IDLE
    base_nav_cleared = fuse_navigation_target(path_res=path_res, qr_results=[], frame_shape=frame_shape)
    nav_out = mission.process(base_nav_cleared, [], frame_shape, now=16.0)
    assert nav_out.uart_flag_byte == 0

    print("[PASS] test_qr_turn_mission_lifecycle: Full QR turn mission cycle (IDLE -> TURN_1 B4=1 -> SEEK TURN_2 B4=0 -> CENTERED TURN_2 B4=2 -> PAUSE 500ms B4=2 -> APPROACH B4=0 -> REACH TURN_2 >=200px B4=3 -> PAUSE 500ms B4=3 -> ROTATE LEFT B4=3 -> DETECT PATH B4=0 -> LOOP BACK TO IDLE TRACKING NEW PATH) 100% verified")


def test_method_3_path_seek_with_offset_and_heading_angle():
    """Kiểm tra Cách 3: Điều kiện chuyển bước bắt luống mới yêu cầu ĐỒNG THỜI:
    1) |lateral_offset_px| <= 200px
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
        path_seek_tolerance_px=200.0,
        path_seek_max_angle_deg=20.0
    )

    # Đưa mission vào trạng thái TURNING_RIGHT_SEEK_PATH (đang xoay trái tìm đường)
    mission.state = QRTurnMissionState.TURNING_RIGHT_SEEK_PATH
    mission.turn_right_start_time = 100.0

    # 1. Luống lệch tâm quá xa (|offset| = 280px > 200px) dù góc thẳng (angle = 0)
    path_far_offset = create_mock_path_result(path_center_x=920, ref_y=500, offset_px=280.0)
    base_nav = fuse_navigation_target(path_res=path_far_offset, qr_results=[], frame_shape=frame_shape)
    nav_out = mission.process(base_nav, [], frame_shape, path_res=path_far_offset, now=101.0)
    assert mission.state == QRTurnMissionState.TURNING_RIGHT_SEEK_PATH, "Lệch tâm > 200px -> Phải tiếp tục xoay trái tìm luống"
    assert nav_out.uart_flag_byte == 3

    # 2. Luống ở gần tâm (|offset| = 80px <= 200px) nhưng góc nghiêng quá chéo (|angle| = ~35 độ > 20 độ)
    path_tilted = create_mock_path_result(path_center_x=720, ref_y=500, offset_px=80.0)
    path_tilted.centerline_points = [(860, 200), (790, 300), (720, 400)]
    angle_calc = calculate_path_heading_angle(path_tilted)
    assert angle_calc is not None and abs(angle_calc) > 20.0, f"Angle {angle_calc} should be > 20 deg"
    
    base_nav = fuse_navigation_target(path_res=path_tilted, qr_results=[], frame_shape=frame_shape)
    nav_out = mission.process(base_nav, [], frame_shape, path_res=path_tilted, now=101.5)
    assert mission.state == QRTurnMissionState.TURNING_RIGHT_SEEK_PATH, "Góc chéo > 20 độ -> Phải tiếp tục xoay trái để nắn thẳng"
    assert nav_out.uart_flag_byte == 3

    # 3. Luống thỏa mãn CẢ HAI (|offset| = 60px <= 200px VÀ |angle| = ~5.7 độ <= 20 độ)
    path_aligned = create_mock_path_result(path_center_x=700, ref_y=500, offset_px=60.0)
    path_aligned.centerline_points = [(720, 200), (710, 300), (700, 400)]
    angle_aligned = calculate_path_heading_angle(path_aligned)
    assert angle_aligned is not None and abs(angle_aligned) <= 20.0

    base_nav = fuse_navigation_target(path_res=path_aligned, qr_results=[], frame_shape=frame_shape)
    # Frame 1: Nhận diện thỏa mãn
    nav_out = mission.process(base_nav, [], frame_shape, path_res=path_aligned, now=102.0)
    assert mission.state == QRTurnMissionState.TURNING_RIGHT_SEEK_PATH
    assert nav_out.uart_flag_byte == 3
    # Frame 2: Xác nhận ổn định -> Chuyển sang IDLE và phát Byte 4 = 0!
    nav_out = mission.process(base_nav, [], frame_shape, path_res=path_aligned, now=102.1)
    assert mission.state == QRTurnMissionState.IDLE, "Thỏa mãn cả offset <= 200px và angle <= 20 độ -> Chuyển sang bám luống!"
    assert nav_out.uart_flag_byte == 0, "Gửi Byte 4 = 0 để STM32 phanh 500ms rồi bám luống chạy tiếp"

    print("[PASS] test_method_3_path_seek_with_offset_and_heading_angle: Method 3 (Dual condition: |offset|<=200px AND |angle|<=20deg) 100% verified!")


def test_turn2_detection_during_rotation_stops_and_approaches():
    """Kiểm tra trường hợp thực tế khi đang xoay tìm TURN_2:
    - YOLO phát hiện QR box ở vị trí cx=516 (lệch -124px trên khung 1280x720)
    - Do xe đang xoay, text QR chưa giải mã hoặc text cũ là TURN_1
    - Hệ thống vẫn nhận diện đúng TURN_2, thỏa mãn align_tolerance_px=180px
    - Gửi Byte 4 = 2 (dừng 500ms) rồi chuyển sang tiến theo tâm với Byte 4 = 0
    """
    frame_shape = (720, 1280) # cam_cx = 640
    mission = QRTurnMissionManager(align_tolerance_px=180.0)
    mission.state = QRTurnMissionState.TURNING_LEFT_SEEK_TURN2

    # Giả lập box phát hiện từ YOLO khi xe đang quay nhòe
    pts = np.array([[492.0, 353.0], [540.0, 353.0], [540.0, 402.0], [492.0, 402.0]], dtype=np.float32)
    qr_ai = QRDetectionResult(
        points=pts,
        text="TURN_2",
        is_decoded=True,
        backend="yolo-fastestv2-qr",
        center=(516.0, 377.5),
        center_int=(516, 377),
        width_px=48.0,
        height_px=49.0,
        area_px=2352.0,
        aspect_ratio=0.98,
        orientation_deg=0.0
    )

    base_nav = FusedNavigationTarget(None, 0.0, '', False, 'IDLE', None, (0, 0), 0.0, 0, False, 0)

    # 1. Khung hình 1: Nhìn thấy trong tâm (lệch -124px <= 180px) -> Đang đếm xác nhận (Frame 1/2, B4=0)
    res0 = mission.process(base_nav, [qr_ai], frame_shape, now=10.0)
    assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_TURN2
    assert res0.uart_flag_byte == 0, f"Expected B4=0 on frame 1 debounce, got {res0.uart_flag_byte}"
    assert mission.consecutive_turn2_frames == 1

    # Khung hình 2: Đủ 2 frame liên tiếp -> Chuyển sang APPROACHING_TURN2 và gửi B4 = 2
    res1 = mission.process(base_nav, [qr_ai], frame_shape, now=10.04)
    assert mission.state == QRTurnMissionState.APPROACHING_TURN2
    assert res1.uart_flag_byte == 2, f"Expected B4=2 (dừng 500ms triệt tiêu quán tính quay), got {res1.uart_flag_byte}"
    assert abs(res1.lateral_offset_px - (-124.0)) < 1e-3

    # 2. Trong 500ms dừng (now = 10.3s): Duy trì B4 = 2
    res2 = mission.process(base_nav, [qr_ai], frame_shape, now=10.3)
    assert res2.uart_flag_byte == 2

    # 3. Hết 500ms dừng (now = 10.7s): Gửi B4 = 0 để xe tiến theo tâm TURN_2 qua PID
    res3 = mission.process(base_nav, [qr_ai], frame_shape, now=10.7)
    assert mission.state == QRTurnMissionState.APPROACHING_TURN2
    assert res3.uart_flag_byte == 0, f"Expected B4=0 (tiến theo tâm TURN_2), got {res3.uart_flag_byte}"
    assert abs(res3.lateral_offset_px - (-124.0)) < 1e-3

    print("[PASS] test_turn2_detection_during_rotation_stops_and_approaches: Detection -> Stop 500ms (B4=2) -> Approach Center (B4=0) 100% verified!")


def test_turn_delays_and_confirm_frames():
    """Kiểm tra các thông số tùy chỉnh mới:
    1. delay_turn1_sec: Trong thời gian trễ bắt đầu xoay tại Turn 1, gửi B4=1, offset=0, bỏ qua mọi nhiễu.
    2. delay_turn2_sec: Trong thời gian trễ bắt đầu xoay tại Turn 2, gửi B4=3, offset=0, bỏ qua mọi nhiễu đường.
    3. confirm_frames: Yêu cầu đủ N frames liên tiếp mới chốt khóa tâm Turn 2 hoặc khóa luống đường mới.
    """
    frame_shape = (720, 1280)
    mission = QRTurnMissionManager(
        delay_turn1_sec=1.5,
        delay_turn2_sec=1.2,
        confirm_frames=3,
        align_tolerance_px=100.0,
        min_stop_qr_dim=200.0
    )

    base_nav = FusedNavigationTarget(None, 0.0, '', False, 'IDLE', None, (0, 0), 0.0, 0, False, 0)

    # --- KỊCH BẢN 1: TRỄ TẠI TURN 1 ---
    qr_turn1 = create_mock_qr(cx=640, cy=400, width=220, height=220, text="TURN_1")
    res1 = mission.process(base_nav, [qr_turn1], frame_shape, now=1.0)
    assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_TURN2
    assert res1.uart_flag_byte == 1

    # Trong 1.5s trễ đầu (now = 1.8s, dt = 0.8s < 1.5s):
    # Dù có nhiễu giả lập TURN_2 lọt vào khung hình, vẫn phải bỏ qua và giữ B4=1, offset=0.0!
    fake_turn2 = create_mock_qr(cx=640, cy=360, width=150, height=150, text="TURN_2")
    res_blind = mission.process(base_nav, [fake_turn2], frame_shape, now=1.8)
    assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_TURN2
    assert res_blind.uart_flag_byte == 1, "Trong 1.5s trễ Turn 1 -> Duy trì B4=1, không bắt Turn 2"
    assert res_blind.lateral_offset_px == 0.0

    # Hết 1.5s trễ (now = 2.6s, dt = 1.6s >= 1.5s): Bắt đầu tìm kiếm Turn 2
    # Với confirm_frames = 3: Cần đúng 3 frames liên tiếp
    # Frame 1:
    res_f1 = mission.process(base_nav, [fake_turn2], frame_shape, now=2.6)
    assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_TURN2
    assert res_f1.uart_flag_byte == 0
    assert mission.consecutive_turn2_frames == 1

    # Giả lập frame 2 bị mất dấu do chớp nháy:
    res_miss = mission.process(base_nav, [], frame_shape, now=2.64)
    assert mission.consecutive_turn2_frames == 0, "Mất dấu phải reset counter về 0"
    assert res_miss.uart_flag_byte == 0

    # Lại thấy liên tiếp 3 frames:
    res_c1 = mission.process(base_nav, [fake_turn2], frame_shape, now=2.68)
    assert mission.consecutive_turn2_frames == 1
    assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_TURN2

    res_c2 = mission.process(base_nav, [fake_turn2], frame_shape, now=2.72)
    assert mission.consecutive_turn2_frames == 2
    assert mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_TURN2

    res_c3 = mission.process(base_nav, [fake_turn2], frame_shape, now=2.76)
    assert mission.consecutive_turn2_frames == 3
    assert mission.state == QRTurnMissionState.APPROACHING_TURN2, "Đủ 3 frames liên tiếp -> Khóa tâm thành công!"
    assert res_c3.uart_flag_byte == 2

    # --- KỊCH BẢN 2: TRỄ TẠI TURN 2 ---
    # Tiến tới trạm Turn 2 (>= 200px)
    qr_turn2_station = create_mock_qr(cx=640, cy=400, width=210, height=210, text="TURN_2")
    res_reach = mission.process(base_nav, [qr_turn2_station], frame_shape, now=4.0)
    assert mission.state == QRTurnMissionState.TURNING_RIGHT_SEEK_PATH
    assert res_reach.uart_flag_byte == 3

    # Trong 1.2s trễ tại Turn 2 (now = 4.8s, dt = 0.8s < 1.2s):
    # Dù có đường (nhiễu rác bên mép trạm) thì KHÔNG được bắt để tránh ủi bậy!
    path_noise = create_mock_path_result(path_center_x=640, ref_y=500, offset_px=0.0)
    res_t2_blind = mission.process(base_nav, [], frame_shape, path_res=path_noise, now=4.8)
    assert mission.state == QRTurnMissionState.TURNING_RIGHT_SEEK_PATH
    assert res_t2_blind.uart_flag_byte == 3, "Trong 1.2s trễ Turn 2 -> Duy trì B4=3, không bắt đường tránh ủi"
    assert mission.consecutive_path_frames == 0

    # Hết 1.2s (now = 5.3s, dt = 1.3s >= 1.2s): Bắt đầu kiểm tra confirm 3 frames đường
    for i in range(1, 3):
        res_p = mission.process(base_nav, [], frame_shape, path_res=path_noise, now=5.3 + i*0.04)
        assert mission.state == QRTurnMissionState.TURNING_RIGHT_SEEK_PATH
        assert mission.consecutive_path_frames == i
        assert res_p.uart_flag_byte == 3

    # Frame thứ 3: Hoàn tất bắt đường mới!
    res_p3 = mission.process(base_nav, [], frame_shape, path_res=path_noise, now=5.42)
    assert mission.state == QRTurnMissionState.IDLE, "Đủ 3 frames đường -> Bắt đầu bám luống mới thành công!"
    assert res_p3.uart_flag_byte == 0

    print("[PASS] test_turn_delays_and_confirm_frames: Delay Turn 1 (1.5s), Delay Turn 2 (1.2s), Confirm Frames (3 frames) 100% verified!")


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
    print("=" * 60)
    print("ALL 13 NAVIGATION FUSION TESTS PASSED 100%!")
    print("=" * 60)

