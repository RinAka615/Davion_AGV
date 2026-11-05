"""
Module for sensor fusion between Vision Line Tracking (YOLOv8 Segmentation)
and QR Code Checkpoint Alignment.

When QR code bounding area exceeds threshold (default 60x60 px),
navigation seamlessly switches from path centerline to QR center alignment.
"""

from dataclasses import dataclass
from enum import Enum
import math
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np

from qr_scanner.detector import QRDetectionResult
from src.path_detector import DetectionResult, TrackingState


class QRTurnMissionState(str, Enum):
    TRACKING_ROW_1 = "TRACKING_ROW_1"               # 1. Chạy thẳng bám luống 1
    TRACKING_QR1 = "TRACKING_QR1"                   # 2. Gặp QR1 (>= 30px): Bám theo tâm QR1
    STOP_AT_QR1 = "STOP_AT_QR1"                     # 3. Đến trạm QR1 (>= 200px): Dừng xe 1.0s (B4=1)
    BLIND_TURN_RIGHT = "BLIND_TURN_RIGHT"           # 4. Xoay phải mù 2.0s (B4=2)
    PAUSE_AFTER_TURN1 = "PAUSE_AFTER_TURN1"         # 5. Dừng dập quán tính 500ms (B4=1)
    TRACKING_QR2 = "TRACKING_QR2"                   # 6. Bắt đầu chạy theo tâm turn2 (QR2) (B4=0)
    STOP_AT_QR2 = "STOP_AT_QR2"                     # 7. Đến trạm QR2 (>= 200px): Dừng xe 1.0s (B4=1)
    BLIND_TURN_LEFT = "BLIND_TURN_LEFT"             # 8. Xoay trái mù 2.0s (B4=3)
    PAUSE_AFTER_TURN2 = "PAUSE_AFTER_TURN2"         # 9. Dừng dập quán tính 500ms (B4=1)
    COMPLETED_STOPPED = "COMPLETED_STOPPED"         # 10. Dừng xe Calib (nếu bật chk_calib_turn2)

    # Aliases tương thích ngược
    IDLE = "TRACKING_ROW_1"
    STOP_AT_TURN1 = "STOP_AT_QR1"
    TURNING_RIGHT_SEEK_PATH = "BLIND_TURN_RIGHT"
    TRACKING_ROW_2 = "TRACKING_QR2"
    TURNING_LEFT_SEEK_PATH = "BLIND_TURN_LEFT"
    APPROACHING_TURN2 = "TRACKING_QR2"
    TURNING_LEFT_SEEK_TURN2 = "BLIND_TURN_RIGHT"


@dataclass
class FusedNavigationTarget:
    """Kết quả mục tiêu dẫn hướng tích hợp (Line Path vs QR Center)."""
    target_pt: Optional[Tuple[int, int]]       # Tọa độ pixel (tx, ty) mục tiêu lái
    lateral_offset_px: Optional[float]        # Độ lệch ngang so với trục giữa camera (px)
    offset_direction: str                     # Chuỗi mô tả hướng và độ lệch ('Lệch trái ...', 'Đúng tâm ...')
    is_qr_mode: bool                          # True nếu đang bám theo tâm QR; False nếu bám theo lối đi
    mode_name: str                            # 'QR_CENTER' hoặc 'PATH_LINE' hoặc 'LOST'
    active_qr: Optional[QRDetectionResult] = None  # Đối tượng mã QR đang được dùng làm mục tiêu
    qr_bbox_dim: Tuple[float, float] = (0.0, 0.0)  # (width, height) của đường bao QR
    qr_effective_area: float = 0.0            # Diện tích đường bao của QR (px^2)
    ref_y: Optional[int] = None               # Chiều cao y tham chiếu trong khung hình
    qr_stop_triggered: bool = False           # True nếu diện tích đường bao QR > 200x200 (kích hoạt trạm dừng)
    uart_flag_byte: int = 0                   # Byte 4 gửi STM32: 0: Chạy, 1: Dừng, 2: Rẽ trái
    mission_state: str = "IDLE"               # Trạng thái nhiệm vụ chuyển hướng TURN1 -> TURN2
    mission_status_desc: str = ""             # Mô tả chi tiết trạng thái nhiệm vụ cho HUD / UI


def check_qr_exceeds_threshold(
    qr: QRDetectionResult,
    min_dim: float = 100.0,
    min_area: Optional[float] = None
) -> Tuple[bool, float, float, float]:
    """
    Kiểm tra xem diện tích đường bao của mã QR có lớn hơn ngưỡng (mặc định > 100x100 = 10,000 px^2) không.
    
    :param qr: Kết quả phát hiện mã QR
    :param min_dim: Kích thước chiều dài/rộng tối thiểu (mặc định 100.0 px)
    :param min_area: Diện tích tối thiểu (nếu None, tính bằng min_dim * min_dim = 10,000 px^2)
    :return: (thỏa_mãn, chiều_rộng_đường_bao, chiều_cao_đường_bao, diện_tích_hiệu_dụng)
    """
    if min_area is None:
        min_area = min_dim * min_dim

    # 1. Tính kích thước bounding box từ 4 đỉnh góc
    if qr.points is not None and len(qr.points) >= 4:
        bw = float(np.max(qr.points[:, 0]) - np.min(qr.points[:, 0]))
        bh = float(np.max(qr.points[:, 1]) - np.min(qr.points[:, 1]))
    else:
        bw = float(qr.width_px)
        bh = float(qr.height_px)

    bbox_area = bw * bh
    # Diện tích hiệu dụng là giá trị lớn nhất giữa diện tích bbox, diện tích đa giác thực tế, và width x height
    effective_area = max(bbox_area, float(getattr(qr, "area_px", 0.0)), float(qr.width_px * qr.height_px))

    # Điều kiện thỏa mãn diện tích đường bao > 100x100:
    # 1) Diện tích >= 10,000 px^2
    # 2) HOẶC cả hai chiều rộng và cao của bbox >= 100 px
    # 3) HOẶC kích thước đo đạc width_px >= 100 và height_px >= 100
    is_exceeded = (
        effective_area >= min_area
        or (bw >= min_dim and bh >= min_dim)
        or (qr.width_px >= min_dim and qr.height_px >= min_dim)
    )

    return is_exceeded, round(bw, 1), round(bh, 1), round(effective_area, 1)


def calculate_path_heading_angle(path_res: Optional[DetectionResult]) -> Optional[float]:
    """
    Tính góc hướng (heading angle) của luống đường so với trục dọc camera (đơn vị: độ).
    0.0 độ : Luống thẳng đứng (robot nhìn thẳng dọc theo luống).
    > 0 độ : Luống nghiêng sang phải về phía xa.
    < 0 độ : Luống nghiêng sang trái về phía xa.
    None   : Không xác định được (mất đường).
    """
    if path_res is None or path_res.state == TrackingState.LOST:
        return None

    # 1. Ưu tiên đường tâm hồi quy (centerline_points): Lấy điểm đầu (xa) và điểm cuối (gần)
    if path_res.centerline_points and len(path_res.centerline_points) >= 2:
        top_pt = path_res.centerline_points[0]      # y nhỏ nhất (ở xa phía trước)
        bot_pt = path_res.centerline_points[-1]     # y lớn nhất (ở gần trước mũi xe)
        dx = float(top_pt[0] - bot_pt[0])
        dy = float(bot_pt[1] - top_pt[1])          # dy > 0
        if dy > 5.0:
            return float(math.degrees(math.atan2(dx, dy)))

    # 2. Nếu có poly_coeffs: x = a*y + b hoặc x = a*y^2 + b*y + c
    if path_res.poly_coeffs is not None and len(path_res.poly_coeffs) >= 2:
        coeffs = path_res.poly_coeffs
        if len(coeffs) == 2:
            dx_dy = float(coeffs[0])
        else:
            y_eval = float(path_res.ref_y_global) if path_res.ref_y_global is not None else 400.0
            dx_dy = float(2.0 * coeffs[0] * y_eval + coeffs[1])
        # Hướng tiến là y giảm (dy < 0), nên góc = atan2(dx, -dy) = atan(-dx_dy)
        return float(math.degrees(math.atan(-dx_dy)))

    # 3. Quét các vạch scanline hợp lệ
    valid_scans = [s for s in getattr(path_res, "scanlines", []) if getattr(s, "is_valid", False)]
    if len(valid_scans) >= 2:
        top_s = min(valid_scans, key=lambda s: s.y_global)
        bot_s = max(valid_scans, key=lambda s: s.y_global)
        dx = float(top_s.x_mid - bot_s.x_mid)
        dy = float(bot_s.y_global - top_s.y_global)
        if dy > 5.0:
            return float(math.degrees(math.atan2(dx, dy)))

    return 0.0


def fuse_navigation_target(
    path_res: Optional[DetectionResult],
    qr_results: List[QRDetectionResult],
    frame_shape: Tuple[int, int],
    min_qr_size: float = 30.0,
    qr_stop_thresh: float = 200.0,
    enable_qr_override: bool = True
) -> FusedNavigationTarget:
    """
    Thuật toán hợp nhất điều hướng thông minh & cờ trạm dừng UART:
    - Cờ Byte 4 gửi STM32 (uart_flag_byte):
      + Nếu phát hiện mã QR có kích thước đường bao >= qr_stop_thresh (mặc định 200px):
        Gửi cờ = 1 (Dừng trạm) xuống STM32.
      + Ngược lại: Gửi cờ = 0 (Chạy bình thường).
    - Sai số lái (Bytes 0..3 - lateral_offset_px):
      + Nếu phát hiện mã QR có kích thước đường bao >= min_qr_size (mặc định >= 30px):
        Ưu tiên TUYỆT ĐỐI tính toán sai số (lateral offset) và hướng lái dựa theo TÂM CỦA QR.
      + Ngược lại:
        Duy trì tính toán sai số theo TÂM LỐI ĐI (Path Centerline) như bình thường.
    """
    h_img, w_img = frame_shape[:2]
    cam_cx = w_img // 2

    # 1. Kiểm tra điều kiện kích hoạt trạm dừng UART (QR >= 200px)
    qr_stop_triggered = False
    if qr_results:
        for qr in qr_results:
            exceeded_stop, _, _, _ = check_qr_exceeds_threshold(qr, min_dim=qr_stop_thresh)
            if exceeded_stop:
                qr_stop_triggered = True
                break
    uart_flag_byte = 1 if qr_stop_triggered else 0

    # 2. Tìm kiếm mã QR lớn nhất thỏa mãn điều kiện đường bao >= min_qr_size (mặc định >= 30px)
    selected_qr: Optional[QRDetectionResult] = None
    best_area = 0.0
    best_bw, best_bh = 0.0, 0.0

    if enable_qr_override and qr_results:
        for qr in qr_results:
            exceeded, bw, bh, area = check_qr_exceeds_threshold(qr, min_dim=min_qr_size)
            if exceeded and area > best_area:
                selected_qr = qr
                best_area = area
                best_bw, best_bh = bw, bh

    # 3. TRƯỜNG HỢP 1: BÁM THEO TÂM MÃ QR (QR CENTER GUIDANCE)
    if selected_qr is not None:
        # Lấy tọa độ tâm QR (đã được tính bằng giao điểm xạ ảnh chính xác)
        if hasattr(selected_qr, "center") and selected_qr.center and (selected_qr.center[0] != 0.0 or selected_qr.center[1] != 0.0):
            cx, cy = selected_qr.center
        else:
            cx = float(np.mean(selected_qr.points[:, 0]))
            cy = float(np.mean(selected_qr.points[:, 1]))

        tx = int(round(cx))
        ty = int(round(cy))
        target_pt = (tx, ty)

        # Tính toán sai số theo trục giữa của camera
        lateral_offset = float(tx - cam_cx)
        lateral_offset = float(np.clip(lateral_offset, -w_img / 2.0, w_img / 2.0))

        if abs(lateral_offset) <= 3.0:
            offset_dir = "Đúng tâm mã QR (0 px)"
        elif lateral_offset < -3.0:
            offset_dir = f"Lệch trái {abs(lateral_offset):.1f} px (Theo tâm QR)"
        else:
            offset_dir = f"Lệch phải {lateral_offset:.1f} px (Theo tâm QR)"

        return FusedNavigationTarget(
            target_pt=target_pt,
            lateral_offset_px=lateral_offset,
            offset_direction=offset_dir,
            is_qr_mode=True,
            mode_name="QR_CENTER",
            active_qr=selected_qr,
            qr_bbox_dim=(best_bw, best_bh),
            qr_effective_area=best_area,
            ref_y=ty,
            qr_stop_triggered=qr_stop_triggered,
            uart_flag_byte=uart_flag_byte
        )

    # 4. TRƯỜNG HỢP 2: BÁM THEO LỐI ĐI NHÀ MÀNG (PATH GUIDANCE)
    if path_res is not None and path_res.state != TrackingState.LOST:
        target_pt = path_res.path_center or path_res.box_center
        return FusedNavigationTarget(
            target_pt=target_pt,
            lateral_offset_px=path_res.lateral_offset_px,
            offset_direction=path_res.offset_direction,
            is_qr_mode=False,
            mode_name="PATH_LINE",
            active_qr=None,
            ref_y=path_res.ref_y_global,
            qr_stop_triggered=qr_stop_triggered,
            uart_flag_byte=uart_flag_byte
        )

    # 5. TRƯỜNG HỢP 3: MẤT CẢ LỐI ĐI VÀ KHÔNG CÓ QR
    return FusedNavigationTarget(
        target_pt=None,
        lateral_offset_px=None,
        offset_direction="N/A (Chờ tín hiệu)",
        is_qr_mode=False,
        mode_name="LOST",
        active_qr=None,
        qr_stop_triggered=qr_stop_triggered,
        uart_flag_byte=uart_flag_byte
    )


class QRTurnMissionManager:
    """Quản lý nhiệm vụ chuyển hướng tự động giữa 2 mã QR:
    1. Chạy thẳng bám luống 1 (TRACKING_ROW_1 / IDLE).
    2. Đến khi gặp QR1 (>= min_qr_nav_dim, mặc định 30px): Bám theo tâm QR1 (TRACKING_QR1).
    3. Chạy theo tâm QR1 cho đến khi >= min_stop_qr_dim (mặc định 200px):
       -> STOP_AT_QR1: Dừng xe stop_delay_t1_sec (mặc định 1.0s, B4=1).
    4. Hết 1.0s dừng:
       -> BLIND_TURN_RIGHT: Robot xoay phải mù blind_turn1_sec (mặc định 2.0s, B4=2).
    5. Hết 2.0s xoay phải mù:
       -> PAUSE_AFTER_TURN1: Dừng dập quán tính pause_turn1_sec (mặc định 500ms = 0.5s, B4=1).
    6. Hết 500ms dừng:
       -> TRACKING_QR2: Bắt đầu chạy theo tâm turn2 (QR2) (B4=0).
    7. Đến khi QR2 >= min_stop_qr_dim (mặc định 200px):
       -> STOP_AT_QR2: Dừng xe stop_delay_t2_sec (mặc định 1.0s, B4=1).
    8. Hết 1.0s dừng:
       -> BLIND_TURN_LEFT: Xoay trái mù blind_turn2_sec (mặc định 2.0s, B4=3).
    9. Hết 2.0s xoay trái mù:
       -> PAUSE_AFTER_TURN2: Dừng dập quán tính pause_turn2_sec (mặc định 500ms = 0.5s, B4=1).
    10. Hết 500ms dừng:
       -> Chạy thẳng tiếp (reset về TRACKING_ROW_1, B4=0)!
          (Nếu bật calib_turn2_stop_only thì dừng hẳn COMPLETED_STOPPED để kiểm tra).
    """

    def __init__(
        self,
        min_qr_nav_dim: float = 30.0,
        min_stop_qr_dim: float = 200.0,
        stop_delay_t1_sec: float = 1.0,
        blind_turn1_sec: float = 2.0,
        pause_turn1_sec: float = 0.5,
        stop_delay_t2_sec: float = 1.0,
        blind_turn2_sec: float = 2.0,
        pause_turn2_sec: float = 0.5,
        calib_turn2_stop_only: bool = False,
        auto_reset_delay_sec: float = 3.0,
        path_seek_tolerance_px: float = 40.0,
        path_seek_max_angle_deg: float = 20.0,
        **kwargs
    ):
        self.min_qr_nav_dim = float(kwargs.get("min_qr_size", min_qr_nav_dim))
        self.min_stop_qr_dim = float(kwargs.get("min_stop_dim", min_stop_qr_dim))
        self.stop_delay_t1_sec = float(kwargs.get("stop_duration_sec", stop_delay_t1_sec))
        self.blind_turn1_sec = float(kwargs.get("delay_turn1_sec", blind_turn1_sec))
        self.pause_turn1_sec = float(kwargs.get("pause_center_sec", pause_turn1_sec))
        self.stop_delay_t2_sec = float(stop_delay_t2_sec)
        self.blind_turn2_sec = float(kwargs.get("delay_turn2_sec", blind_turn2_sec))
        self.pause_turn2_sec = float(pause_turn2_sec)
        self.calib_turn2_stop_only = bool(kwargs.get("calib_turn2_stop_only", calib_turn2_stop_only))
        self.auto_reset_delay_sec = float(auto_reset_delay_sec)
        self.path_seek_tolerance_px = float(path_seek_tolerance_px)
        self.path_seek_max_angle_deg = float(path_seek_max_angle_deg)

        # Backward compatibility properties
        self.delay_turn1_sec = self.blind_turn1_sec
        self.delay_turn2_sec = self.blind_turn2_sec
        self.stop_duration_sec = self.stop_delay_t1_sec
        self.pause_center_sec = self.pause_turn1_sec
        self.confirm_frames = int(kwargs.get("confirm_frames", 1))
        self.min_path_confirm_frames = self.confirm_frames
        self.align_tolerance_px = float(kwargs.get("align_tolerance_px", 180.0))

        self.state = QRTurnMissionState.TRACKING_ROW_1
        self.last_status_desc = "Sẵn sàng (Bám luống 1, chờ mã QR1)"
        self.phase_start_time: Optional[float] = None
        self.turn_start_time: Optional[float] = None
        self.turn_right_start_time: Optional[float] = None
        self.completed_time: Optional[float] = None
        self.consecutive_path_frames: int = 0
        self.is_enabled = True

    def reset(self):
        """Đặt lại trạng thái nhiệm vụ về bám luống 1 ban đầu."""
        self.state = QRTurnMissionState.TRACKING_ROW_1
        self.last_status_desc = "Sẵn sàng (Bám luống 1, chờ mã QR1)"
        self.phase_start_time = None
        self.turn_start_time = None
        self.turn_right_start_time = None
        self.completed_time = None
        self.consecutive_path_frames = 0

    @staticmethod
    def is_turn1_code(text: Optional[str]) -> bool:
        if not text:
            return False
        norm = text.strip().upper().replace(" ", "").replace("_", "")
        return "TURN1" in norm or "QR1" in norm

    @staticmethod
    def is_turn2_code(text: Optional[str]) -> bool:
        if not text:
            return False
        norm = text.strip().upper().replace(" ", "").replace("_", "")
        return "TURN2" in norm or "QR2" in norm

    def _make_straight_target(self, cam_cx: int, h_img: int, desc: str) -> FusedNavigationTarget:
        return FusedNavigationTarget(
            target_pt=(cam_cx, h_img // 2),
            lateral_offset_px=0.0,
            offset_direction=desc,
            is_qr_mode=False,
            mode_name="RUN_STRAIGHT",
            active_qr=None,
            ref_y=h_img // 2,
            qr_stop_triggered=False,
            uart_flag_byte=0,
            mission_state=self.state.value,
            mission_status_desc=desc
        )

    def _make_stop_target(self, cam_cx: int, h_img: int, active_qr: Optional[QRDetectionResult], desc: str, b4: int = 1) -> FusedNavigationTarget:
        cy = int(round(active_qr.center[1])) if (active_qr and hasattr(active_qr, "center") and active_qr.center) else (h_img // 2)
        cx = int(round(active_qr.center[0])) if (active_qr and hasattr(active_qr, "center") and active_qr.center) else cam_cx
        return FusedNavigationTarget(
            target_pt=(cx, cy),
            lateral_offset_px=0.0,
            offset_direction=desc,
            is_qr_mode=True if active_qr else False,
            mode_name="STATION_STOP",
            active_qr=active_qr,
            ref_y=cy,
            qr_stop_triggered=True,
            uart_flag_byte=b4,
            mission_state=self.state.value,
            mission_status_desc=desc
        )

    def _make_qr_target(self, cam_cx: int, h_img: int, qr: QRDetectionResult, desc: str, b4: int = 0) -> FusedNavigationTarget:
        cx = float(qr.center[0]) if (hasattr(qr, "center") and qr.center) else float(np.mean(qr.points[:, 0]))
        cy = float(qr.center[1]) if (hasattr(qr, "center") and qr.center) else float(np.mean(qr.points[:, 1]))
        tx, ty = int(round(cx)), int(round(cy))
        offset = float(tx - cam_cx)
        offset = float(np.clip(offset, -cam_cx, cam_cx))
        return FusedNavigationTarget(
            target_pt=(tx, ty),
            lateral_offset_px=offset,
            offset_direction=f"Bám tâm QR ({offset:+.1f} px)",
            is_qr_mode=True,
            mode_name="QR_CENTER",
            active_qr=qr,
            qr_bbox_dim=(qr.width_px, qr.height_px),
            qr_effective_area=qr.area_px,
            ref_y=ty,
            qr_stop_triggered=False,
            uart_flag_byte=b4,
            mission_state=self.state.value,
            mission_status_desc=desc
        )

    def _make_blind_turn_target(self, cam_cx: int, h_img: int, mode_name: str, desc: str, b4: int) -> FusedNavigationTarget:
        return FusedNavigationTarget(
            target_pt=(cam_cx, h_img // 2),
            lateral_offset_px=0.0,
            offset_direction=desc,
            is_qr_mode=False,
            mode_name=mode_name,
            active_qr=None,
            ref_y=h_img // 2,
            qr_stop_triggered=False,
            uart_flag_byte=b4,
            mission_state=self.state.value,
            mission_status_desc=desc
        )

    def process(
        self,
        base_nav: FusedNavigationTarget,
        qr_results: List[QRDetectionResult],
        frame_shape: Tuple[int, int],
        path_res: Optional[DetectionResult] = None,
        now: Optional[float] = None
    ) -> FusedNavigationTarget:
        if not self.is_enabled:
            base_nav.mission_state = self.state.value
            base_nav.mission_status_desc = "Nhiệm vụ QR Turn tắt"
            return base_nav

        h_img, w_img = frame_shape[:2]
        cam_cx = w_img // 2
        t_now = time.time() if now is None else now

        # -------------------------------------------------------------
        # 1. TRACKING_ROW_1 / IDLE: Chạy thẳng bám luống 1
        # -------------------------------------------------------------
        if self.state in (QRTurnMissionState.TRACKING_ROW_1, QRTurnMissionState.IDLE):
            found_qr1: Optional[QRDetectionResult] = None
            found_qr2_direct: Optional[QRDetectionResult] = None
            for qr in qr_results:
                if self.is_turn1_code(qr.text) or (qr.backend == "yolo-fastestv2-qr" and not self.calib_turn2_stop_only):
                    found_qr1 = qr
                    break
                elif self.is_turn2_code(qr.text):
                    found_qr2_direct = qr
                elif found_qr1 is None:
                    found_qr1 = qr

            # Nếu phát hiện trực tiếp QR2 khi đang ở IDLE (ví dụ test riêng trạm 2)
            if found_qr1 is None and found_qr2_direct is not None:
                exceeded_stop, bw, bh, area = check_qr_exceeds_threshold(found_qr2_direct, min_dim=self.min_stop_qr_dim)
                if exceeded_stop:
                    self.state = QRTurnMissionState.STOP_AT_QR2
                    self.phase_start_time = t_now
                    self.turn_right_start_time = t_now
                    self.last_status_desc = f"🛑 ĐẾN TRẠM QR2 (≥{self.min_stop_qr_dim:.0f}px): DỪNG {self.stop_delay_t2_sec:.1f}S (B4=1)"
                    return self._make_stop_target(cam_cx, h_img, found_qr2_direct, self.last_status_desc, b4=1)

            if found_qr1 is not None:
                exceeded_stop, bw, bh, area = check_qr_exceeds_threshold(found_qr1, min_dim=self.min_stop_qr_dim)
                exceeded_nav, _, _, _ = check_qr_exceeds_threshold(found_qr1, min_dim=self.min_qr_nav_dim)

                if exceeded_stop:
                    # Chạm trạm dừng QR1 (>= 200px) -> Chuyển STOP_AT_QR1
                    self.state = QRTurnMissionState.STOP_AT_QR1
                    self.phase_start_time = t_now
                    self.turn_start_time = t_now
                    self.last_status_desc = f"🛑 ĐẾN TRẠM QR1 (≥{self.min_stop_qr_dim:.0f}px): DỪNG {self.stop_delay_t1_sec:.1f}S (B4=1)"
                    return self._make_stop_target(cam_cx, h_img, found_qr1, self.last_status_desc, b4=1)
                elif exceeded_nav:
                    # Gặp QR1 >= min_qr_nav_dim (30px) -> Chuyển sang bám tâm QR1
                    self.state = QRTurnMissionState.TRACKING_QR1
                    self.last_status_desc = f"🎯 GẶP QR1 (≥{self.min_qr_nav_dim:.0f}px): CHẠY THEO TÂM TURN1 (B4=0)"
                    return self._make_qr_target(cam_cx, h_img, found_qr1, self.last_status_desc, b4=0)

            # Hoàn toàn bỏ luống: Chạy thẳng tuyệt đối với offset = 0.0 (B4=0)
            self.last_status_desc = f"🚀 CHẠY THẲNG (Chờ QR1 ≥ {self.min_qr_nav_dim:.0f}px, B4=0)"
            return self._make_straight_target(cam_cx, h_img, self.last_status_desc)

        # -------------------------------------------------------------
        # 2. TRACKING_QR1: Chạy theo tâm turn1 (QR1)
        # -------------------------------------------------------------
        if self.state == QRTurnMissionState.TRACKING_QR1:
            found_qr1 = None
            for qr in qr_results:
                if self.is_turn1_code(qr.text) or (qr.backend == "yolo-fastestv2-qr" and not self.calib_turn2_stop_only):
                    found_qr1 = qr
                    break
                elif not self.is_turn2_code(qr.text):
                    found_qr1 = qr
                    break

            if found_qr1 is not None:
                exceeded_stop, bw, bh, area = check_qr_exceeds_threshold(found_qr1, min_dim=self.min_stop_qr_dim)
                if exceeded_stop:
                    self.state = QRTurnMissionState.STOP_AT_QR1
                    self.phase_start_time = t_now
                    self.turn_start_time = t_now
                    self.last_status_desc = f"🛑 ĐẾN TRẠM QR1 (≥{self.min_stop_qr_dim:.0f}px): DỪNG {self.stop_delay_t1_sec:.1f}S (B4=1)"
                    return self._make_stop_target(cam_cx, h_img, found_qr1, self.last_status_desc, b4=1)
                else:
                    self.last_status_desc = f"🎯 BÁM TÂM QR1 ({bw:.0f}×{bh:.0f}px → {self.min_stop_qr_dim:.0f}px, B4=0)"
                    return self._make_qr_target(cam_cx, h_img, found_qr1, self.last_status_desc, b4=0)
            else:
                self.last_status_desc = "Đang bám tâm QR1 (Tạm mất dấu -> Chạy thẳng, B4=0)"
                return self._make_straight_target(cam_cx, h_img, self.last_status_desc)

        # -------------------------------------------------------------
        # 3. STOP_AT_QR1: Delay 1s dừng tại trạm QR1
        # -------------------------------------------------------------
        if self.state in (QRTurnMissionState.STOP_AT_QR1, QRTurnMissionState.STOP_AT_TURN1):
            if self.phase_start_time is None:
                self.phase_start_time = t_now
            dt = t_now - self.phase_start_time
            if dt < self.stop_delay_t1_sec:
                self.last_status_desc = f"🛑 TRẠM QR1: DỪNG ({dt:.1f}/{self.stop_delay_t1_sec:.1f}s, B4=1)..."
                return self._make_stop_target(cam_cx, h_img, None, self.last_status_desc, b4=1)
            else:
                # Hết 1.0s dừng -> Chuyển sang xoay phải mù 2s (B4=2)
                self.state = QRTurnMissionState.BLIND_TURN_RIGHT
                self.phase_start_time = t_now
                self.turn_start_time = t_now
                self.last_status_desc = f"↪️ BẮT ĐẦU XOAY PHẢI MÙ {self.blind_turn1_sec:.1f}S (B4=2)"
                return self._make_blind_turn_target(cam_cx, h_img, "BLIND_TURN_RIGHT", self.last_status_desc, b4=2)

        # -------------------------------------------------------------
        # 4. BLIND_TURN_RIGHT: Robot xoay phải mù 2s
        # -------------------------------------------------------------
        if self.state in (QRTurnMissionState.BLIND_TURN_RIGHT, QRTurnMissionState.TURNING_RIGHT_SEEK_PATH):
            if self.phase_start_time is None:
                self.phase_start_time = t_now
            dt = t_now - self.phase_start_time
            if dt < self.blind_turn1_sec:
                self.last_status_desc = f"↪️ XOAY PHẢI MÙ ({dt:.1f}/{self.blind_turn1_sec:.1f}s, B4=2)..."
                return self._make_blind_turn_target(cam_cx, h_img, "BLIND_TURN_RIGHT", self.last_status_desc, b4=2)
            else:
                # Hết 2.0s xoay phải mù -> Delay 500ms dập quán tính xoay (B4=1)
                self.state = QRTurnMissionState.PAUSE_AFTER_TURN1
                self.phase_start_time = t_now
                self.last_status_desc = f"⏸️ DỪNG DẬP ĐÀ {self.pause_turn1_sec*1000:.0f}MS SAU XOAY PHẢI (B4=1)"
                return self._make_stop_target(cam_cx, h_img, None, self.last_status_desc, b4=1)

        # -------------------------------------------------------------
        # 5. PAUSE_AFTER_TURN1: Delay 500ms dập quán tính sau xoay phải
        # -------------------------------------------------------------
        if self.state == QRTurnMissionState.PAUSE_AFTER_TURN1:
            if self.phase_start_time is None:
                self.phase_start_time = t_now
            dt = t_now - self.phase_start_time
            if dt < self.pause_turn1_sec:
                self.last_status_desc = f"⏸️ DỪNG DẬP ĐÀ ({dt*1000:.0f}/{self.pause_turn1_sec*1000:.0f}ms, B4=1)..."
                return self._make_stop_target(cam_cx, h_img, None, self.last_status_desc, b4=1)
            else:
                # Hết 500ms dừng -> Bắt đầu chạy tâm theo turn2 (QR2)
                self.state = QRTurnMissionState.TRACKING_QR2
                self.phase_start_time = t_now
                self.last_status_desc = "🎯 BẮT ĐẦU CHẠY THEO TÂM TURN2 (QR2, B4=0)"
                return self._make_straight_target(cam_cx, h_img, self.last_status_desc)

        # -------------------------------------------------------------
        # 6. TRACKING_QR2: Bắt đầu chạy theo tâm turn2 (QR2)
        # -------------------------------------------------------------
        if self.state in (QRTurnMissionState.TRACKING_QR2, QRTurnMissionState.TRACKING_ROW_2):
            found_qr2 = None
            for qr in qr_results:
                if self.is_turn2_code(qr.text) or (qr.backend == "yolo-fastestv2-qr" and not self.is_turn1_code(qr.text)):
                    found_qr2 = qr
                    break
                elif not self.is_turn1_code(qr.text):
                    found_qr2 = qr
                    break

            if found_qr2 is not None:
                exceeded_stop, bw, bh, area = check_qr_exceeds_threshold(found_qr2, min_dim=self.min_stop_qr_dim)
                if exceeded_stop:
                    # Đến khi >200px -> Delay 1s dừng trạm QR2
                    self.state = QRTurnMissionState.STOP_AT_QR2
                    self.phase_start_time = t_now
                    self.turn_right_start_time = t_now
                    self.last_status_desc = f"🛑 ĐẾN TRẠM QR2 (≥{self.min_stop_qr_dim:.0f}px): DỪNG {self.stop_delay_t2_sec:.1f}S (B4=1)"
                    return self._make_stop_target(cam_cx, h_img, found_qr2, self.last_status_desc, b4=1)
                else:
                    self.last_status_desc = f"🎯 BÁM TÂM TURN2 ({bw:.0f}×{bh:.0f}px → {self.min_stop_qr_dim:.0f}px, B4=0)"
                    return self._make_qr_target(cam_cx, h_img, found_qr2, self.last_status_desc, b4=0)
            else:
                self.last_status_desc = "🎯 TIẾN THẲNG THEO TÂM TURN2 (Chờ mã QR2, B4=0)"
                return self._make_straight_target(cam_cx, h_img, self.last_status_desc)

        # -------------------------------------------------------------
        # 7. STOP_AT_QR2: Delay 1s dừng tại trạm QR2
        # -------------------------------------------------------------
        if self.state == QRTurnMissionState.STOP_AT_QR2:
            if self.phase_start_time is None:
                self.phase_start_time = t_now
            dt = t_now - self.phase_start_time
            if dt < self.stop_delay_t2_sec:
                self.last_status_desc = f"🛑 TRẠM QR2: DỪNG ({dt:.1f}/{self.stop_delay_t2_sec:.1f}s, B4=1)..."
                return self._make_stop_target(cam_cx, h_img, None, self.last_status_desc, b4=1)
            else:
                # Hết 1.0s dừng -> Chuyển sang xoay trái mù 2s (B4=3)
                self.state = QRTurnMissionState.BLIND_TURN_LEFT
                self.phase_start_time = t_now
                self.turn_right_start_time = t_now
                self.last_status_desc = f"↩️ BẮT ĐẦU XOAY TRÁI MÙ {self.blind_turn2_sec:.1f}S (B4=3)"
                return self._make_blind_turn_target(cam_cx, h_img, "BLIND_TURN_LEFT", self.last_status_desc, b4=3)

        # -------------------------------------------------------------
        # 8. BLIND_TURN_LEFT: Robot xoay trái mù 2s
        # -------------------------------------------------------------
        if self.state in (QRTurnMissionState.BLIND_TURN_LEFT, QRTurnMissionState.TURNING_LEFT_SEEK_PATH):
            if self.phase_start_time is None:
                self.phase_start_time = t_now
            dt = t_now - self.phase_start_time
            if dt < self.blind_turn2_sec:
                self.last_status_desc = f"↩️ XOAY TRÁI MÙ ({dt:.1f}/{self.blind_turn2_sec:.1f}s, B4=3)..."
                return self._make_blind_turn_target(cam_cx, h_img, "BLIND_TURN_LEFT", self.last_status_desc, b4=3)
            else:
                # Hết 2.0s xoay trái mù -> Delay 500ms dập quán tính xoay (B4=1)
                self.state = QRTurnMissionState.PAUSE_AFTER_TURN2
                self.phase_start_time = t_now
                self.last_status_desc = f"⏸️ DỪNG DẬP ĐÀ {self.pause_turn2_sec*1000:.0f}MS SAU XOAY TRÁI (B4=1)"
                return self._make_stop_target(cam_cx, h_img, None, self.last_status_desc, b4=1)

        # -------------------------------------------------------------
        # 9. PAUSE_AFTER_TURN2: Delay 500ms dập quán tính sau xoay trái
        # -------------------------------------------------------------
        if self.state == QRTurnMissionState.PAUSE_AFTER_TURN2:
            if self.phase_start_time is None:
                self.phase_start_time = t_now
            dt = t_now - self.phase_start_time
            if dt < self.pause_turn2_sec:
                self.last_status_desc = f"⏸️ DỪNG DẬP ĐÀ ({dt*1000:.0f}/{self.pause_turn2_sec*1000:.0f}ms, B4=1)..."
                return self._make_stop_target(cam_cx, h_img, None, self.last_status_desc, b4=1)
            else:
                if self.calib_turn2_stop_only:
                    self.state = QRTurnMissionState.COMPLETED_STOPPED
                    self.completed_time = t_now
                    self.last_status_desc = f"🛑 [CALIB] ĐÃ XOAY XONG {self.blind_turn2_sec:.1f}S: DỪNG HẲN (B4=1)! (Bấm Reset để thử lại)"
                    return self._make_stop_target(cam_cx, h_img, None, self.last_status_desc, b4=1)
                else:
                    # Chạy thẳng tiếp (hoàn toàn bỏ luống, chạy thẳng tuyệt đối offset = 0.0)
                    self.reset()
                    self.last_status_desc = "🚀 ĐÃ HOÀN TẤT CHUYỂN HƯỚNG: CHẠY THẲNG TIẾP (B4=0)"
                    return self._make_straight_target(cam_cx, h_img, self.last_status_desc)

        # -------------------------------------------------------------
        # 10. COMPLETED_STOPPED: Dừng xe Calib
        # -------------------------------------------------------------
        if self.state == QRTurnMissionState.COMPLETED_STOPPED:
            self.last_status_desc = f"🛑 [CALIB] ĐÃ DỪNG HẲN ĐỂ XEM ĐẦU ROBOT (B4=1)"
            return self._make_stop_target(cam_cx, h_img, None, self.last_status_desc, b4=1)

        return base_nav

