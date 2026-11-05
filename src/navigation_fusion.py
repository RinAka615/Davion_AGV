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
    IDLE = "IDLE"                                           # 1. Bám luống 1 (hoặc bám tâm QR1 >100px)
    TURNING_RIGHT_SEEK_PATH = "TURNING_RIGHT_SEEK_PATH"     # 2. Đến trạm QR1 >200px: STM32 dừng 1s & quay PHẢI tìm luống 2 (B4=1)
    TRACKING_ROW_2 = "TRACKING_ROW_2"                       # 3. Bám luống 2 (hoặc bám tâm QR2 >100px) (B4=0)
    TURNING_LEFT_SEEK_PATH = "TURNING_LEFT_SEEK_PATH"       # 4. Đến trạm QR2 >200px: STM32 dừng 1s & quay TRÁI tìm luống (B4=3)
    COMPLETED_STOPPED = "COMPLETED_STOPPED"                 # 5. Dừng xe Calib (nếu bật chk_calib_turn2)

    # Aliases tương thích ngược
    STOP_AT_TURN1 = "TURNING_RIGHT_SEEK_PATH"
    TURNING_LEFT_SEEK_TURN2 = "TURNING_RIGHT_SEEK_PATH"
    APPROACHING_TURN2 = "TRACKING_ROW_2"


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
    min_qr_size: float = 100.0,
    qr_stop_thresh: float = 200.0,
    enable_qr_override: bool = True
) -> FusedNavigationTarget:
    """
    Thuật toán hợp nhất điều hướng thông minh & cờ trạm dừng UART:
    - Cờ Byte 4 gửi STM32 (uart_flag_byte):
      + Nếu phát hiện mã QR có diện tích đường bao > 200x200 (S >= 40,000 px^2 hoặc W, H >= 200 px):
        Gửi cờ = 1 (0x01) xuống STM32.
      + Ngược lại: Gửi cờ = 0 (0x00).
      + Hoàn toàn KHÔNG phụ thuộc vào điều kiện có đường hay mất đường (Line Tracking / Lost).
    - Sai số lái (Bytes 0..3 - lateral_offset_px):
      + Nếu phát hiện mã QR có diện tích đường bao > 100x100:
        Ưu tiên TUYỆT ĐỐI tính toán sai số (lateral offset) và hướng lái dựa theo TÂM CỦA QR.
      + Ngược lại:
        Duy trì tính toán sai số theo TÂM LỐI ĐI (Path Centerline) như bình thường.
    """
    h_img, w_img = frame_shape[:2]
    cam_cx = w_img // 2

    # 1. Kiểm tra điều kiện kích hoạt trạm dừng UART (QR > 200x200)
    qr_stop_triggered = False
    if qr_results:
        for qr in qr_results:
            exceeded_stop, _, _, _ = check_qr_exceeds_threshold(qr, min_dim=qr_stop_thresh)
            if exceeded_stop:
                qr_stop_triggered = True
                break
    uart_flag_byte = 1 if qr_stop_triggered else 0

    # 2. Tìm kiếm mã QR lớn nhất thỏa mãn điều kiện diện tích đường bao > min_qr_size (mặc định > 100x100)
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
    """Quản lý nhiệm vụ chuyển hướng tự động giữa 2 mã QR (Chu trình Ziczac 2 Luống):
    1. Tầng trên (Python - Pi 5):
       - Chỉ đảm nhiệm xử lý thị giác (nhận diện luống, quét mã QR1 & QR2, tính toán sai số px).
       - Hoàn toàn KHÔNG delay/sleep để giữ luồng camera 30-40 FPS và gửi gói tin UART liên tục.
       - Gửi cờ Byte 4:
         • 0: Chạy bám luống hoặc bám tâm QR qua PID (hoặc đã bắt được luống mới ±40px).
         • 1: Chạm trạm QR1 (>200px) -> Yêu cầu STM32 dừng 1.0s và quay PHẢI tìm luống 2.
         • 3: Chạm trạm QR2 (>200px) -> Yêu cầu STM32 dừng 1.0s và quay TRÁI tìm luống mới.
    2. Tầng dưới (STM32):
       - Đảm nhận toàn bộ điều khiển động cơ, phanh hãm và bộ định thời phần cứng (HAL_GetTick()):
         • Nhận flag 1: Phanh dừng 1.0s -> Tự động quay PHẢI (cmd_r = -TURN_SPEED).
         • Nhận flag 0 khi đang quay: Dừng xe 500ms triệt tiêu quán tính xoay -> Vào IDLE bám luống 2.
         • Nhận flag 3: Phanh dừng 1.0s -> Tự động quay TRÁI (cmd_r = +TURN_SPEED).
         • Nhận flag 0 khi đang quay: Dừng xe 500ms triệt tiêu quán tính xoay -> Vào IDLE bám luống mới.
    """
    def __init__(
        self,
        stop_duration_sec: float = 1.0,
        pause_center_sec: float = 0.5,
        align_tolerance_px: float = 180.0,
        min_stop_qr_dim: float = 200.0,
        auto_reset_delay_sec: float = 3.0,
        path_seek_tolerance_px: float = 40.0,
        path_seek_max_angle_deg: float = 20.0,
        delay_turn1_sec: float = 1.0,
        delay_turn2_sec: float = 1.0,
        confirm_frames: int = 1,
        calib_turn2_stop_only: bool = False,
        **kwargs
    ):
        self.stop_duration_sec = stop_duration_sec
        self.pause_center_sec = pause_center_sec
        self.align_tolerance_px = align_tolerance_px
        self.min_stop_qr_dim = min_stop_qr_dim
        self.auto_reset_delay_sec = auto_reset_delay_sec
        self.path_seek_tolerance_px = path_seek_tolerance_px
        self.path_seek_max_angle_deg = path_seek_max_angle_deg
        self.delay_turn1_sec = delay_turn1_sec
        self.delay_turn2_sec = delay_turn2_sec
        self.confirm_frames = int(kwargs.get("confirm_frames", confirm_frames))
        self.min_path_confirm_frames = int(kwargs.get("min_path_confirm_frames", self.confirm_frames))
        self.calib_turn2_stop_only = bool(kwargs.get("calib_turn2_stop_only", calib_turn2_stop_only))

        self.state = QRTurnMissionState.IDLE
        self.last_status_desc = "Sẵn sàng (Bám luống 1, chờ mã QR1)"
        self.turn_start_time: Optional[float] = None
        self.center_locked_time: Optional[float] = None
        self.turn_right_start_time: Optional[float] = None
        self.completed_time: Optional[float] = None
        self.last_turn2_offset: float = 0.0
        self.last_turn2_seen_time: Optional[float] = None
        self.consecutive_turn2_frames: int = 0
        self.consecutive_path_frames: int = 0
        self.is_enabled = True

    def reset(self):
        """Đặt lại trạng thái nhiệm vụ về IDLE ban đầu."""
        self.state = QRTurnMissionState.IDLE
        self.last_status_desc = "Sẵn sàng (Bám luống 1, chờ mã QR1)"
        self.turn_start_time = None
        self.center_locked_time = None
        self.turn_right_start_time = None
        self.completed_time = None
        self.last_turn2_offset = 0.0
        self.last_turn2_seen_time = None
        self.consecutive_turn2_frames = 0
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

    def process(
        self,
        base_nav: FusedNavigationTarget,
        qr_results: List[QRDetectionResult],
        frame_shape: Tuple[int, int],
        path_res: Optional[DetectionResult] = None,
        now: Optional[float] = None
    ) -> FusedNavigationTarget:
        """Xử lý và cập nhật máy trạng thái theo frame hiện tại (chu trình 2 QR ziczac):
        - LUỐNG 1 (IDLE): Bám luống 1 hoặc bám tâm QR1 (>100px).
          Khi QR1 > 200px -> Chuyển TURNING_RIGHT_SEEK_PATH, gửi Byte 4 = 1.
        - QUAY PHẢI TÌM LUỐNG 2 (TURNING_RIGHT_SEEK_PATH):
          STM32 tự động dừng 1.0s rồi quay PHẢI (cmd_r = -TURN_SPEED).
          Pi chờ sau delay_turn1_sec (chống nhiễu), khi bắt được luống mới |dx| <= 40px:
          Chuyển sang TRACKING_ROW_2, gửi Byte 4 = 0 (STM32 dừng 500ms rồi bám luống 2).
        - LUỐNG 2 (TRACKING_ROW_2): Bám luống 2 hoặc bám tâm QR2 (>100px).
          Khi QR2 > 200px -> Chuyển TURNING_LEFT_SEEK_PATH, gửi Byte 4 = 3.
        - QUAY TRÁI TÌM LUỐNG MỚI (TURNING_LEFT_SEEK_PATH):
          STM32 tự động dừng 1.0s rồi quay TRÁI (cmd_r = +TURN_SPEED).
          Pi chờ sau delay_turn2_sec (chống nhiễu), khi bắt được luống mới |dx| <= 40px:
          Chuyển về IDLE, gửi Byte 4 = 0 (STM32 dừng 500ms rồi bám luống mới lặp lại chu trình).
        """
        if not self.is_enabled:
            base_nav.mission_state = self.state.value
            base_nav.mission_status_desc = "Nhiệm vụ QR Turn tắt"
            return base_nav

        h_img, w_img = frame_shape[:2]
        cam_cx = w_img // 2
        t_now = time.time() if now is None else now

        # -------------------------------------------------------------
        # TRẠNG THÁI 1: IDLE - Bám luống 1 hoặc bám tâm QR1 (>100px)
        # -------------------------------------------------------------
        if self.state == QRTurnMissionState.IDLE:
            turn1_stop_qr: Optional[QRDetectionResult] = None
            turn2_stop_qr: Optional[QRDetectionResult] = None
            for qr in qr_results:
                if self.is_turn1_code(qr.text) or (qr.backend == "yolo-fastestv2-qr" and not self.calib_turn2_stop_only):
                    exceeded, bw, bh, area = check_qr_exceeds_threshold(qr, min_dim=self.min_stop_qr_dim)
                    if exceeded:
                        turn1_stop_qr = qr
                        break
                elif self.is_turn2_code(qr.text):
                    exceeded, bw, bh, area = check_qr_exceeds_threshold(qr, min_dim=self.min_stop_qr_dim)
                    if exceeded:
                        turn2_stop_qr = qr
                        break

            if turn1_stop_qr is not None:
                # Đã chạm trạm dừng QR1 (>= 200px) -> Kích hoạt dừng 1s & xoay PHẢI tìm luống 2 (B4=1)
                self.state = QRTurnMissionState.TURNING_RIGHT_SEEK_PATH
                self.turn_start_time = t_now
                self.consecutive_path_frames = 0
                self.last_status_desc = f"🛑 ĐÃ ĐẾN TRẠM QR1 (≥{self.min_stop_qr_dim:.0f}px): GỬI B4=1 (STM32 DỪNG 1.0S & QUAY PHẢI)"

                cx = float(turn1_stop_qr.center[0]) if (hasattr(turn1_stop_qr, "center") and turn1_stop_qr.center) else float(np.mean(turn1_stop_qr.points[:, 0]))
                cy = float(turn1_stop_qr.center[1]) if (hasattr(turn1_stop_qr, "center") and turn1_stop_qr.center) else float(np.mean(turn1_stop_qr.points[:, 1]))
                return FusedNavigationTarget(
                    target_pt=(int(round(cx)), int(round(cy))),
                    lateral_offset_px=0.0,
                    offset_direction="Đã đến trạm QR1 (Gửi B4=1)",
                    is_qr_mode=True,
                    mode_name="TURN1_STATION",
                    active_qr=turn1_stop_qr,
                    ref_y=int(round(cy)),
                    qr_stop_triggered=True,
                    uart_flag_byte=1,
                    mission_state=self.state.value,
                    mission_status_desc=self.last_status_desc
                )
            elif turn2_stop_qr is not None:
                # Kích hoạt trực tiếp tại trạm QR2 (>= 200px) -> Dừng 1s & xoay TRÁI tìm luống (B4=3)
                self.state = QRTurnMissionState.TURNING_LEFT_SEEK_PATH
                self.turn_right_start_time = t_now
                self.consecutive_path_frames = 0
                self.last_status_desc = f"🛑 ĐÃ ĐẾN TRẠM QR2 (≥{self.min_stop_qr_dim:.0f}px): GỬI B4=3 (STM32 DỪNG 1.0S & QUAY TRÁI)"

                cx = float(turn2_stop_qr.center[0]) if (hasattr(turn2_stop_qr, "center") and turn2_stop_qr.center) else float(np.mean(turn2_stop_qr.points[:, 0]))
                cy = float(turn2_stop_qr.center[1]) if (hasattr(turn2_stop_qr, "center") and turn2_stop_qr.center) else float(np.mean(turn2_stop_qr.points[:, 1]))
                return FusedNavigationTarget(
                    target_pt=(int(round(cx)), int(round(cy))),
                    lateral_offset_px=0.0,
                    offset_direction="Đã đến trạm QR2 (Gửi B4=3)",
                    is_qr_mode=True,
                    mode_name="TURNING_LEFT_SEEK_PATH",
                    active_qr=turn2_stop_qr,
                    ref_y=int(round(cy)),
                    qr_stop_triggered=True,
                    uart_flag_byte=3,
                    mission_state=self.state.value,
                    mission_status_desc=self.last_status_desc
                )
            else:
                base_nav.mission_state = self.state.value
                base_nav.mission_status_desc = self.last_status_desc
                base_nav.uart_flag_byte = 0
                base_nav.qr_stop_triggered = False
                return base_nav

        # -------------------------------------------------------------
        # TRẠNG THÁI 2: TURNING_RIGHT_SEEK_PATH - Quay PHẢI tìm luống 2 (B4=1)
        # -------------------------------------------------------------
        if self.state == QRTurnMissionState.TURNING_RIGHT_SEEK_PATH:
            if self.turn_start_time is None:
                self.turn_start_time = t_now
            dt = t_now - self.turn_start_time

            # 1. TRỄ BAN ĐẦU (delay_turn1_sec): STM32 đang dừng 1.0s và bắt đầu quay phải
            if dt < self.delay_turn1_sec:
                self.consecutive_path_frames = 0
                self.last_status_desc = f"🛑 TRẠM QR1: TRỄ BAN ĐẦU ({dt:.1f}/{self.delay_turn1_sec:.1f}s, B4=1)..."
                return FusedNavigationTarget(
                    target_pt=(cam_cx, h_img // 2),
                    lateral_offset_px=0.0,
                    offset_direction=f"Trễ bắt đầu xoay QR1 ({dt:.1f}s)",
                    is_qr_mode=False,
                    mode_name="TURNING_RIGHT_SEEK_PATH",
                    active_qr=None,
                    ref_y=h_img // 2,
                    qr_stop_triggered=True,
                    uart_flag_byte=1,
                    mission_state=self.state.value,
                    mission_status_desc=self.last_status_desc
                )

            # 2. Sau delay_turn1_sec: Quét tìm luống 2 với độ lệch tâm <= path_seek_tolerance_px (mặc định ±40px)
            has_path = False
            cur_offset: Optional[float] = None
            cur_angle: float = 0.0

            if path_res is not None and path_res.state == TrackingState.TRACKING:
                cur_offset = path_res.lateral_offset_px
                cur_angle = calculate_path_heading_angle(path_res) or 0.0
                if cur_offset is not None:
                    is_offset_ok = abs(cur_offset) <= self.path_seek_tolerance_px
                    is_angle_ok = abs(cur_angle) <= self.path_seek_max_angle_deg
                    if is_offset_ok and is_angle_ok:
                        has_path = True
            elif base_nav.mode_name == "PATH_LINE" and base_nav.target_pt is not None:
                cur_offset = base_nav.lateral_offset_px
                cur_angle = 0.0
                if cur_offset is not None and abs(cur_offset) <= self.path_seek_tolerance_px:
                    has_path = True

            min_confirm = getattr(self, "confirm_frames", getattr(self, "min_path_confirm_frames", 1))
            if has_path:
                self.consecutive_path_frames += 1
            else:
                self.consecutive_path_frames = 0

            if self.consecutive_path_frames >= min_confirm:
                # ĐÃ BẮT TRÚNG TIM LUỐNG 2 (<= 40px) -> Chuyển sang bám luống 2, gửi B4 = 0!
                self.state = QRTurnMissionState.TRACKING_ROW_2
                self.consecutive_path_frames = 0
                off_val = cur_offset if cur_offset is not None else 0.0
                self.last_status_desc = f"🌿 BẮT ĐƯỢC LUỐNG 2 (Lệch {off_val:+.1f}px ≤ ±{self.path_seek_tolerance_px:.0f}px): GỬI B4=0 (STM32 DỪNG 500MS RỒI BÁM LUỐNG)"

                target_pt = (path_res.path_center or path_res.box_center) if (path_res and path_res.state == TrackingState.TRACKING) else base_nav.target_pt
                return FusedNavigationTarget(
                    target_pt=target_pt,
                    lateral_offset_px=off_val,
                    offset_direction=f"Bắt trúng luống 2 ({off_val:+.1f} px)",
                    is_qr_mode=False,
                    mode_name="PATH_LINE",
                    active_qr=None,
                    ref_y=h_img // 2,
                    qr_stop_triggered=False,
                    uart_flag_byte=0, # 0 = Báo STM32 dừng 500ms dập đà xoay rồi vào IDLE bám luống 2
                    mission_state=self.state.value,
                    mission_status_desc=self.last_status_desc
                )
            else:
                # Chưa thỏa mãn -> Tiếp tục gửi Byte 4 = 1 để STM32 duy trì quay phải
                self.last_status_desc = f"🔄 ĐANG QUAY PHẢI TÌM LUỐNG 2 (Chờ vào tâm ±{self.path_seek_tolerance_px:.0f}px, hiện tại: {cur_offset:+.1f}px)..." if cur_offset is not None else "🔄 ĐANG QUAY PHẢI TÌM LUỐNG 2 (B4=1)..."
                return FusedNavigationTarget(
                    target_pt=(cam_cx, h_img // 2),
                    lateral_offset_px=0.0,
                    offset_direction="Đang quay phải tìm luống 2",
                    is_qr_mode=False,
                    mode_name="TURNING_RIGHT_SEEK_PATH",
                    active_qr=None,
                    ref_y=h_img // 2,
                    qr_stop_triggered=False,
                    uart_flag_byte=1,
                    mission_state=self.state.value,
                    mission_status_desc=self.last_status_desc
                )

        # -------------------------------------------------------------
        # TRẠNG THÁI 3: TRACKING_ROW_2 - Bám luống 2 hoặc bám tâm QR2 (>100px)
        # -------------------------------------------------------------
        if self.state == QRTurnMissionState.TRACKING_ROW_2:
            turn2_stop_qr: Optional[QRDetectionResult] = None
            for qr in qr_results:
                if self.is_turn2_code(qr.text) or (qr.backend == "yolo-fastestv2-qr" and not self.is_turn1_code(qr.text)):
                    exceeded, bw, bh, area = check_qr_exceeds_threshold(qr, min_dim=self.min_stop_qr_dim)
                    if exceeded:
                        turn2_stop_qr = qr
                        break

            if turn2_stop_qr is not None:
                # ĐÃ ĐẾN TRẠM QR2 (>= 200px) -> Kích hoạt dừng 1s & xoay TRÁI tìm luống (B4=3)!
                self.state = QRTurnMissionState.TURNING_LEFT_SEEK_PATH
                self.turn_right_start_time = t_now
                self.consecutive_path_frames = 0
                self.last_status_desc = f"🛑 ĐÃ ĐẾN TRẠM QR2 (≥{self.min_stop_qr_dim:.0f}px): GỬI B4=3 (STM32 DỪNG 1.0S & QUAY TRÁI)"

                cx = float(turn2_stop_qr.center[0]) if (hasattr(turn2_stop_qr, "center") and turn2_stop_qr.center) else float(np.mean(turn2_stop_qr.points[:, 0]))
                cy = float(turn2_stop_qr.center[1]) if (hasattr(turn2_stop_qr, "center") and turn2_stop_qr.center) else float(np.mean(turn2_stop_qr.points[:, 1]))
                return FusedNavigationTarget(
                    target_pt=(int(round(cx)), int(round(cy))),
                    lateral_offset_px=0.0,
                    offset_direction="Đã đến trạm QR2 (Gửi B4=3)",
                    is_qr_mode=True,
                    mode_name="TURNING_LEFT_SEEK_PATH",
                    active_qr=turn2_stop_qr,
                    ref_y=int(round(cy)),
                    qr_stop_triggered=True,
                    uart_flag_byte=3,
                    mission_state=self.state.value,
                    mission_status_desc=self.last_status_desc
                )
            else:
                base_nav.mission_state = self.state.value
                base_nav.mission_status_desc = self.last_status_desc
                base_nav.uart_flag_byte = 0
                base_nav.qr_stop_triggered = False
                return base_nav

        # -------------------------------------------------------------
        # TRẠNG THÁI 4: TURNING_LEFT_SEEK_PATH - Quay TRÁI tìm luống mới (B4=3)
        # -------------------------------------------------------------
        if self.state in (QRTurnMissionState.TURNING_LEFT_SEEK_PATH,):
            if self.turn_right_start_time is None:
                self.turn_right_start_time = t_now
            dt = t_now - self.turn_right_start_time

            # 1. TRỄ BAN ĐẦU (delay_turn2_sec): STM32 đang dừng 1.0s và bắt đầu quay trái
            if dt < self.delay_turn2_sec:
                self.consecutive_path_frames = 0
                self.last_status_desc = f"🔄 [CALIB T2] ĐANG XOAY TRÁI ({dt:.1f}/{self.delay_turn2_sec:.1f}s, B4=3)..."
                return FusedNavigationTarget(
                    target_pt=(cam_cx, h_img // 2),
                    lateral_offset_px=0.0,
                    offset_direction=f"Đang xoay T2 ({dt:.1f}/{self.delay_turn2_sec:.1f}s)",
                    is_qr_mode=False,
                    mode_name="TURNING_LEFT_SEEK_PATH",
                    active_qr=None,
                    ref_y=h_img // 2,
                    qr_stop_triggered=False,
                    uart_flag_byte=3,
                    mission_state=self.state.value,
                    mission_status_desc=self.last_status_desc
                )

            # 2. Chế độ Calib: Dừng kiểm tra đầu xe
            if getattr(self, "calib_turn2_stop_only", False):
                self.state = QRTurnMissionState.COMPLETED_STOPPED
                self.completed_time = t_now
                self.last_status_desc = f"🛑 [CALIB T2] ĐÃ XOAY XONG {self.delay_turn2_sec:.1f}S: DỪNG HẲN (B4=0)! (Bấm Reset để thử lại)"
                return FusedNavigationTarget(
                    target_pt=(cam_cx, h_img // 2),
                    lateral_offset_px=0.0,
                    offset_direction=f"Calib xong {self.delay_turn2_sec:.1f}s -> Dừng hẳn (B4=0)",
                    is_qr_mode=False,
                    mode_name="COMPLETED_STOPPED",
                    active_qr=None,
                    ref_y=h_img // 2,
                    qr_stop_triggered=True,
                    uart_flag_byte=0,
                    mission_state=self.state.value,
                    mission_status_desc=self.last_status_desc
                )

            # 3. Quét tìm luống mới với độ lệch tâm <= path_seek_tolerance_px (mặc định ±40px)
            has_path = False
            cur_offset: Optional[float] = None
            cur_angle: float = 0.0

            if path_res is not None and path_res.state == TrackingState.TRACKING:
                cur_offset = path_res.lateral_offset_px
                cur_angle = calculate_path_heading_angle(path_res) or 0.0
                if cur_offset is not None:
                    is_offset_ok = abs(cur_offset) <= self.path_seek_tolerance_px
                    is_angle_ok = abs(cur_angle) <= self.path_seek_max_angle_deg
                    if is_offset_ok and is_angle_ok:
                        has_path = True
            elif base_nav.mode_name == "PATH_LINE" and base_nav.target_pt is not None:
                cur_offset = base_nav.lateral_offset_px
                cur_angle = 0.0
                if cur_offset is not None and abs(cur_offset) <= self.path_seek_tolerance_px:
                    has_path = True

            min_confirm = getattr(self, "confirm_frames", getattr(self, "min_path_confirm_frames", 1))
            if has_path:
                self.consecutive_path_frames += 1
            else:
                self.consecutive_path_frames = 0

            if self.consecutive_path_frames >= min_confirm:
                # ĐÃ BẮT TRÚNG TIM LUỐNG MỚI (<= 40px) -> Loop lại IDLE, gửi B4 = 0!
                self.reset()
                off_val = cur_offset if cur_offset is not None else 0.0
                self.last_status_desc = f"🌿 BẮT ĐƯỢC LUỐNG MỚI (Lệch {off_val:+.1f}px ≤ ±{self.path_seek_tolerance_px:.0f}px): GỬI B4=0 (STM32 DỪNG 500MS RỒI BÁM LUỐNG)"

                target_pt = (path_res.path_center or path_res.box_center) if (path_res and path_res.state == TrackingState.TRACKING) else base_nav.target_pt
                return FusedNavigationTarget(
                    target_pt=target_pt,
                    lateral_offset_px=off_val,
                    offset_direction=f"Bắt trúng luống mới ({off_val:+.1f} px)",
                    is_qr_mode=False,
                    mode_name="PATH_LINE",
                    active_qr=None,
                    ref_y=h_img // 2,
                    qr_stop_triggered=False,
                    uart_flag_byte=0, # 0 = Báo STM32 dừng 500ms dập đà xoay rồi vào IDLE bám luống mới
                    mission_state=self.state.value,
                    mission_status_desc=self.last_status_desc
                )
            else:
                # Chưa thỏa mãn -> Tiếp tục gửi Byte 4 = 3 để STM32 duy trì quay trái
                self.last_status_desc = f"🔄 ĐANG QUAY TRÁI TÌM LUỐNG (Chờ vào tâm ±{self.path_seek_tolerance_px:.0f}px, hiện tại: {cur_offset:+.1f}px)..." if cur_offset is not None else "🔄 ĐANG QUAY TRÁI TÌM LUỐNG (B4=3)..."
                return FusedNavigationTarget(
                    target_pt=(cam_cx, h_img // 2),
                    lateral_offset_px=0.0,
                    offset_direction="Đang quay trái tìm luống mới",
                    is_qr_mode=False,
                    mode_name="TURNING_LEFT_SEEK_PATH",
                    active_qr=None,
                    ref_y=h_img // 2,
                    qr_stop_triggered=False,
                    uart_flag_byte=3,
                    mission_state=self.state.value,
                    mission_status_desc=self.last_status_desc
                )

        # -------------------------------------------------------------
        # TRẠNG THÁI 5: COMPLETED_STOPPED - Dừng xe Calib
        # -------------------------------------------------------------
        if self.state == QRTurnMissionState.COMPLETED_STOPPED:
            t_now = time.time() if now is None else now
            if self.completed_time is None:
                self.completed_time = t_now

            dt_stop = t_now - self.completed_time
            # Trong 0.5s đầu: Gửi B4=0 để STM32 nhận diện dừng xoay và hãm phanh 500ms
            # Sau 0.5s: Gửi B4=1 với lateral_offset_px = 0.0 để giữ xe đứng yên hoàn toàn không bám luống
            flag_stop = 0 if dt_stop < 0.5 else 1

            self.last_status_desc = f"🛑 [CALIB T2] ĐÃ XOAY XONG {self.delay_turn2_sec:.1f}S: DỪNG HẲN ĐỂ XEM ĐẦU ROBOT (B4={flag_stop})"
            return FusedNavigationTarget(
                target_pt=(cam_cx, h_img // 2),
                lateral_offset_px=0.0,
                offset_direction=f"Calib dừng hoàn tất (B4={flag_stop})",
                is_qr_mode=False,
                mode_name="COMPLETED_STOPPED",
                active_qr=None,
                ref_y=h_img // 2,
                qr_stop_triggered=True,
                uart_flag_byte=flag_stop,
                mission_state=self.state.value,
                mission_status_desc=self.last_status_desc
            )

        return base_nav

