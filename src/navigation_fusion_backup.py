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
    IDLE = "IDLE"                                       # 1. Bám đường / tiếp cận TURN_1
    STOP_AT_TURN1 = "STOP_AT_TURN1"                     # 2. Dừng 1s tại trạm TURN_1 (Byte 4 = 1)
    TURNING_LEFT_SEEK_TURN2 = "TURNING_LEFT_SEEK_TURN2" # 3. Hết 1s, rẽ trái cho đến khi bắt được tâm TURN_2 (Byte 4 = 2)
    APPROACHING_TURN2 = "APPROACHING_TURN2"             # 4. Tạm dừng 500ms triệt tiêu quán tính rồi tiến theo tâm TURN_2
    TURNING_RIGHT_SEEK_PATH = "TURNING_RIGHT_SEEK_PATH" # 5. Đã đến trạm TURN_2: Dừng 500ms rồi xoay trái tìm đường (Byte 4 = 3)
    TURNING_LEFT_SEEK_PATH = "TURNING_RIGHT_SEEK_PATH"  # Bí danh (alias) đồng nhất toàn bộ xoay trái
    COMPLETED_STOPPED = "COMPLETED_STOPPED"             # 6. Dừng xe hoàn tất (nếu có)


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
    """Quản lý nhiệm vụ chuyển hướng tự động giữa 2 mã QR (Giao thức phân tầng với STM32):
    1. Tầng trên (Python): Tiếp cận TURN_1 (đường bao >= min_stop_qr_dim, mặc định 200px) -> Gửi Byte 4 = 1.
    2. Tầng dưới (STM32): Khi nhận Byte 4 = 1, STM32 tự động dừng xe 1.0 giây bằng phần cứng,
       sau đó tự động quay trái tại chỗ (cmd_r = TURN_SPEED = 60.0f).
    3. Tầng trên (Python): Trong khi STM32 quay trái, camera tìm mã TURN_2 và đo độ lệch tâm:
       - Khi chưa vào tâm (|offset| > align_tolerance_px): Gửi Byte 4 = 0 (STM32 tiếp tục quay trái).
       - Khi đã bắt trúng tâm TURN_2 (|offset| <= align_tolerance_px): Gửi Byte 4 = 2.
    4. Tầng dưới (STM32): Khi nhận Byte 4 = 2, STM32 dừng 500ms triệt tiêu quán tính quay,
       sau đó tự động tiến lên bám theo tâm TURN_2 qua PID (uout_path).
    5. Tầng trên (Python): Tiếp tục gửi sai số tâm TURN_2 (Bytes 0..3):
       - Trong 500ms đầu: gửi Byte 4 = 2 (tạm dừng triệt tiêu quán tính).
       - Sau 500ms: gửi Byte 4 = 0 (xe tiến theo tâm TURN_2).
       - Khi khoảng cách tới TURN_2 đạt ngưỡng trạm dừng (>= 200px): Gửi Byte 4 = 1 để STM32 DỪNG HẲN HOÀN TẤT!
    """
    def __init__(
        self,
        stop_duration_sec: float = 1.0,
        pause_center_sec: float = 0.5,
        align_tolerance_px: float = 180.0, # Biến offset để bắt tâm (mặc định ±180px trên khung 1280px)
        min_stop_qr_dim: float = 200.0,
        auto_reset_delay_sec: float = 3.0,
        path_seek_tolerance_px: float = 200.0,
        path_seek_max_angle_deg: float = 20.0,
        delay_turn1_sec: float = 1.0,
        delay_turn2_sec: float = 1.0,
        confirm_frames: int = 2,
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

        self.state = QRTurnMissionState.IDLE
        self.last_status_desc = "Sẵn sàng (Chờ tiếp cận mã TURN_1)"
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
        self.last_status_desc = "Sẵn sàng (Chờ tiếp cận mã TURN_1)"
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
        return "TURN1" in norm

    @staticmethod
    def is_turn2_code(text: Optional[str]) -> bool:
        if not text:
            return False
        norm = text.strip().upper().replace(" ", "").replace("_", "")
        return "TURN2" in norm

    def process(
        self,
        base_nav: FusedNavigationTarget,
        qr_results: List[QRDetectionResult],
        frame_shape: Tuple[int, int],
        path_res: Optional[DetectionResult] = None,
        now: Optional[float] = None
    ) -> FusedNavigationTarget:
        """Xử lý và cập nhật máy trạng thái theo frame hiện tại."""
        if not self.is_enabled:
            base_nav.mission_state = self.state.value
            base_nav.mission_status_desc = "Nhiệm vụ QR Turn tắt"
            return base_nav

        h_img, w_img = frame_shape[:2]
        cam_cx = w_img // 2

        # -------------------------------------------------------------
        # TRẠNG THÁI 1: IDLE - Chờ tiếp cận mã TURN_1
        # -------------------------------------------------------------
        if self.state == QRTurnMissionState.IDLE:
            turn1_stop_qr: Optional[QRDetectionResult] = None
            for qr in qr_results:
                if self.is_turn1_code(qr.text):
                    exceeded, bw, bh, area = check_qr_exceeds_threshold(qr, min_dim=self.min_stop_qr_dim)
                    if exceeded:
                        turn1_stop_qr = qr
                        break

            if turn1_stop_qr is not None:
                # Đã chạm trạm dừng TURN_1 (>= 200px) -> Gửi cờ Flag 1 đến STM32!
                self.state = QRTurnMissionState.TURNING_LEFT_SEEK_TURN2
                t_now = time.time() if now is None else now
                self.turn_start_time = t_now
                self.consecutive_turn2_frames = 0
                self.last_status_desc = f"🛑 ĐÃ ĐẾN TRẠM TURN_1 (≥{self.min_stop_qr_dim:.0f}px): GỬI B4=1 (CHỜ TRỄ T1 {self.delay_turn1_sec:.1f}S & QUAY TRÁI)"

                cx = float(turn1_stop_qr.center[0]) if (hasattr(turn1_stop_qr, "center") and turn1_stop_qr.center) else float(np.mean(turn1_stop_qr.points[:, 0]))
                cy = float(turn1_stop_qr.center[1]) if (hasattr(turn1_stop_qr, "center") and turn1_stop_qr.center) else float(np.mean(turn1_stop_qr.points[:, 1]))
                return FusedNavigationTarget(
                    target_pt=(int(round(cx)), int(round(cy))),
                    lateral_offset_px=0.0,
                    offset_direction="Đã đến trạm TURN_1 (Gửi B4=1)",
                    is_qr_mode=True,
                    mode_name="TURN1_STATION",
                    active_qr=turn1_stop_qr,
                    ref_y=int(round(cy)),
                    qr_stop_triggered=True,
                    uart_flag_byte=1,  # 1 = Đã đến trạm TURN_1 (Kích hoạt delay 1s & quay trái trên STM32)
                    mission_state=self.state.value,
                    mission_status_desc=self.last_status_desc
                )
            else:
                base_nav.mission_state = self.state.value
                base_nav.mission_status_desc = self.last_status_desc
                # Khi ở IDLE và chưa gặp trạm TURN_1 (kể cả vừa auto-reset xong trạm TURN_2):
                # Đảm bảo cờ gửi STM32 = 0 để xe tự do bám đường chạy tiếp luống mới
                base_nav.uart_flag_byte = 0
                base_nav.qr_stop_triggered = False
                return base_nav

        # -------------------------------------------------------------
        # TRẠNG THÁI 2: TURNING_LEFT_SEEK_TURN2 - Tìm và căn tâm mã TURN_2
        # (STM32 đang tự thực hiện delay 1s và quay trái tại chỗ)
        # -------------------------------------------------------------
        if self.state == QRTurnMissionState.TURNING_LEFT_SEEK_TURN2:
            t_now = time.time() if now is None else now

            # 1. TRỄ GIAI ĐOẠN ĐẦU (delay_turn1_sec):
            # Nếu vừa chuyển từ trạm TURN_1 sang xoay, trong khoảng delay_turn1_sec đầu:
            # Gửi Byte 4 = 1 và offset = 0.0 để STM32 nhận trọn tín hiệu dừng 1s & loại bỏ hoàn toàn nhiễu ban đầu
            if self.turn_start_time is not None:
                dt_turn = t_now - self.turn_start_time
                if dt_turn < self.delay_turn1_sec:
                    self.consecutive_turn2_frames = 0
                    self.last_status_desc = f"🛑 TRẠM TURN_1: TRỄ BAN ĐẦU ({dt_turn:.1f}/{self.delay_turn1_sec:.1f}s, B4=1)..."
                    return FusedNavigationTarget(
                        target_pt=(cam_cx, h_img // 2),
                        lateral_offset_px=0.0,
                        offset_direction=f"Trễ bắt đầu xoay TURN_1 ({dt_turn:.1f}s)",
                        is_qr_mode=True,
                        mode_name="TURN1_STATION",
                        active_qr=None,
                        ref_y=h_img // 2,
                        qr_stop_triggered=True,
                        uart_flag_byte=1,
                        mission_state=self.state.value,
                        mission_status_desc=self.last_status_desc
                    )

            turn2_found: Optional[QRDetectionResult] = None
            for qr in qr_results:
                if self.is_turn2_code(qr.text) or (qr.backend == "yolo-fastestv2-qr" and not self.is_turn1_code(qr.text)):
                    turn2_found = qr
                    break

            if turn2_found is not None:
                # Đã nhìn thấy TURN_2 trong tầm nhìn camera
                if hasattr(turn2_found, "center") and turn2_found.center and (turn2_found.center[0] != 0.0 or turn2_found.center[1] != 0.0):
                    cx, cy = turn2_found.center
                else:
                    cx = float(np.mean(turn2_found.points[:, 0]))
                    cy = float(np.mean(turn2_found.points[:, 1]))

                offset_px = float(cx - cam_cx)
                target_pt = (int(round(cx)), int(round(cy)))
                is_centered = abs(offset_px) <= self.align_tolerance_px
                self.last_turn2_offset = offset_px
                self.last_turn2_seen_time = t_now

                if is_centered:
                    # BẮT TRÚNG TÂM TURN_2 -> Kiểm tra số frame xác nhận (Debounce)
                    self.consecutive_turn2_frames += 1
                    if self.consecutive_turn2_frames >= self.confirm_frames:
                        # Đã đạt đủ số frame xác nhận -> CHUYỂN SANG TIẾP CẬN TURN_2 (APPROACHING_TURN2)!
                        # Gửi Flag 2 để STM32 dừng 500ms triệt tiêu quán tính quay
                        self.state = QRTurnMissionState.APPROACHING_TURN2
                        self.center_locked_time = t_now
                        self.last_status_desc = f"🎯 ĐÃ BẮT TÂM TURN_2 (Lệch {offset_px:+.1f}px, {self.consecutive_turn2_frames}/{self.confirm_frames}f): DỪNG {int(self.pause_center_sec*1000)}MS RỒI TIẾN ĐẾN ĐÍCH"
                        return FusedNavigationTarget(
                            target_pt=target_pt,
                            lateral_offset_px=offset_px,
                            offset_direction=f"Đúng tâm TURN_2 ({offset_px:+.1f} px)",
                            is_qr_mode=True,
                            mode_name="APPROACHING_TURN2",
                            active_qr=turn2_found,
                            ref_y=int(round(cy)),
                            qr_stop_triggered=False,
                            uart_flag_byte=2,  # 2 = Bắt đúng tâm TURN_2 -> STM32 tạm dừng 500ms
                            mission_state=self.state.value,
                            mission_status_desc=self.last_status_desc
                        )
                    else:
                        # Chưa đủ số frame xác nhận -> Tiếp tục gửi Flag 0 để xe quay mượt mà, không dừng non do nhiễu chớp tắt
                        self.last_status_desc = f"🎯 ĐANG XÁC NHẬN TÂM TURN_2 ({self.consecutive_turn2_frames}/{self.confirm_frames}f, Lệch {offset_px:+.1f}px)..."
                        return FusedNavigationTarget(
                            target_pt=target_pt,
                            lateral_offset_px=offset_px,
                            offset_direction=f"Xác nhận tâm TURN_2 ({self.consecutive_turn2_frames}/{self.confirm_frames}f)",
                            is_qr_mode=True,
                            mode_name="SEEKING_TURN2_CENTER",
                            active_qr=turn2_found,
                            ref_y=int(round(cy)),
                            qr_stop_triggered=False,
                            uart_flag_byte=0,  # 0 = Để STM32 tiếp tục quay trái
                            mission_state=self.state.value,
                            mission_status_desc=self.last_status_desc
                        )
                else:
                    # Nhìn thấy TURN_2 nhưng chưa vào tâm -> Reset bộ đếm frame xác nhận, gửi Flag 0 để STM32 tiếp tục quay trái
                    self.consecutive_turn2_frames = 0
                    self.last_status_desc = f"🎯 ĐANG CĂN TÂM TURN_2 (Lệch {offset_px:+.1f}px, chờ vào tâm ±{self.align_tolerance_px:.0f}px)..."
                    return FusedNavigationTarget(
                        target_pt=target_pt,
                        lateral_offset_px=offset_px,
                        offset_direction=f"Căn tâm TURN_2 ({offset_px:+.1f} px)",
                        is_qr_mode=True,
                        mode_name="SEEKING_TURN2_CENTER",
                        active_qr=turn2_found,
                        ref_y=int(round(cy)),
                        qr_stop_triggered=False,
                        uart_flag_byte=0,  # 0 = Để STM32 tiếp tục quay trái
                        mission_state=self.state.value,
                        mission_status_desc=self.last_status_desc
                    )
            else:
                self.consecutive_turn2_frames = 0
                # Chưa nhìn thấy TURN_2:
                # Nếu camera vẫn còn thấy TURN_1 >= 200px (trong 1.0s dừng ban đầu của STM32), gửi Flag 1 để đảm bảo tín hiệu
                turn1_still_visible: Optional[QRDetectionResult] = None
                for qr in qr_results:
                    if self.is_turn1_code(qr.text):
                        exceeded, bw, bh, area = check_qr_exceeds_threshold(qr, min_dim=self.min_stop_qr_dim)
                        if exceeded:
                            turn1_still_visible = qr
                            break

                if turn1_still_visible is not None:
                    cx = float(turn1_still_visible.center[0]) if (hasattr(turn1_still_visible, "center") and turn1_still_visible.center) else float(np.mean(turn1_still_visible.points[:, 0]))
                    cy = float(turn1_still_visible.center[1]) if (hasattr(turn1_still_visible, "center") and turn1_still_visible.center) else float(np.mean(turn1_still_visible.points[:, 1]))
                    self.last_status_desc = "🛑 TRẠM TURN_1: GỬI B4=1 (STM32 đang đếm 1.0s delay & chuẩn bị quay)"
                    return FusedNavigationTarget(
                        target_pt=(int(round(cx)), int(round(cy))),
                        lateral_offset_px=0.0,
                        offset_direction="Đang dừng tại TURN_1 (B4=1)",
                        is_qr_mode=True,
                        mode_name="TURN1_STATION",
                        active_qr=turn1_still_visible,
                        ref_y=int(round(cy)),
                        qr_stop_triggered=True,
                        uart_flag_byte=1,
                        mission_state=self.state.value,
                        mission_status_desc=self.last_status_desc
                    )
                else:
                    # Đã quay đi khỏi TURN_1 và chưa thấy TURN_2 -> Gửi Byte 4 = 0
                    self.last_status_desc = "↩️ STM32 ĐANG QUAY TRÁI TÌM MÃ TURN_2..."
                    return FusedNavigationTarget(
                        target_pt=(0, h_img // 2),
                        lateral_offset_px=-float(cam_cx),
                        offset_direction="STM32 đang quay tìm TURN_2",
                        is_qr_mode=True,
                        mode_name="SEEKING_TURN2",
                        active_qr=None,
                        ref_y=h_img // 2,
                        qr_stop_triggered=False,
                        uart_flag_byte=0,  # 0 = Chạy/Xoay bình thường
                        mission_state=self.state.value,
                        mission_status_desc=self.last_status_desc
                    )

        # -------------------------------------------------------------
        # TRẠNG THÁI 3: APPROACHING_TURN2 - Tạm dừng 500ms rồi tiến theo tâm TURN_2
        # (Khi TURN_2 kích thước >= 200px -> Chuyển sang COMPLETED_STOPPED)
        # -------------------------------------------------------------
        if self.state == QRTurnMissionState.APPROACHING_TURN2:
            t_now = time.time() if now is None else now
            dt_pause = t_now - (self.center_locked_time if self.center_locked_time is not None else t_now)

            # Ưu tiên mã có text TURN_2 hoặc box AI (hoặc bất kỳ mã QR nào không phải là TURN_1)
            turn2_found = next((qr for qr in qr_results if self.is_turn2_code(qr.text) or (qr.backend == "yolo-fastestv2-qr" and not self.is_turn1_code(qr.text))), None)
            if turn2_found is None:
                turn2_found = next((qr for qr in qr_results if not self.is_turn1_code(qr.text)), None)

            if turn2_found is not None:
                if hasattr(turn2_found, "center") and turn2_found.center and (turn2_found.center[0] != 0.0 or turn2_found.center[1] != 0.0):
                    cx, cy = turn2_found.center
                else:
                    cx = float(np.mean(turn2_found.points[:, 0]))
                    cy = float(np.mean(turn2_found.points[:, 1]))

                offset_px = float(cx - cam_cx)
                target_pt = (int(round(cx)), int(round(cy)))
                self.last_turn2_offset = offset_px
                self.last_turn2_seen_time = t_now

                exceeded_stop, bw, bh, area = check_qr_exceeds_threshold(turn2_found, min_dim=self.min_stop_qr_dim)

                if exceeded_stop:
                    # ĐÃ ĐẾN TRẠM TURN_2 (Kích thước >= 200px)
                    # -> Chuyển sang DỪNG 500MS & XOAY TRÁI TÌM ĐƯỜNG (Byte 4 = 3)!
                    self.state = QRTurnMissionState.TURNING_RIGHT_SEEK_PATH
                    self.turn_right_start_time = t_now
                    self.consecutive_path_frames = 0
                    self.last_status_desc = f"🛑 ĐÃ ĐẾN TRẠM TURN_2 ({bw:.0f}x{bh:.0f}px ≥ {self.min_stop_qr_dim:.0f}px): DỪNG 500MS & XOAY TRÁI TÌM ĐƯỜNG"
                    return FusedNavigationTarget(
                        target_pt=target_pt,
                        lateral_offset_px=0.0,
                        offset_direction=f"Đến đích TURN_2 ({bw:.0f}x{bh:.0f}px) -> Dừng 500ms xoay trái (B4=3)",
                        is_qr_mode=True,
                        mode_name="TURNING_LEFT_SEEK_PATH",
                        active_qr=turn2_found,
                        qr_bbox_dim=(bw, bh),
                        qr_effective_area=area,
                        ref_y=int(round(cy)),
                        qr_stop_triggered=False,
                        uart_flag_byte=3,  # 3 = Đến trạm TURN_2: Dừng 500ms rồi xoay trái tìm đường trên STM32
                        mission_state=self.state.value,
                        mission_status_desc=self.last_status_desc
                    )
                else:
                    # Đang trong 500ms dừng triệt tiêu quán tính hay đang tiến?
                    if dt_pause < self.pause_center_sec:
                        flag_to_send = 2
                        self.last_status_desc = f"⏸️ TẠM DỪNG 500MS TRIỆT TIÊU QUÁN TÍNH ({dt_pause*1000:.0f}/{self.pause_center_sec*1000:.0f}ms) (Lệch {offset_px:+.1f}px)"
                    else:
                        flag_to_send = 0
                        self.last_status_desc = f"🚀 TIẾN THEO TÂM TURN_2 ({bw:.0f}x{bh:.0f}px / {self.min_stop_qr_dim:.0f}px, Lệch {offset_px:+.1f}px)"

                    return FusedNavigationTarget(
                        target_pt=target_pt,
                        lateral_offset_px=offset_px,
                        offset_direction=f"Theo tâm TURN_2 ({offset_px:+.1f} px)",
                        is_qr_mode=True,
                        mode_name="APPROACHING_TURN2",
                        active_qr=turn2_found,
                        qr_bbox_dim=(bw, bh),
                        qr_effective_area=area,
                        ref_y=int(round(cy)),
                        qr_stop_triggered=False,
                        uart_flag_byte=flag_to_send,
                        mission_state=self.state.value,
                        mission_status_desc=self.last_status_desc
                    )
            else:
                # Tạm thời mất dấu TURN_2 do rung lắc xe khi bắt đầu tăng tốc:
                # Giữ nguyên độ lệch tâm gần nhất (trong vòng 1.2s) để xe không bị giật lái về 0!
                flag_to_send = 2 if dt_pause < self.pause_center_sec else 0
                time_since_seen = (t_now - self.last_turn2_seen_time) if self.last_turn2_seen_time is not None else 999.0
                if time_since_seen <= 1.2:
                    offset_to_use = self.last_turn2_offset
                    self.last_status_desc = f"🚀 TIẾN THEO TÂM TURN_2 (Giữ tâm {offset_to_use:+.1f}px, rung lắc {time_since_seen:.1f}s)..."
                else:
                    offset_to_use = 0.0
                    self.last_status_desc = f"🚀 TIẾN THẲNG THEO HƯỚNG TURN_2 (Mất dấu >1.2s)..."

                return FusedNavigationTarget(
                    target_pt=(int(round(cam_cx + offset_to_use)), h_img // 2),
                    lateral_offset_px=offset_to_use,
                    offset_direction=f"Tiến theo tâm TURN_2 ({offset_to_use:+.1f} px)",
                    is_qr_mode=True,
                    mode_name="APPROACHING_TURN2",
                    active_qr=None,
                    ref_y=h_img // 2,
                    qr_stop_triggered=False,
                    uart_flag_byte=flag_to_send,
                    mission_state=self.state.value,
                    mission_status_desc=self.last_status_desc
                )

        # -------------------------------------------------------------
        # TRẠNG THÁI 4: TURNING_RIGHT_SEEK_PATH - Dừng delay_turn2_sec rồi xoay trái tìm đường
        # (Khi camera detect được cái đường -> Dừng xoay và Reset nhiệm vụ về IDLE)
        # -------------------------------------------------------------
        if self.state == QRTurnMissionState.TURNING_RIGHT_SEEK_PATH:
            t_now = time.time() if now is None else now
            if self.turn_right_start_time is None:
                self.turn_right_start_time = t_now

            dt = t_now - self.turn_right_start_time

            # 1. Trong khoảng delay_turn2_sec đầu:
            # Tạm dừng triệt tiêu quán tính tiến và bắt đầu xoay thoát khỏi khu vực trạm TURN_2.
            # Hoàn toàn BỎ QUA dò tìm đường ban đầu để chống nhận nhầm vạch rác làm xe ủi bậy bạ!
            if dt < self.delay_turn2_sec:
                self.consecutive_path_frames = 0
                self.last_status_desc = f"⏸️ TRẠM TURN_2: TRỄ XOAY ({dt:.1f}/{self.delay_turn2_sec:.1f}s, B4=3) TRƯỚC KHI DÒ ĐƯỜNG..."
                return FusedNavigationTarget(
                    target_pt=(cam_cx, h_img // 2),
                    lateral_offset_px=0.0,
                    offset_direction=f"Trễ bắt đầu xoay TURN_2 ({dt:.1f}s)",
                    is_qr_mode=False,
                    mode_name="TURNING_LEFT_SEEK_PATH",
                    active_qr=None,
                    ref_y=h_img // 2,
                    qr_stop_triggered=False,
                    uart_flag_byte=3,
                    mission_state=self.state.value,
                    mission_status_desc=self.last_status_desc
                )

            # 2. Sau delay_turn2_sec: STM32 đang xoay trái tìm đường (cmd_r = +TURN_SPEED)
            # CÁCH 3: Kiểm tra ĐỒNG THỜI cả Vị trí tâm luống (|offset| <= 200px) VÀ Góc nghiêng (|angle| <= 20 độ)
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

            min_confirm = getattr(self, "confirm_frames", getattr(self, "min_path_confirm_frames", 2))
            if has_path:
                self.consecutive_path_frames += 1
            else:
                self.consecutive_path_frames = 0

            if self.consecutive_path_frames >= min_confirm:
                # ĐÃ DETECT ĐƯỢC LUỐNG ĐƯỜNG MỚI THỎA MÃN CẢ VỊ TRÍ VÀ GÓC NGHIÊNG:
                # BẮT ĐẦU CHẠY BÁM LUỐNG MỚI!
                # Tầng Python chỉ ưu tiên truyền nhận: reset trạng thái về IDLE và gửi B4 = 0
                # STM32 sẽ tự động dừng 500ms (STM_QR_PAUSE_PATH_500MS) để triệt tiêu quán tính xoay,
                # sau đó tự động chuyển sang STM_QR_IDLE để chạy thẳng bám luống mới theo PID.
                self.reset()
                off_txt = f"|off|={abs(cur_offset):.0f}px≤{self.path_seek_tolerance_px:.0f}" if cur_offset is not None else ""
                ang_txt = f"|góc|={abs(cur_angle):.1f}°≤{self.path_seek_max_angle_deg:.0f}°"
                self.last_status_desc = f"🌿 ĐÃ BẮT TRÚNG TIM LUỐNG ({off_txt}, {ang_txt}): BẮT ĐẦU CHẠY BÁM LUỐNG MỚI!"

                if path_res is not None and path_res.state == TrackingState.TRACKING:
                    target_pt = path_res.path_center or path_res.box_center
                    lat_off = path_res.lateral_offset_px or 0.0
                    off_dir = path_res.offset_direction
                elif base_nav.target_pt is not None:
                    target_pt = base_nav.target_pt
                    lat_off = base_nav.lateral_offset_px or 0.0
                    off_dir = base_nav.offset_direction
                else:
                    target_pt = (cam_cx, h_img // 2)
                    lat_off = 0.0
                    off_dir = "Bám luống mới"

                return FusedNavigationTarget(
                    target_pt=target_pt,
                    lateral_offset_px=lat_off,
                    offset_direction=f"Bám luống mới ({off_dir}, Góc {cur_angle:+.1f}°)",
                    is_qr_mode=False,
                    mode_name="PATH_LINE",
                    active_qr=None,
                    ref_y=target_pt[1] if target_pt else (h_img // 2),
                    qr_stop_triggered=False,
                    uart_flag_byte=0,  # 0 = Gửi STM32 cờ 0: STM32 dừng 500ms rồi bám luống mới chạy tiếp
                    mission_state=self.state.value,
                    mission_status_desc=self.last_status_desc
                )
            else:
                # Chưa thỏa mãn điều kiện (chưa thấy luống HOẶC lệch > 200px HOẶC góc chéo > 20 độ)
                # -> Tiếp tục xoay trái tìm đường (Gửi B4 = 3)
                if cur_offset is not None:
                    desc_detail = f"|off|={abs(cur_offset):.0f}/{self.path_seek_tolerance_px:.0f}px, |góc|={abs(cur_angle):.1f}/{self.path_seek_max_angle_deg:.0f}°"
                    self.last_status_desc = f"↩️ ĐANG XOAY VÀO TIM LUỐNG ({desc_detail})..."
                else:
                    self.last_status_desc = "↩️ STM32 ĐANG XOAY TRÁI TÌM LỐI ĐI/ĐƯỜNG MỚI (Đang dò đường)..."

                return FusedNavigationTarget(
                    target_pt=(0, h_img // 2),
                    lateral_offset_px=-float(cam_cx),
                    offset_direction="STM32 đang xoay trái tìm đường (B4=3)",
                    is_qr_mode=False,
                    mode_name="TURNING_LEFT_SEEK_PATH",
                    active_qr=None,
                    ref_y=h_img // 2,
                    qr_stop_triggered=False,
                    uart_flag_byte=3,  # 3 = Đang xoay trái tìm đường
                    mission_state=self.state.value,
                    mission_status_desc=self.last_status_desc
                )

        # -------------------------------------------------------------
        # TRẠNG THÁI 5: COMPLETED_STOPPED - Đã hoàn thành nhiệm vụ & dừng hẳn
        # (Xe dừng luôn, đứng yên tuyệt đối, KHÔNG tự ý chạy bám đường lại)
        # -------------------------------------------------------------
        if self.state == QRTurnMissionState.COMPLETED_STOPPED:
            t_now = time.time() if now is None else now
            if self.completed_time is None:
                self.completed_time = t_now

            target_pt = None
            lat_off = 0.0
            off_dir = "Dừng hẳn"

            if path_res is not None and path_res.state == TrackingState.TRACKING:
                target_pt = path_res.path_center or path_res.box_center
                lat_off = path_res.lateral_offset_px or 0.0
                off_dir = path_res.offset_direction
            elif base_nav.target_pt:
                target_pt = base_nav.target_pt
                lat_off = base_nav.lateral_offset_px or 0.0
                off_dir = base_nav.offset_direction

            self.last_status_desc = "🛑 ĐÃ HOÀN THÀNH NHIỆM VỤ: DỪNG HẲN (KHÔNG CHẠY BÁM ĐƯỜNG LẠI)"
            return FusedNavigationTarget(
                target_pt=target_pt,
                lateral_offset_px=lat_off,
                offset_direction=f"Dừng hoàn tất ({off_dir})",
                is_qr_mode=False,
                mode_name="COMPLETED_STOPPED",
                active_qr=None,
                ref_y=target_pt[1] if target_pt else (h_img // 2),
                qr_stop_triggered=True,
                uart_flag_byte=1,  # 1 = Giữ lệnh dừng hẳn tuyệt đối trên STM32
                mission_state=self.state.value,
                mission_status_desc=self.last_status_desc
            )

        return base_nav

