"""
Module for sensor fusion between Vision Line Tracking (YOLOv8 Segmentation)
and QR Code Checkpoint Alignment.

When QR code bounding area exceeds threshold (default 60x60 px),
navigation seamlessly switches from path centerline to QR center alignment.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

from qr_scanner.detector import QRDetectionResult
from src.path_detector import DetectionResult, TrackingState


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
    uart_flag_byte: int = 0                   # Byte 4 gửi STM32: 1 nếu QR > 200x200, 0 cho mọi trường hợp còn lại


def check_qr_exceeds_threshold(
    qr: QRDetectionResult,
    min_dim: float = 60.0,
    min_area: Optional[float] = None
) -> Tuple[bool, float, float, float]:
    """
    Kiểm tra xem diện tích đường bao của mã QR có lớn hơn ngưỡng (mặc định > 60x60 = 3,600 px^2) không.
    
    :param qr: Kết quả phát hiện mã QR
    :param min_dim: Kích thước chiều dài/rộng tối thiểu (mặc định 60.0 px)
    :param min_area: Diện tích tối thiểu (nếu None, tính bằng min_dim * min_dim = 3,600 px^2)
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

    # Điều kiện thỏa mãn diện tích đường bao > 60x60:
    # 1) Diện tích >= 3,600 px^2
    # 2) HOẶC cả hai chiều rộng và cao của bbox >= 60 px
    # 3) HOẶC kích thước đo đạc width_px >= 60 và height_px >= 60
    is_exceeded = (
        effective_area >= min_area
        or (bw >= min_dim and bh >= min_dim)
        or (qr.width_px >= min_dim and qr.height_px >= min_dim)
    )

    return is_exceeded, round(bw, 1), round(bh, 1), round(effective_area, 1)


def fuse_navigation_target(
    path_res: Optional[DetectionResult],
    qr_results: List[QRDetectionResult],
    frame_shape: Tuple[int, int],
    min_qr_size: float = 60.0,
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
      + Nếu phát hiện mã QR có diện tích đường bao > 60x60:
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

    # 2. Tìm kiếm mã QR lớn nhất thỏa mãn điều kiện diện tích đường bao > min_qr_size (mặc định > 60x60)
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

