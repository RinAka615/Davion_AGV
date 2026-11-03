"""
Module hỗ trợ đọc Camera Đo Độ Sâu (Depth Camera) & Ước lượng khoảng cách vật thể/luống đi.
Hỗ trợ dòng camera Orbbec Astra / Astra Pro qua OpenNI2 kết hợp mô hình hình học Monocular.
"""

import os
import sys
import math
import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger("DepthCamera")

# Đường dẫn thư viện OpenNI2 khả dĩ trên Linux ARM64 (Raspberry Pi 5) và x86_64
OPENNI_LIB_DIRS = [
    "/usr/lib/aarch64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu/OpenNI2/Drivers",
    "/usr/lib/x86_64-linux-gnu",
    "/tmp/openni_arm64/OpenNI_2.3.0-1.86_202210111155_4c8f5aa4_beta6_a311d/samples/bin"
]


def estimate_monocular_distance(
    y_bottom_px: float,
    img_height: int = 720,
    cam_height_m: float = 0.65,
    cam_pitch_deg: float = 25.0,
    hfov_deg: float = 60.0
) -> float:
    """
    Ước lượng khoảng cách từ robot đến mép dưới của luống bằng mô hình hình học camera Pinhole.
    
    :param y_bottom_px: Tọa độ y đáy bounding box (càng lớn tức càng gần đầu xe).
    :param img_height: Chiều cao khung hình (720px).
    :param cam_height_m: Chiều cao lắp đặt camera tính từ mặt sàn (mặc định 0.65m / 65cm).
    :param cam_pitch_deg: Góc chúc xuống của camera so với phương ngang (mặc định 25 độ).
    :param hfov_deg: Góc nhìn ngang camera (~60 độ với Astra Pro).
    :return: Khoảng cách dọc trục Z tính bằng mét (m).
    """
    if y_bottom_px is None or y_bottom_px <= 0:
        return 3.0
    
    # Góc nhìn dọc tương ứng tỉ lệ 16:9
    vfov_rad = 2.0 * math.atan(math.tan(math.radians(hfov_deg) / 2.0) * (img_height / 1280.0))
    fy = (img_height / 2.0) / math.tan(vfov_rad / 2.0)
    
    cy = img_height / 2.0
    # Góc lệch theo phương dọc so với trục quang học
    angle_offset = math.atan((y_bottom_px - cy) / fy)
    total_angle = math.radians(cam_pitch_deg) + angle_offset
    
    if total_angle <= 0.05:  # Tránh chia cho 0 khi nhìn lên chân trời
        return 5.0
    
    dist_m = cam_height_m / math.tan(total_angle)
    # Giới hạn hợp lý trong dải hoạt động của xe nhà màng (0.3m -> 6.0m)
    return float(max(0.2, min(8.0, dist_m)))


class AstraDepthCamera:
    """Quản lý luồng Depth từ cảm biến Orbbec Astra với cơ chế Fallback an toàn tuyệt đối."""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.is_opened = False
        self._dev = None
        self._stream = None
        self._openni_loaded = False
        self._latest_depth_map: Optional[np.ndarray] = None
        
        self._try_init_hardware()

    def _try_init_hardware(self):
        """Khởi tạo OpenNI2 phần cứng một cách an toàn, không làm crash app nếu lỗi USB."""
        try:
            from openni import openni2
            
            # Tìm thư mục chứa libOpenNI2 khả dụng
            target_dir = None
            for d in OPENNI_LIB_DIRS:
                if os.path.exists(d) and (os.path.exists(os.path.join(d, "libOpenNI2.so")) or os.path.exists(os.path.join(d, "libOpenNI2.so.0"))):
                    target_dir = d
                    break
            
            if target_dir:
                try:
                    openni2.initialize(target_dir)
                except Exception:
                    openni2.initialize()
            else:
                openni2.initialize()
                
            self._dev = openni2.Device.open_any()
            self._stream = self._dev.create_depth_stream()
            self._stream.start()
            self.is_opened = True
            self._openni_loaded = True
            logger.info("✅ Đã kết nối phần cứng Orbbec Astra Depth Sensor thành công!")
        except Exception as e:
            self.is_opened = False
            logger.warning(f"ℹ️ Không kích hoạt được Depth Sensor phần cứng ({e}). Sẽ sử dụng Monocular Geometry Fallback.")

    def update_frame(self) -> Optional[np.ndarray]:
        """Đọc khung hình depth mới nhất (đơn vị: mm)."""
        if not self.is_opened or self._stream is None:
            return None
        try:
            frame = self._stream.read_frame()
            data = frame.get_buffer_as_uint16()
            arr = np.frombuffer(data, dtype=np.uint16).reshape((frame.height, frame.width))
            self._latest_depth_map = arr
            return arr
        except Exception as e:
            logger.debug(f"Lỗi đọc depth frame: {e}")
            return None

    def get_distance_m(
        self,
        bounding_box: Optional[Tuple[int, int, int, int]],
        img_width: int = 1280,
        img_height: int = 720
    ) -> Tuple[float, str]:
        """
        Lấy khoảng cách tới đối tượng/luống (đơn vị: mét).
        
        :param bounding_box: (bx, by, bw, bh) trên ảnh màu RGB.
        :param img_width: Bề rộng ảnh màu (1280px).
        :param img_height: Chiều cao ảnh màu (720px).
        :return: (khoảng_cách_mét, nguồn_đo: 'Depth Cam' hoặc 'Hình học')
        """
        if bounding_box is None:
            return 2.0, "Mặc định"
            
        bx, by, bw, bh = bounding_box
        y_bottom = float(by + bh)

        # 1. Nếu có dữ liệu từ Depth Camera phần cứng:
        if self._latest_depth_map is not None:
            # Map tọa độ bounding box từ ảnh RGB sang khung Depth (thường 640x480)
            dh, dw = self._latest_depth_map.shape
            scale_x = dw / float(img_width)
            scale_y = dh / float(img_height)
            
            dx1 = max(0, int(bx * scale_x))
            dy1 = max(0, int(by * scale_y))
            dx2 = min(dw, int((bx + bw) * scale_x))
            dy2 = min(dh, int((by + bh) * scale_y))
            
            if dx2 > dx1 and dy2 > dy1:
                roi = self._latest_depth_map[dy1:dy2, dx1:dx2]
                valid_mask = (roi > 200) & (roi < 8000)  # Dải hiệu dụng Astra (0.2m -> 8m)
                if np.any(valid_mask):
                    median_mm = float(np.median(roi[valid_mask]))
                    dist_m = round(median_mm / 1000.0, 2)
                    return dist_m, "Depth Cam"

        # 2. Fallback sang tính toán hình học Monocular:
        geom_dist = estimate_monocular_distance(y_bottom_px=y_bottom, img_height=img_height)
        return round(geom_dist, 2), "Hình học"

    def stop(self):
        """Dừng luồng camera đo độ sâu an toàn."""
        if self.is_opened:
            try:
                if self._stream:
                    self._stream.stop()
                from openni import openni2
                openni2.unload()
            except Exception:
                pass
            self.is_opened = False
