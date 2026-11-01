"""
Core Multi-backend QR Code Detection and Decoding Engine.
Supports ZXing-C++ and OpenCV QRCodeDetector with perspective dewarping.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

try:
    import zxingcpp
    HAS_ZXING = True
except ImportError:
    HAS_ZXING = False

from .preprocessor import order_corners, warp_perspective_qr, enhance_contrast_clahe, adaptive_threshold_patch, compute_qr_geometry


@dataclass
class QRDetectionResult:
    """Kết quả phát hiện và giải mã một mã QR trong khung hình."""
    points: np.ndarray             # 4 điểm góc [[x, y], ...] trên ảnh gốc, shape (4, 2)
    text: str                      # Chuỗi văn bản đã giải mã (rỗng nếu chưa giải mã được)
    is_decoded: bool               # True nếu đã giải mã thành công; False nếu chỉ là candidate
    error: Optional[str] = None    # Thông báo lỗi (checksum, unreadable, etc.) nếu is_decoded=False
    backend: str = "zxing-cpp"     # Tên backend đã xử lý ("zxing-cpp" hoặc "opencv-qr")
    format_name: str = "QRCode"    # Định dạng mã (QRCode, MicroQRCode, etc.)
    dewarped_patch: Optional[np.ndarray] = None  # Ảnh đã nắn thẳng vuông góc
    
    # --- CÁC THÔNG SỐ ĐO LƯỜNG KÍCH THƯỚC VÀ HÌNH HỌC ---
    center: Tuple[float, float] = (0.0, 0.0)      # Tọa độ tâm pixel của mã QR (cx, cy)
    center_int: Tuple[int, int] = (0, 0)          # Tọa độ tâm làm tròn nguyên (cx, cy)
    width_px: float = 0.0          # Chiều rộng trung bình (pixels)
    height_px: float = 0.0         # Chiều cao trung bình (pixels)
    area_px: float = 0.0           # Diện tích bao phủ (pixels^2)
    aspect_ratio: float = 1.0      # Tỷ lệ width / height
    orientation_deg: float = 0.0   # Góc xoay nghiêng (độ)
    geometry: Optional[dict] = None # Chi tiết độ dài từng cạnh: top, bottom, left, right


class QRDetector:
    """
    Bộ phát hiện và giải mã mã QR đa tầng:
    1. Tầng 1: ZXing-C++ (Tốc độ cao ~100-200 FPS, hỗ trợ đa mã đồng thời, chịu biến dạng và nghiêng).
    2. Tầng 2: Nắn phối cảnh (Perspective Dewarping) + CLAHE / Phân ngưỡng đối với các vùng lỗi/khó đọc.
    3. Tầng 3: OpenCV QRCodeDetector (Dự phòng độc lập hoặc khi chỉ định backend='opencv').
    """

    def __init__(self, backend: str = "auto", enable_dewarp: bool = True):
        """
        :param backend: "auto", "zxing", hoặc "opencv"
        :param enable_dewarp: Cho phép tự động nắn thẳng vùng QR để tăng tỉ lệ đọc
        """
        self.backend = backend.lower()
        self.enable_dewarp = enable_dewarp

        # Khởi tạo OpenCV QRCodeDetector
        self.cv_detector = cv2.QRCodeDetector()

        # Cấu hình định dạng quét cho ZXing-C++
        if HAS_ZXING:
            self.zxing_formats = [
                zxingcpp.BarcodeFormat.QRCode,
                zxingcpp.BarcodeFormat.MicroQRCode,
                zxingcpp.BarcodeFormat.RMQRCode
            ]
        else:
            self.zxing_formats = []

    def detect_and_decode(
        self,
        frame: np.ndarray,
        confidence: float = 0.5,
        include_other_formats: bool = False
    ) -> List[QRDetectionResult]:
        """
        Xử lý toàn diện một khung hình: phát hiện tất cả mã QR, nắn thẳng và giải mã.
        
        :param frame: Khung hình BGR từ camera hoặc video
        :param confidence: Ngưỡng độ tin cậy từ 0.0 đến 1.0 (mặc định 0.5).
                           Càng giảm độ tin cậy xuống thấp (ví dụ < 0.3), hệ thống càng quét sâu
                           đa tầng (CLAHE, phóng to, đa bộ phân ngưỡng, hiển thị vùng nghi vấn).
        :param include_other_formats: Cho phép quét thêm DataMatrix và các mã 2D khác nếu bật.
        :return: Danh sách các đối tượng QRDetectionResult
        """
        results: List[QRDetectionResult] = []

        # Cấu hình danh sách định dạng quét
        target_formats = list(self.zxing_formats)
        if include_other_formats and HAS_ZXING:
            if hasattr(zxingcpp.BarcodeFormat, "DataMatrix"):
                target_formats.append(zxingcpp.BarcodeFormat.DataMatrix)

        if self.backend in ("auto", "zxing") and HAS_ZXING:
            results = self._detect_zxing(frame, target_formats=target_formats, confidence=confidence)

        # Nếu backend là "opencv" hoặc nếu zxing không tìm thấy mã nào và mode là "auto",
        # thử bổ sung bằng OpenCV QRCodeDetector
        if not results and self.backend in ("auto", "opencv"):
            results = self._detect_opencv(frame)

        # Lọc kết quả dựa trên độ tin cậy (confidence)
        # Nếu confidence cao (> 0.6), loại bỏ các candidate chưa giải mã được (chế độ nghiêm ngặt)
        # Nếu confidence thấp (<= 0.6), giữ lại các candidate để người dùng quan sát vị trí
        if confidence > 0.6:
            results = [r for r in results if r.is_decoded]

        return results

    def _detect_zxing(
        self,
        frame: np.ndarray,
        target_formats: Optional[List] = None,
        confidence: float = 0.5
    ) -> List[QRDetectionResult]:
        results: List[QRDetectionResult] = []
        formats = target_formats if target_formats is not None else self.zxing_formats

        # 1. Quét lượt 1: Quét tiêu chuẩn
        try:
            barcodes = zxingcpp.read_barcodes(
                frame,
                formats=formats,
                try_rotate=True,
                try_downscale=True,
                try_invert=True,
                return_errors=True
            )
        except Exception:
            barcodes = []

        # 2. Nếu giảm độ tin cậy xuống thấp (confidence <= 0.35) và chưa thấy mã giải mã thành công:
        # Kích hoạt quét đa tầng tăng cường (CLAHE và Upscale)
        has_decoded = any(b.valid and b.text for b in barcodes)
        if not has_decoded and confidence <= 0.35:
            # Thử tăng cường tương phản CLAHE
            try:
                enhanced_frame = enhance_contrast_clahe(frame, clip_limit=3.0)
                extra_barcodes = zxingcpp.read_barcodes(
                    enhanced_frame,
                    formats=formats,
                    try_rotate=True,
                    try_downscale=True,
                    try_invert=True,
                    return_errors=True
                )
                if extra_barcodes:
                    barcodes = extra_barcodes
            except Exception:
                pass

        for b in barcodes:
            pos = b.position
            # Bỏ qua nếu không xác định được tọa độ góc (trường hợp format error trước khi định vị)
            if (pos.top_left.x == 0 and pos.top_left.y == 0 and
                pos.bottom_right.x == 0 and pos.bottom_right.y == 0):
                continue

            # 4 góc theo thứ tự: top_left, top_right, bottom_right, bottom_left
            pts = np.array([
                [pos.top_left.x, pos.top_left.y],
                [pos.top_right.x, pos.top_right.y],
                [pos.bottom_right.x, pos.bottom_right.y],
                [pos.bottom_left.x, pos.bottom_left.y]
            ], dtype=np.float32)

            pts_ordered = order_corners(pts)
            dewarped = None
            if self.enable_dewarp:
                try:
                    dewarped, _ = warp_perspective_qr(frame, pts_ordered)
                except Exception:
                    dewarped = None

            fmt_str = str(b.format).split(".")[-1]
            geom = compute_qr_geometry(pts_ordered)

            if b.valid and b.text and b.text.strip():
                # Giải mã thành công
                results.append(QRDetectionResult(
                    points=pts_ordered,
                    text=b.text.strip(),
                    is_decoded=True,
                    error=None,
                    backend="zxing-cpp",
                    format_name=fmt_str,
                    dewarped_patch=dewarped,
                    center=geom["center"],
                    center_int=geom["center_int"],
                    width_px=geom["width_px"],
                    height_px=geom["height_px"],
                    area_px=geom["area_px"],
                    aspect_ratio=geom["aspect_ratio"],
                    orientation_deg=geom["orientation_deg"],
                    geometry=geom
                ))
            else:
                # Tìm thấy vùng QR (Candidate) nhưng giải mã chưa thành công
                # Thử cứu mã bằng cách tiền xử lý trên ảnh đã nắn thẳng (dewarped patch)
                recovered_text = None
                if dewarped is not None:
                    recovered_text = self._try_recover_dewarped(dewarped)

                if recovered_text:
                    results.append(QRDetectionResult(
                        points=pts_ordered,
                        text=recovered_text,
                        is_decoded=True,
                        error=None,
                        backend="zxing-cpp(dewarped)",
                        format_name=fmt_str,
                        dewarped_patch=dewarped,
                        center=geom["center"],
                        center_int=geom["center_int"],
                        width_px=geom["width_px"],
                        height_px=geom["height_px"],
                        area_px=geom["area_px"],
                        aspect_ratio=geom["aspect_ratio"],
                        orientation_deg=geom["orientation_deg"],
                        geometry=geom
                    ))
                else:
                    err_msg = str(b.error) if b.error else "Checksum/Decode Error"
                    results.append(QRDetectionResult(
                        points=pts_ordered,
                        text="",
                        is_decoded=False,
                        error=err_msg,
                        backend="zxing-cpp",
                        format_name=fmt_str,
                        dewarped_patch=dewarped,
                        center=geom["center"],
                        center_int=geom["center_int"],
                        width_px=geom["width_px"],
                        height_px=geom["height_px"],
                        area_px=geom["area_px"],
                        aspect_ratio=geom["aspect_ratio"],
                        orientation_deg=geom["orientation_deg"],
                        geometry=geom
                    ))

        return results

    def _try_recover_dewarped(self, patch: np.ndarray) -> Optional[str]:
        """
        Thử giải mã lại trên ảnh nắn thẳng vuông góc bằng các phương pháp tiền xử lý có chọn lọc:
        1. Ảnh patch gốc
        2. CLAHE tăng tương phản
        3. Phân ngưỡng thích nghi (Adaptive Threshold)
        """
        if not HAS_ZXING:
            return None

        # 1. Thử trực tiếp trên patch
        try:
            res = zxingcpp.read_barcodes(patch, formats=self.zxing_formats, is_pure=False)
            for r in res:
                if r.valid and r.text and r.text.strip():
                    return r.text.strip()
        except Exception:
            pass

        # 2. Thử CLAHE tăng độ tương phản
        try:
            enhanced = enhance_contrast_clahe(patch, clip_limit=3.0)
            res = zxingcpp.read_barcodes(enhanced, formats=self.zxing_formats, is_pure=False)
            for r in res:
                if r.valid and r.text and r.text.strip():
                    return r.text.strip()
        except Exception:
            pass

        # 3. Thử Adaptive Threshold
        try:
            thresh = adaptive_threshold_patch(patch)
            res = zxingcpp.read_barcodes(thresh, formats=self.zxing_formats, is_pure=False)
            for r in res:
                if r.valid and r.text and r.text.strip():
                    return r.text.strip()
        except Exception:
            pass

        return None

    def _detect_opencv(self, frame: np.ndarray) -> List[QRDetectionResult]:
        results: List[QRDetectionResult] = []
        try:
            # OpenCV QRCodeDetector.detectMulti hoặc detect
            if hasattr(self.cv_detector, "detectMulti"):
                ret_val, points = self.cv_detector.detectMulti(frame)
            else:
                ret_val, pts = self.cv_detector.detect(frame)
                points = [pts] if ret_val and pts is not None else []
        except Exception:
            return results

        if not ret_val or points is None or len(points) == 0:
            return results

        for pts in points:
            if pts is None or len(pts) == 0:
                continue
            pts_ordered = order_corners(pts.reshape(4, 2))
            dewarped = None
            if self.enable_dewarp:
                try:
                    dewarped, _ = warp_perspective_qr(frame, pts_ordered)
                except Exception:
                    dewarped = None

            # Thử decode bằng OpenCV
            decoded_text = ""
            try:
                decode_res = self.cv_detector.decode(frame, pts_ordered)
                if isinstance(decode_res, (tuple, list)):
                    decoded_text = decode_res[0]
                elif isinstance(decode_res, str):
                    decoded_text = decode_res
            except Exception:
                decoded_text = ""

            geom = compute_qr_geometry(pts_ordered)
            if decoded_text and decoded_text.strip():
                results.append(QRDetectionResult(
                    points=pts_ordered,
                    text=decoded_text.strip(),
                    is_decoded=True,
                    error=None,
                    backend="opencv-qr",
                    format_name="QRCode",
                    dewarped_patch=dewarped,
                    center=geom["center"],
                    center_int=geom["center_int"],
                    width_px=geom["width_px"],
                    height_px=geom["height_px"],
                    area_px=geom["area_px"],
                    aspect_ratio=geom["aspect_ratio"],
                    orientation_deg=geom["orientation_deg"],
                    geometry=geom
                ))
            else:
                # Nếu decode ảnh gốc thất bại, thử decode trên dewarped patch
                recovered = None
                if dewarped is not None:
                    try:
                        dec_patch = self.cv_detector.decode(dewarped, np.array([
                            [0, 0], [dewarped.shape[1], 0],
                            [dewarped.shape[1], dewarped.shape[0]], [0, dewarped.shape[0]]
                        ], dtype=np.float32))
                        if isinstance(dec_patch, (tuple, list)) and dec_patch[0]:
                            recovered = dec_patch[0]
                    except Exception:
                        pass

                if recovered and recovered.strip():
                    results.append(QRDetectionResult(
                        points=pts_ordered,
                        text=recovered.strip(),
                        is_decoded=True,
                        error=None,
                        backend="opencv-qr(dewarped)",
                        format_name="QRCode",
                        dewarped_patch=dewarped,
                        center=geom["center"],
                        center_int=geom["center_int"],
                        width_px=geom["width_px"],
                        height_px=geom["height_px"],
                        area_px=geom["area_px"],
                        aspect_ratio=geom["aspect_ratio"],
                        orientation_deg=geom["orientation_deg"],
                        geometry=geom
                    ))
                else:
                    results.append(QRDetectionResult(
                        points=pts_ordered,
                        text="",
                        is_decoded=False,
                        error="OpenCV Decode Failed",
                        backend="opencv-qr",
                        format_name="QRCode",
                        dewarped_patch=dewarped,
                        center=geom["center"],
                        center_int=geom["center_int"],
                        width_px=geom["width_px"],
                        height_px=geom["height_px"],
                        area_px=geom["area_px"],
                        aspect_ratio=geom["aspect_ratio"],
                        orientation_deg=geom["orientation_deg"],
                        geometry=geom
                    ))

        return results
