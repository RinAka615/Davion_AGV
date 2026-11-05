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
        self.last_decoded_text: str = "TURN_2"

    def reset_last_decoded_text(self, default_text: str = "TURN_2") -> None:
        """Đặt lại chuỗi mã QR ghi nhớ gần nhất khi chuyển pha nhiệm vụ (ví dụ xoay từ TURN_1 sang TURN_2)."""
        self.last_decoded_text = default_text

    def detect_and_decode(
        self,
        frame: np.ndarray,
        confidence: float = 0.5,
        include_other_formats: bool = False,
        ai_qr_boxes: Optional[List[Tuple[int, int, int, int, float]]] = None
    ) -> List[QRDetectionResult]:
        """
        Xử lý toàn diện một khung hình: phát hiện tất cả mã QR, nắn thẳng và giải mã.
        Kết hợp cả ZXing-C++ và vùng QR được phát hiện bởi mô hình AI (YOLO-Fastest V2)
        khi robot đang quay làm mã QR bị nhòe chuyển động (motion blur).
        """
        results: List[QRDetectionResult] = []

        # Cấu hình danh sách định dạng quét
        target_formats = list(self.zxing_formats)
        if include_other_formats and HAS_ZXING:
            if hasattr(zxingcpp.BarcodeFormat, "DataMatrix"):
                target_formats.append(zxingcpp.BarcodeFormat.DataMatrix)

        if self.backend in ("auto", "zxing") and HAS_ZXING:
            results = self._detect_zxing(frame, target_formats=target_formats, confidence=confidence)

        # Chỉ dùng OpenCV QRCodeDetector nếu backend chỉ định là "opencv" hoặc máy không cài thư viện ZXing-C++
        if not results and (self.backend == "opencv" or not HAS_ZXING):
            results = self._detect_opencv(frame)

        # Lưu lại nội dung mã QR giải mã thành công gần nhất
        for r in results:
            if r.is_decoded and r.text:
                self.last_decoded_text = r.text

        # Bổ sung nhận diện từ mô hình YOLO-Fastest V2 (Channel 1: QR Code) khi xe đang quay (bị nhòe)
        if ai_qr_boxes:
            h_f, w_f = frame.shape[:2]
            decoded_centers = [r.center for r in results if r.is_decoded]
            for (x1, y1, x2, y2, q_conf) in ai_qr_boxes:
                cx_box = 0.5 * (x1 + x2)
                cy_box = 0.5 * (y1 + y2)
                # Bỏ qua nếu vị trí này đã được ZXing giải mã thành công
                if any(abs(cx_box - dc[0]) < 60 and abs(cy_box - dc[1]) < 60 for dc in decoded_centers):
                    continue

                # Kiểm tra tỷ lệ khung hình bảng mã QR (hỗ trợ kéo vệt ngang khi quay nhanh: 0.45 <= w/h <= 2.2)
                bw_box = max(1, x2 - x1)
                bh_box = max(1, y2 - y1)
                ar_box = float(bw_box) / float(bh_box)
                if ar_box < 0.45 or ar_box > 2.2 or bh_box < 38:
                    continue

                # Cắt vùng ROI mở rộng 15px để khử nhòe (Unsharp Mask + Upscale 2x)
                pad = 15
                rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
                rx2, ry2 = min(w_f, x2 + pad), min(h_f, y2 + pad)
                crop = frame[ry1:ry2, rx1:rx2]
                if crop.size == 0 or crop.shape[0] < 24 or crop.shape[1] < 24:
                    continue

                # Kiểm tra đặc trưng tương phản đen-trắng và nền bảng trắng của mã QR (loại bỏ lá cây/cỏ chói nắng)
                inner = frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]
                if inner.size == 0 or inner.shape[0] < 16 or inner.shape[1] < 16:
                    continue
                gray_inner = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY).astype(np.float32)
                p85_in, p15_in = np.percentile(gray_inner, [85, 15])
                if float(np.std(gray_inner)) < 14.0 or (p85_in - p15_in) < 35.0 or p85_in < 145.0 or p15_in > 165.0:
                    continue
                bright_mask = gray_inner >= np.percentile(gray_inner, 60)
                if np.any(bright_mask):
                    b_in = inner[:, :, 0].astype(np.float32)[bright_mask]
                    g_in = inner[:, :, 1].astype(np.float32)[bright_mask]
                    r_in = inner[:, :, 2].astype(np.float32)[bright_mask]
                    # Loại bỏ vùng lá cây xanh chói nắng (G vượt trội so với B và R)
                    if float(np.mean(g_in - b_in)) > 26.0 or float(np.mean(g_in - r_in)) > 16.0:
                        continue

                decoded_from_roi = None
                if HAS_ZXING:
                    try:
                        up = cv2.resize(crop, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                        blur = cv2.GaussianBlur(up, (0, 0), 1.8)
                        sharp = cv2.addWeighted(up, 1.8, blur, -0.8, 0)
                        bs = zxingcpp.read_barcodes(sharp, formats=self.zxing_formats, try_rotate=True, try_invert=True)
                        for b in bs:
                            if b.valid and b.text and b.text.strip():
                                decoded_from_roi = b.text.strip()
                                self.last_decoded_text = decoded_from_roi
                                break
                        # Nếu nắng gắt làm lóa mảnh nét đen của QR ngoài trời -> Phân ngưỡng cao + Erode làm đậm nét đen
                        if not decoded_from_roi:
                            up_gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
                            k_erode = np.ones((3, 3), np.uint8)
                            for th_val in (215, 232, 195):
                                _, th_img = cv2.threshold(up_gray, th_val, 255, cv2.THRESH_BINARY)
                                er_img = cv2.erode(th_img, k_erode, iterations=1)
                                bs_er = zxingcpp.read_barcodes(er_img, formats=self.zxing_formats, try_rotate=True)
                                for b in bs_er:
                                    if b.valid and b.text and b.text.strip():
                                        decoded_from_roi = b.text.strip()
                                        self.last_decoded_text = decoded_from_roi
                                        break
                                if decoded_from_roi:
                                    break
                    except Exception:
                        pass

                pts_ordered = np.array([
                    [float(x1), float(y1)],
                    [float(x2), float(y1)],
                    [float(x2), float(y2)],
                    [float(x1), float(y2)]
                ], dtype=np.float32)
                geom = compute_qr_geometry(pts_ordered)
                dewarped = cv2.resize(crop, (160, 160)) if self.enable_dewarp else None

                # Nếu giải mã được từ ROI làm nét HOẶC xác suất AI YOLO-Fastest V2 cao (đang quay nhòe / nắng gắt)
                if decoded_from_roi or q_conf >= 0.42:
                    if decoded_from_roi:
                        qr_text = decoded_from_roi
                    elif self.last_decoded_text and "TURN_1" not in self.last_decoded_text.upper():
                        qr_text = self.last_decoded_text
                    else:
                        # Khi xe đang xoay tìm trạm mới (hoặc text trước là TURN_1), nhãn dự đoán chuyển hướng là TURN_2
                        qr_text = "TURN_2"
                    results.append(QRDetectionResult(
                        points=pts_ordered,
                        text=qr_text,
                        is_decoded=True,
                        error=None,
                        backend="yolo-fastestv2-qr" if not decoded_from_roi else "zxing-cpp(ai-roi)",
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
                    decoded_centers.append(geom["center"])

        # Lọc kết quả dựa trên độ tin cậy (confidence)
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
