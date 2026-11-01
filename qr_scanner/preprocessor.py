"""
Image Preprocessing and Perspective Dewarping Utilities for QR Detection.
Pure Computer Vision (NumPy + OpenCV).
"""

from typing import Tuple, Optional
import cv2
import numpy as np


def order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Sắp xếp 4 tọa độ góc theo thứ tự chuẩn:
    [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
    
    :param pts: Mảng tọa độ hình dạng (4, 2)
    :return: Mảng float32 hình dạng (4, 2) đã sắp xếp
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    # Tổng x + y: Top-Left nhỏ nhất, Bottom-Right lớn nhất
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-Left
    rect[2] = pts[np.argmax(s)]  # Bottom-Right

    # Hiệu y - x: Top-Right nhỏ nhất (x lớn, y nhỏ), Bottom-Left lớn nhất (x nhỏ, y lớn)
    diff = pts[:, 1] - pts[:, 0]
    rect[1] = pts[np.argmin(diff)]  # Top-Right
    rect[3] = pts[np.argmax(diff)]  # Bottom-Left

    return rect


def warp_perspective_qr(
    image: np.ndarray,
    pts: np.ndarray,
    margin_ratio: float = 0.08,
    target_size: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nắn phối cảnh (Perspective Dewarp) vùng mã QR về dạng ma trận vuông nhìn chính diện.
    
    :param image: Ảnh gốc (BGR hoặc Grayscale)
    :param pts: 4 điểm góc của QR (4, 2)
    :param margin_ratio: Tỷ lệ mở rộng biên (margin/padding) tránh cắt sát finder pattern
    :param target_size: Kích thước cạnh vuông mong muốn (None = tự tính từ khoảng cách góc)
    :return: (ảnh nắn phẳng dewarped_patch, ma trận chuyển đổi transform_matrix)
    """
    rect = order_corners(pts)
    (tl, tr, br, bl) = rect

    # Tính độ dài các cạnh bằng khoảng cách Euclidean
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_w = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_h = max(int(height_a), int(height_b))

    side = target_size if target_size else max(max_w, max_h, 120)

    # Thêm lề (margin) để giữ nguyên viền quiet zone chuẩn của QR
    m = int(side * margin_ratio)
    total_size = side + 2 * m

    dst = np.array([
        [m, m],
        [total_size - 1 - m, m],
        [total_size - 1 - m, total_size - 1 - m],
        [m, total_size - 1 - m]
    ], dtype=np.float32)

    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(
        image,
        transform_matrix,
        (total_size, total_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return warped, transform_matrix


def enhance_contrast_clahe(bgr_image: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
    """
    Cân bằng sáng cục bộ thích nghi (CLAHE) trên kênh L (Luminance) trong không gian màu LAB.
    Giúp tăng độ tương phản rõ rệt cho các vùng QR bị lóa hoặc thiếu sáng.
    """
    if len(bgr_image.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(bgr_image)

    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def adaptive_threshold_patch(gray_image: np.ndarray) -> np.ndarray:
    """
    Phân ngưỡng thích nghi (Adaptive Threshold) cho vùng mã QR.
    """
    if len(gray_image.shape) == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
    )
    return thresh


def compute_quad_center(pts: np.ndarray) -> Tuple[float, float]:
    """
    Tính tọa độ tâm chính xác của tứ giác mã QR (cx, cy).
    Sử dụng giao điểm 2 đường chéo (Top-Left đến Bottom-Right và Top-Right đến Bottom-Left).
    Trong hình học xạ ảnh (Projective Geometry), giao điểm 2 đường chéo là điểm bất biến
    dưới phép chiếu phối cảnh và phản ánh chính xác tâm vật lý của mã QR trong không gian thực.
    Nếu 2 đường chéo song song hoặc suy biến, tự động fallback về trọng tâm trung bình cộng 4 góc.
    """
    rect = order_corners(pts)
    tl, tr, br, bl = rect[0], rect[1], rect[2], rect[3]

    x1, y1 = float(tl[0]), float(tl[1])
    x2, y2 = float(br[0]), float(br[1])
    x3, y3 = float(tr[0]), float(tr[1])
    x4, y4 = float(bl[0]), float(bl[1])

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) > 1e-5:
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        cx = x1 + t * (x2 - x1)
        cy = y1 + t * (y2 - y1)
        return float(cx), float(cy)

    # Fallback trung bình cộng 4 góc
    cx = float(np.mean(rect[:, 0]))
    cy = float(np.mean(rect[:, 1]))
    return cx, cy


def compute_qr_geometry(pts: np.ndarray) -> dict:
    """
    Tính toán chi tiết các thông số hình học và kích thước của mã QR:
    - Tọa độ tâm pixel của mã QR (center, center_x, center_y, center_int)
    - Kích thước chiều rộng và chiều cao theo pixel (width_px, height_px)
    - Chiều dài 4 cạnh: top, bottom, left, right (px)
    - Diện tích bao phủ theo pixel (area_px) bằng công thức Shoelace
    - Tỷ lệ khung hình (aspect_ratio = width / height)
    - Góc xoay nghiêng của mã (orientation_deg)
    """
    rect = order_corners(pts)
    (tl, tr, br, bl) = rect

    cx, cy = compute_quad_center(rect)
    center = (round(cx, 1), round(cy, 1))
    center_int = (int(round(cx)), int(round(cy)))

    top_edge = float(np.linalg.norm(tr - tl))
    bottom_edge = float(np.linalg.norm(br - bl))
    left_edge = float(np.linalg.norm(bl - tl))
    right_edge = float(np.linalg.norm(br - tr))

    width_px = (top_edge + bottom_edge) / 2.0
    height_px = (left_edge + right_edge) / 2.0
    aspect_ratio = width_px / max(height_px, 1e-5)

    # Diện tích đa giác 4 điểm (công thức Shoelace)
    x = rect[:, 0]
    y = rect[:, 1]
    area_px = float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    # Góc xoay theo cạnh trên (từ -180 đến 180 độ)
    angle_rad = np.arctan2(tr[1] - tl[1], tr[0] - tl[0])
    orientation_deg = float(np.degrees(angle_rad))

    return {
        "width_px": round(width_px, 1),
        "height_px": round(height_px, 1),
        "center": center,
        "center_x": round(cx, 1),
        "center_y": round(cy, 1),
        "center_int": center_int,
        "top_edge_px": round(top_edge, 1),
        "bottom_edge_px": round(bottom_edge, 1),
        "left_edge_px": round(left_edge, 1),
        "right_edge_px": round(right_edge, 1),
        "area_px": round(area_px, 1),
        "aspect_ratio": round(aspect_ratio, 2),
        "orientation_deg": round(orientation_deg, 1)
    }

