#!/usr/bin/env python3
"""
Ứng dụng Phát hiện, Nắn thẳng (Perspective Dewarp) và Giải mã QR Code từ Video
Sử dụng hoàn toàn Thị giác máy tính cổ điển (Classical Computer Vision - OpenCV + PyQt5).
"""

import os
import sys
import time
import argparse
from typing import Optional, Tuple, Dict, Any, List

import cv2

# Khắc phục lỗi xung đột Qt Platform Plugin giữa opencv-python và PyQt5 trên Linux
import PyQt5
_pyqt5_plugins = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
if os.path.exists(_pyqt5_plugins):
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _pyqt5_plugins

if os.environ.get("XDG_SESSION_TYPE") == "wayland" and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "wayland;xcb"

import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QGroupBox, QSplitter,
    QMessageBox, QComboBox, QLineEdit
)


# ==============================================================================
# PHẦN 1: THUẬT TOÁN THỊ GIÁC MÁY TÍNH CỔ ĐIỂN (CLASSICAL COMPUTER VISION)
# ==============================================================================

def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Sắp xếp 4 đỉnh tứ giác theo thứ tự: [TL, TR, BR, BL]."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-Left (tổng x+y nhỏ nhất)
    rect[2] = pts[np.argmax(s)]  # Bottom-Right (tổng x+y lớn nhất)

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-Right (hiệu y-x nhỏ nhất)
    rect[3] = pts[np.argmax(diff)]  # Bottom-Left (hiệu y-x lớn nhất)
    return rect


def preprocess_frame(frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bước 1 & 2: Tiền xử lý ảnh xám, khử nhiễu và phân ngưỡng thích nghi cục bộ."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Thresholding xử lý bóng đổ và ánh sáng không đồng đều
    thresh_inv = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 7
    )
    thresh_bin = cv2.bitwise_not(thresh_inv)
    return gray, thresh_inv, thresh_bin


def get_contour_centroid(contour: np.ndarray) -> Optional[Tuple[float, float]]:
    """Tính trọng tâm (Centroid) đường bao."""
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    return (float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"]))


def find_finder_patterns(thresh_inv: np.ndarray) -> List[Dict[str, Any]]:
    """
    Bước 3.1: Tìm 3 cụm Finder Patterns dựa trên cấu trúc đường bao 3 cấp lồng nhau
    với tỷ lệ diện tích 49:25:9 (tương ứng tỷ lệ bề rộng 1:1:3:1:1).
    """
    contours, hierarchy = cv2.findContours(thresh_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        return []

    hierarchy = hierarchy[0]
    finders = []

    for i in range(len(contours)):
        child = hierarchy[i][2]
        if child == -1:
            continue
        grandchild = hierarchy[child][2]
        if grandchild == -1:
            continue

        pt_outer = get_contour_centroid(contours[i])
        pt_mid = get_contour_centroid(contours[child])
        pt_inner = get_contour_centroid(contours[grandchild])
        if not pt_outer or not pt_mid or not pt_inner:
            continue

        # Kiểm tra độ lệch tâm giữa 3 đường bao đồng tâm (< 5.0 pixel)
        if np.hypot(pt_outer[0] - pt_mid[0], pt_outer[1] - pt_mid[1]) > 5.0:
            continue
        if np.hypot(pt_mid[0] - pt_inner[0], pt_mid[1] - pt_inner[1]) > 5.0:
            continue

        x, y, w, h = cv2.boundingRect(contours[i])
        aspect = float(w) / max(1, h)
        if not (0.70 <= aspect <= 1.30):
            continue

        area_outer = cv2.contourArea(contours[i])
        area_mid = cv2.contourArea(contours[child])
        area_inner = cv2.contourArea(contours[grandchild])

        if area_outer < 50:
            continue

        r_inner = area_inner / area_outer  # Lý thuyết 9/49 ~ 0.1836
        r_mid = area_mid / area_outer      # Lý thuyết 25/49 ~ 0.5102

        if 0.08 <= r_inner <= 0.32 and 0.32 <= r_mid <= 0.78:
            finders.append({
                "center": np.array(pt_outer, dtype=np.float32),
                "area": area_outer,
                "size_px": float(np.sqrt(area_outer))
            })

    return finders


def identify_corners_and_quad(finders: List[Dict[str, Any]]) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Bước 3.2 & 4.1: Xác định TL, TR, BL bằng tích có hướng 2D & Định lý Pytago;
    suy diễn 4 góc ngoài kèm vùng đệm Quiet Zone.
    """
    if len(finders) < 3:
        return None

    best_triplet = None
    best_score = float("inf")
    n = len(finders)

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p0, p1, p2 = finders[i]["center"], finders[j]["center"], finders[k]["center"]
                d01 = float(np.linalg.norm(p0 - p1))
                d12 = float(np.linalg.norm(p1 - p2))
                d20 = float(np.linalg.norm(p2 - p0))

                sides = sorted([(d01, p2, p0, p1), (d12, p0, p1, p2), (d20, p1, p2, p0)], key=lambda x: x[0])
                hyp, opp_hyp, a_pt, b_pt = sides[2]
                leg1, leg2 = sides[0][0], sides[1][0]

                leg_diff = abs(leg1 / max(1e-4, leg2) - 1.0)
                pyth_diff = abs((leg1**2 + leg2**2) - hyp**2) / (hyp**2 + 1e-4)
                score = leg_diff + pyth_diff

                if score < best_score:
                    best_score = score
                    tl_pt = opp_hyp
                    v_a = a_pt - tl_pt
                    v_b = b_pt - tl_pt
                    cross = v_a[0] * v_b[1] - v_a[1] * v_b[0]
                    if cross > 0:
                        tr_pt, bl_pt = a_pt, b_pt
                    else:
                        tr_pt, bl_pt = b_pt, a_pt
                    avg_size = (finders[i]["size_px"] + finders[j]["size_px"] + finders[k]["size_px"]) / 3.0
                    best_triplet = (tl_pt, tr_pt, bl_pt, avg_size)

    if best_triplet is None or best_score > 0.65:
        return None

    tl, tr, bl, pattern_size = best_triplet
    module_size = pattern_size / 7.0

    vec_u = tr - tl
    vec_v = bl - tl
    len_u = np.linalg.norm(vec_u)
    len_v = np.linalg.norm(vec_v)
    if len_u < 1e-4 or len_v < 1e-4:
        return None

    u_norm = vec_u / len_u
    v_norm = vec_v / len_v
    br = tr + bl - tl
    offset = (3.5 + 2.5) * module_size  # Kèm 2.5 module Quiet Zone

    c_tl = tl - offset * u_norm - offset * v_norm
    c_tr = tr + offset * u_norm - offset * v_norm
    c_bl = bl - offset * u_norm + offset * v_norm
    c_br = br + offset * u_norm + offset * v_norm

    quad_corners = np.float32([c_tl, c_tr, c_br, c_bl])
    return tl, tr, bl, quad_corners


def warp_perspective_qr(gray_img: np.ndarray, quad_corners: np.ndarray, target_size: int = 360) -> np.ndarray:
    """Bước 4.2: Nắn phối cảnh về hình vuông kích thước target_size x target_size."""
    dst = np.float32([
        [0, 0],
        [target_size - 1, 0],
        [target_size - 1, target_size - 1],
        [0, target_size - 1]
    ])
    M = cv2.getPerspectiveTransform(quad_corners, dst)
    return cv2.warpPerspective(gray_img, M, (target_size, target_size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def decode_qr(warped_gray: np.ndarray, detector: cv2.QRCodeDetector) -> Tuple[bool, str]:
    """Bước 5: Giải mã mã QR từ ảnh đã nắn thẳng."""
    variants = [
        warped_gray,
        cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5),
        cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    ]
    for var in variants:
        try:
            text, _, _ = detector.detectAndDecode(var)
            if text and len(text.strip()) > 0:
                return True, text.strip()
        except Exception:
            pass

    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
        for var in variants:
            objs = pyzbar_decode(var)
            for o in objs:
                txt = o.data.decode("utf-8")
                if txt:
                    return True, txt.strip()
    except ImportError:
        pass

    return False, ""


# ==============================================================================
# PHẦN 2: LUỒNG XỬ LÝ VIDEO & WORKER THREAD
# ==============================================================================

class VideoProcessingWorker(QThread):
    """Worker đa luồng đọc video và chạy pipeline thị giác máy tính theo thời gian thực."""
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, bool, str, float, int, int)
    playback_finished = pyqtSignal()

    def __init__(self, source_path_or_idx):
        super().__init__()
        self.source = source_path_or_idx
        self.is_running = False
        self.is_paused = False
        self.speed = 1.0
        self.seek_requested_frame = -1

    def run(self):
        self.is_running = True
        detector = cv2.QRCodeDetector()
        cap = cv2.VideoCapture(int(self.source) if str(self.source).isdigit() else str(self.source))
        if not cap.isOpened():
            print(f"[Worker] Không thể mở nguồn video: {self.source}")
            self.playback_finished.emit()
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1

        last_decoded_text = ""

        while self.is_running:
            if self.seek_requested_frame >= 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_requested_frame)
                self.seek_requested_frame = -1

            if self.is_paused:
                time.sleep(0.04)
                continue

            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret or frame is None:
                # Video lặp lại nếu chạy hết file
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # 1. Pipeline thị giác máy tính
            gray, thresh_inv, thresh_bin = preprocess_frame(frame)
            finders = find_finder_patterns(thresh_inv)
            res = identify_corners_and_quad(finders)

            warped = np.zeros((360, 360), dtype=np.uint8)
            annotated = frame.copy()
            detected = False

            if res is not None:
                tl, tr, bl, quad = res
                detected = True
                warped = warp_perspective_qr(gray, quad, target_size=360)
                ok, text = decode_qr(warped, detector)
                if ok:
                    last_decoded_text = text

                # Vẽ overlay
                pts_i = quad.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts_i], True, (0, 255, 0), 3)
                cv2.circle(annotated, (int(tl[0]), int(tl[1])), 8, (0, 0, 255), -1)   # TL: Đỏ
                cv2.putText(annotated, "TL", (int(tl[0]) - 20, int(tl[1]) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.circle(annotated, (int(tr[0]), int(tr[1])), 8, (255, 0, 0), -1)   # TR: Xanh
                cv2.putText(annotated, "TR", (int(tr[0]) + 10, int(tr[1]) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.circle(annotated, (int(bl[0]), int(bl[1])), 8, (0, 255, 255), -1) # BL: Vàng
                cv2.putText(annotated, "BL", (int(bl[0]) - 20, int(bl[1]) + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # Fallback tìm viền tứ giác giấy
                quad_paper = order_quad_points(np.zeros((4, 2), dtype=np.float32))

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Gửi frame lên GUI
            self.frame_ready.emit(
                annotated, warped, thresh_inv, detected, last_decoded_text,
                elapsed_ms, current_frame_idx, total_frames
            )

            # Pacing tốc độ phát
            period = 1.0 / max(1.0, (fps * self.speed))
            rem = period - (time.perf_counter() - t0)
            if rem > 0.002:
                time.sleep(rem)

        cap.release()

    def stop(self):
        self.is_running = False
        self.wait(1000)

    def seek_to(self, frame_idx: int):
        self.seek_requested_frame = frame_idx

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        return self.is_paused

    def set_speed(self, speed: float):
        self.speed = max(0.1, speed)


# ==============================================================================
# PHẦN 3: GIAO DIỆN NGƯỜI DÙNG PYQT5 (DESKTOP GUI)
# ==============================================================================

class QRAppWindow(QMainWindow):
    """Cửa sổ ứng dụng Desktop hiển thị trực quan toàn bộ quá trình phát hiện & giải mã."""

    def __init__(self, initial_video: str = ""):
        super().__init__()
        self.setWindowTitle("QR Code Dewarp & Decoder [Classical CV - OpenCV Pipeline]")
        self.resize(1360, 880)

        self.current_video_path = initial_video
        self.worker: Optional[VideoProcessingWorker] = None
        self.latest_annotated: Optional[np.ndarray] = None
        self.latest_warped: Optional[np.ndarray] = None

        self._init_ui()

        if self.current_video_path and os.path.isfile(self.current_video_path):
            self.load_video(self.current_video_path)

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        root_layout = QVBoxLayout(main_widget)
        root_layout.setContentsMargins(12, 10, 12, 10)
        root_layout.setSpacing(10)

        # 1. Thanh công cụ phía trên (Top Toolbar)
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)

        self.btn_open_video = QPushButton("📂 Mở Video (Import Video)")
        self.btn_open_video.setStyleSheet("background-color: #0d6efd; color: white; font-weight: bold; padding: 7px 14px; border-radius: 4px;")
        self.btn_open_video.clicked.connect(self.on_open_video_clicked)
        top_bar.addWidget(self.btn_open_video)

        self.btn_open_cam = QPushButton("📷 Mở Camera Trực Tiếp")
        self.btn_open_cam.setStyleSheet("background-color: #198754; color: white; font-weight: bold; padding: 7px 12px; border-radius: 4px;")
        self.btn_open_cam.clicked.connect(self.on_open_cam_clicked)
        top_bar.addWidget(self.btn_open_cam)

        self.btn_play_pause = QPushButton("⏸ Tạm dừng")
        self.btn_play_pause.setStyleSheet("padding: 7px 12px; font-weight: bold;")
        self.btn_play_pause.clicked.connect(self.on_play_pause_clicked)
        top_bar.addWidget(self.btn_play_pause)

        self.btn_step_prev = QPushButton("⏮ -1F")
        self.btn_step_prev.clicked.connect(lambda: self.step_frame(-1))
        top_bar.addWidget(self.btn_step_prev)

        self.btn_step_next = QPushButton("⏭ +1F")
        self.btn_step_next.clicked.connect(lambda: self.step_frame(1))
        top_bar.addWidget(self.btn_step_next)

        top_bar.addWidget(QLabel("Tốc độ:"))
        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"])
        self.combo_speed.setCurrentText("1.0x")
        self.combo_speed.currentIndexChanged.connect(self.on_speed_changed)
        top_bar.addWidget(self.combo_speed)

        self.btn_snapshot = QPushButton("💾 Xuất Ảnh Minh Chứng")
        self.btn_snapshot.setStyleSheet("background-color: #6c757d; color: white; padding: 7px 12px; border-radius: 4px;")
        self.btn_snapshot.clicked.connect(self.on_snapshot_clicked)
        top_bar.addWidget(self.btn_snapshot)

        top_bar.addStretch()

        self.lbl_fps_telemetry = QLabel("FPS: -- | Trễ: -- ms")
        self.lbl_fps_telemetry.setStyleSheet("font-weight: bold; color: #0dcaf0; font-size: 13px;")
        top_bar.addWidget(self.lbl_fps_telemetry)

        root_layout.addLayout(top_bar)

        # 2. Thanh tua Timeline (Seek Slider)
        timeline_bar = QHBoxLayout()
        timeline_bar.setSpacing(8)
        self.lbl_time_cur = QLabel("0 / 0")
        self.lbl_time_cur.setMinimumWidth(80)
        timeline_bar.addWidget(self.lbl_time_cur)

        self.slider_timeline = QSlider(Qt.Horizontal)
        self.slider_timeline.setRange(0, 100)
        self.slider_timeline.sliderMoved.connect(self.on_slider_moved)
        timeline_bar.addWidget(self.slider_timeline)
        root_layout.addLayout(timeline_bar)

        # 3. Khu vực hiển thị trung tâm (Main Views Splitter)
        splitter = QSplitter(Qt.Horizontal)

        # View 1: Video gốc kèm đường bao định vị (Main Feed)
        grp_main = QGroupBox("1. Video Gốc - Định vị Finder Patterns (TL: Đỏ, TR: Xanh, BL: Vàng)")
        vbox_main = QVBoxLayout(grp_main)
        self.view_main = QLabel("Chưa nạp video. Hãy bấm '📂 Mở Video' hoặc chạy start_qr.sh.")
        self.view_main.setAlignment(Qt.AlignCenter)
        self.view_main.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333; border-radius: 4px; color: #888;")
        self.view_main.setMinimumSize(640, 480)
        vbox_main.addWidget(self.view_main)
        splitter.addWidget(grp_main)

        # Cột bên phải: 2 View phụ
        side_panel = QWidget()
        vbox_side = QVBoxLayout(side_panel)
        vbox_side.setContentsMargins(0, 0, 0, 0)
        vbox_side.setSpacing(8)

        # View 2: QR đã nắn thẳng trực diện (Dewarped QR)
        grp_warped = QGroupBox("2. Mã QR Đã Nắn Phẳng Phối Cảnh (Perspective Dewarp)")
        vbox_warped = QVBoxLayout(grp_warped)
        self.view_warped = QLabel("Chờ phát hiện...")
        self.view_warped.setAlignment(Qt.AlignCenter)
        self.view_warped.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333; border-radius: 4px; color: #888;")
        self.view_warped.setMinimumSize(280, 280)
        vbox_warped.addWidget(self.view_warped)
        vbox_side.addWidget(grp_warped, stretch=1)

        # View 3: Ảnh phân ngưỡng thích nghi (Adaptive Threshold)
        grp_thresh = QGroupBox("3. Ảnh Phân Ngưỡng Thích Nghi (Adaptive Threshold)")
        vbox_thresh = QVBoxLayout(grp_thresh)
        self.view_thresh = QLabel("Chờ phát hiện...")
        self.view_thresh.setAlignment(Qt.AlignCenter)
        self.view_thresh.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333; border-radius: 4px; color: #888;")
        self.view_thresh.setMinimumSize(280, 180)
        vbox_thresh.addWidget(self.view_thresh)
        vbox_side.addWidget(grp_thresh, stretch=1)

        splitter.addWidget(side_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root_layout.addWidget(splitter, stretch=1)

        # 4. Thanh trạng thái & Kết quả Giải mã (Bottom Banner)
        bot_bar = QHBoxLayout()
        bot_bar.setSpacing(10)

        self.lbl_status_badge = QLabel("TRẠNG THÁI: CHỜ")
        self.lbl_status_badge.setStyleSheet("background-color: #6c757d; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        bot_bar.addWidget(self.lbl_status_badge)

        bot_bar.addWidget(QLabel("Nội dung giải mã (Text):"))
        self.txt_decoded = QLineEdit("Chưa giải mã được nội dung.")
        self.txt_decoded.setReadOnly(True)
        self.txt_decoded.setStyleSheet("font-size: 14px; font-weight: bold; color: #20c997; background-color: #212529; border: 1px solid #444; padding: 6px;")
        bot_bar.addWidget(self.txt_decoded, stretch=1)

        self.btn_copy = QPushButton("📋 Copy Text")
        self.btn_copy.setStyleSheet("padding: 6px 14px; font-weight: bold;")
        self.btn_copy.clicked.connect(self.on_copy_clicked)
        bot_bar.addWidget(self.btn_copy)

        root_layout.addLayout(bot_bar)

    def load_video(self, source):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

        self.current_video_path = str(source)
        self.worker = VideoProcessingWorker(source)
        self.worker.frame_ready.connect(self.on_frame_processed)
        self.worker.start()
        self.btn_play_pause.setText("⏸ Tạm dừng")
        print(f"[QRApp] Đã nạp nguồn video: {source}")

    def on_open_video_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn File Video chứa mã QR", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)"
        )
        if file_path:
            self.load_video(file_path)

    def on_open_cam_clicked(self):
        # Thử mở camera 0
        self.load_video("0")

    def on_play_pause_clicked(self):
        if self.worker:
            paused = self.worker.toggle_pause()
            self.btn_play_pause.setText("▶ Tiếp tục" if paused else "⏸ Tạm dừng")

    def step_frame(self, delta):
        if self.worker and self.worker.is_paused:
            cur = self.slider_timeline.value()
            self.worker.seek_to(max(0, cur + delta))

    def on_speed_changed(self):
        txt = self.combo_speed.currentText().replace("x", "")
        try:
            val = float(txt)
            if self.worker:
                self.worker.set_speed(val)
        except ValueError:
            pass

    def on_slider_moved(self, value):
        if self.worker:
            self.worker.seek_to(value)

    def on_copy_clicked(self):
        text = self.txt_decoded.text()
        if text and text != "Chưa giải mã được nội dung.":
            QApplication.clipboard().setText(text)
            QMessageBox.information(self, "Đã Copy", f"Đã sao chép nội dung vào Clipboard:\n\n{text}")

    def on_snapshot_clicked(self):
        if self.latest_annotated is None:
            QMessageBox.warning(self, "Lỗi", "Chưa có khung hình để lưu snapshot!")
            return
        out_dir = os.path.join(os.getcwd(), "output_qr")
        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time())
        p1 = os.path.join(out_dir, f"snap_{ts}_detected.jpg")
        p2 = os.path.join(out_dir, f"snap_{ts}_warped.jpg")
        cv2.imwrite(p1, self.latest_annotated)
        if self.latest_warped is not None:
            cv2.imwrite(p2, self.latest_warped)
        QMessageBox.information(self, "Thành công", f"Đã lưu ảnh snapshot vào:\n{p1}\n{p2}")

    def on_frame_processed(self, annotated, warped, thresh_inv, detected, text, elapsed_ms, cur_frame, total_frames):
        self.latest_annotated = annotated
        self.latest_warped = warped

        # 1. Update View 1 (Main Annotated)
        h, w, ch = annotated.shape
        bytes_per_line = ch * w
        q_img_main = QImage(annotated.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pix_main = QPixmap.fromImage(q_img_main).scaled(
            self.view_main.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.view_main.setPixmap(pix_main)

        # 2. Update View 2 (Warped 2D QR)
        if detected and warped is not None:
            wh, ww = warped.shape[:2]
            q_img_warp = QImage(warped.data, ww, wh, ww, QImage.Format_Grayscale8)
            pix_warp = QPixmap.fromImage(q_img_warp).scaled(
                self.view_warped.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.view_warped.setPixmap(pix_warp)
        else:
            self.view_warped.setText("Đang tìm kiếm...")

        # 3. Update View 3 (Thresholded)
        th, tw = thresh_inv.shape[:2]
        q_img_th = QImage(thresh_inv.data, tw, th, tw, QImage.Format_Grayscale8)
        pix_th = QPixmap.fromImage(q_img_th).scaled(
            self.view_thresh.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.view_thresh.setPixmap(pix_th)

        # 4. Status badge & text
        if detected:
            if text:
                self.lbl_status_badge.setText("✔ ĐÃ GIẢI MÃ")
                self.lbl_status_badge.setStyleSheet("background-color: #198754; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
                self.txt_decoded.setText(text)
            else:
                self.lbl_status_badge.setText("👁 ĐÃ TÌM THẤY QR")
                self.lbl_status_badge.setStyleSheet("background-color: #fd7e14; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")
        else:
            self.lbl_status_badge.setText("🔍 ĐANG TÌM KIẾM")
            self.lbl_status_badge.setStyleSheet("background-color: #6c757d; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px;")

        # 5. Telemetry
        fps = 1000.0 / max(1.0, elapsed_ms)
        self.lbl_fps_telemetry.setText(f"FPS: {fps:.1f} | Trễ: {elapsed_ms:.1f} ms")

        # 6. Timeline slider update
        self.slider_timeline.blockSignals(True)
        self.slider_timeline.setMaximum(max(1, total_frames))
        self.slider_timeline.setValue(cur_frame)
        self.slider_timeline.blockSignals(False)
        self.lbl_time_cur.setText(f"{cur_frame} / {total_frames}")

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        event.accept()


# ==============================================================================
# PHẦN 4: ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="QR Code Video Dewarp & Decoder Application")
    parser.add_argument("--video", "-v", type=str, default="", help="Đường dẫn đến file video hoặc index camera (ví dụ 0)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark Theme Palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(33, 37, 41))
    dark_palette.setColor(QPalette.WindowText, QColor(248, 249, 250))
    dark_palette.setColor(QPalette.Base, QColor(25, 28, 31))
    dark_palette.setColor(QPalette.AlternateBase, QColor(33, 37, 41))
    dark_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.Text, QColor(248, 249, 250))
    dark_palette.setColor(QPalette.Button, QColor(43, 48, 53))
    dark_palette.setColor(QPalette.ButtonText, QColor(248, 249, 250))
    dark_palette.setColor(QPalette.Highlight, QColor(13, 110, 253))
    dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(dark_palette)

    video_source = args.video
    if not video_source:
        default_sample = "/home/warlord/RobotNongNghiep/sample_qr_office.mp4"
        if os.path.isfile(default_sample):
            video_source = default_sample

    window = QRAppWindow(initial_video=video_source)
    window.show()

    exec_func = getattr(app, "exec", getattr(app, "exec_", None))
    sys.exit(exec_func())


if __name__ == "__main__":
    main()

