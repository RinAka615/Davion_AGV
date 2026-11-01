#!/usr/bin/env bash
# ==============================================================================
# Giao diện Đơn giản: Phát hiện & Đọc mã QR với Bộ chỉnh Độ tin cậy (Confidence)
# ==============================================================================
""":"
# Bash wrapper khởi động bằng python thích hợp
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_EXE="$SCRIPT_DIR/.venv/bin/python"
elif [ -f "$SCRIPT_DIR/.venv-yolo/bin/python" ]; then
    PYTHON_EXE="$SCRIPT_DIR/.venv-yolo/bin/python"
else
    PYTHON_EXE="python3"
fi
exec "$PYTHON_EXE" "$0" "$@"
"""

import os
import sys
import time
from typing import Optional, List, Dict, Any

import cv2
import numpy as np

# KHẮC PHỤC TRIỆT ĐỂ: cv2/config-3.py tự động ghi đè QT_QPA_PLATFORM_PLUGIN_PATH khi import cv2.
# Do đó ta cần khôi phục lại đường dẫn plugin chính xác của PyQt5 sau khi import cv2.
import PyQt5
_pyqt5_plugins = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
if os.path.exists(_pyqt5_plugins):
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _pyqt5_plugins
else:
    for _p in [
        "/usr/lib/aarch64-linux-gnu/qt5/plugins",
        "/usr/lib/x86_64-linux-gnu/qt5/plugins",
        "/usr/lib/qt5/plugins"
    ]:
        if os.path.exists(_p):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _p
            break
    else:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

if (os.environ.get("XDG_SESSION_TYPE") == "wayland" or bool(os.environ.get("WAYLAND_DISPLAY"))) and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "wayland;xcb"

os.environ["QT_LOGGING_RULES"] = "*.warning=false"

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QClipboard
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QCheckBox, QFileDialog, QMessageBox,
    QGroupBox, QFrame, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QTextEdit, QStatusBar, QComboBox, QDoubleSpinBox
)

# Thêm thư mục hiện tại vào sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from qr_scanner import QRDetector, QRJsonLogger, QRVisualizer, QRDetectionResult


DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1a1b26;
    color: #c0caf5;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #414868;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 14px;
    font-weight: bold;
    color: #7aa2f7;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    background-color: #1a1b26;
}
QPushButton {
    background-color: #24283b;
    border: 1px solid #414868;
    border-radius: 6px;
    padding: 7px 14px;
    color: #c0caf5;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #3b4261;
    border-color: #7aa2f7;
}
QPushButton:pressed {
    background-color: #7aa2f7;
    color: #15161e;
}
QPushButton#btnPlay {
    background-color: #2ac3de;
    color: #15161e;
    font-weight: bold;
}
QPushButton#btnPlay:hover {
    background-color: #7dcfff;
}
QPushButton#btnCopy {
    background-color: #9ece6a;
    color: #15161e;
    font-weight: bold;
}
QPushButton#btnCopy:hover {
    background-color: #b9f27c;
}
QSlider::groove:horizontal {
    height: 8px;
    background: #24283b;
    border-radius: 4px;
}
QSlider::sub-page:horizontal {
    background: #7aa2f7;
    border-radius: 4px;
}
QSlider::handle:horizontal {
    background: #c0caf5;
    border: 2px solid #7aa2f7;
    width: 18px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 9px;
}
QSlider::handle:horizontal:hover {
    background: #ffffff;
    border-color: #bb9af7;
}
QTextEdit {
    background-color: #15161e;
    border: 1px solid #414868;
    border-radius: 6px;
    color: #73daca;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 14px;
    padding: 8px;
}
QTableWidget {
    background-color: #15161e;
    border: 1px solid #414868;
    border-radius: 6px;
    gridline-color: #24283b;
    color: #c0caf5;
}
QHeaderView::section {
    background-color: #24283b;
    border: none;
    border-bottom: 1px solid #414868;
    padding: 5px;
    color: #7aa2f7;
    font-weight: bold;
}
QTableWidget::item:selected {
    background-color: #3b4261;
    color: #7dcfff;
}
QCheckBox {
    color: #c0caf5;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #414868;
    border-radius: 3px;
    background: #24283b;
}
QCheckBox::indicator:checked {
    background: #7aa2f7;
}
"""


class QRScannerGUI(QMainWindow):
    def __init__(self, initial_video: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("Bộ Quét & Đọc Mã QR - Tùy chỉnh Độ tin cậy (Robot Nông Nghiệp)")
        self.resize(1240, 820)
        self.setStyleSheet(DARK_STYLESHEET)

        # Trạng thái capture & video
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_camera = False
        self.total_frames = 0
        self.current_frame_idx = 0
        self.fps = 30.0
        self.is_playing = False
        self.last_frame_bgr: Optional[np.ndarray] = None

        # QR Pipeline Engine
        self.detector = QRDetector(backend="auto", enable_dewarp=True)
        self.visualizer = QRVisualizer()
        self.logger = QRJsonLogger(log_path="output_qr/gui_qr_events.jsonl", dedup_cooldown_sec=2.0)

        # Lịch sử quét mã trong session
        self.scan_history: List[Dict[str, Any]] = []

        # Timer vòng lặp phát video
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_tick)

        self._init_ui()

        # Nạp video mặc định nếu có
        if initial_video and os.path.exists(initial_video):
            self.load_source(initial_video)
        else:
            # Tự động nạp video kiểm thử nếu có
            test_vid = os.path.join(CURRENT_DIR, "test_qr_dataset", "test_qr_multiscene.mp4")
            greenhouse_vid = os.path.join(CURRENT_DIR, "Quỳnh Robotics [13-10-2026 20_33]", "2aobor4lgjkrognqzs1jz63kldqrlxveqvfef1rw4.mp4")
            if os.path.exists(test_vid):
                self.load_source(test_vid)
            elif os.path.exists(greenhouse_vid):
                self.load_source(greenhouse_vid)

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ==========================================
        # PANEL TRÁI: VIDEO & ĐIỀU KHIỂN PHÁT
        # ==========================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # Thanh toolbar chọn nguồn
        src_bar = QHBoxLayout()
        btn_open_file = QPushButton("📂 Chọn File Video...")
        btn_open_file.clicked.connect(self._choose_video_file)
        self.btn_open_cam = QPushButton("📷 Mở Webcam (Live Stream)")
        self.btn_open_cam.clicked.connect(self._open_webcam)

        self.btn_snapshot = QPushButton("📸 Chụp Ảnh Frame")
        self.btn_snapshot.clicked.connect(self._take_snapshot)

        self.lbl_source_info = QLabel("Chưa mở nguồn nào")
        self.lbl_source_info.setStyleSheet("color: #bb9af7; font-weight: bold;")

        src_bar.addWidget(btn_open_file)
        src_bar.addWidget(self.btn_open_cam)
        src_bar.addWidget(self.btn_snapshot)
        src_bar.addWidget(self.lbl_source_info)
        src_bar.addStretch()
        left_layout.addLayout(src_bar)

        # Màn hình hiển thị Video
        self.lbl_video = QLabel("Vui lòng mở video hoặc webcam để quét mã QR")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setMinimumSize(640, 480)
        self.lbl_video.setStyleSheet("background-color: #15161e; border: 2px solid #24283b; border-radius: 8px;")
        left_layout.addWidget(self.lbl_video, stretch=1)

        # Thanh trượt tua video (Timeline Slider)
        timeline_bar = QHBoxLayout()
        self.slider_timeline = QSlider(Qt.Horizontal)
        self.slider_timeline.setRange(0, 0)
        self.slider_timeline.sliderMoved.connect(self._on_timeline_slider_moved)
        self.slider_timeline.sliderPressed.connect(self._on_timeline_pressed)

        self.lbl_time_info = QLabel("00:00 / 00:00 (Frame 0/0)")
        self.lbl_time_info.setStyleSheet("color: #a9b1d6; font-family: monospace;")

        timeline_bar.addWidget(self.slider_timeline)
        timeline_bar.addWidget(self.lbl_time_info)
        left_layout.addLayout(timeline_bar)

        # Nút điều khiển phát
        ctrl_bar = QHBoxLayout()
        self.btn_step_back = QPushButton("◀ 1 Frame")
        self.btn_step_back.clicked.connect(self._step_backward)

        self.btn_play_pause = QPushButton("▶ Phát Video")
        self.btn_play_pause.setObjectName("btnPlay")
        self.btn_play_pause.clicked.connect(self._toggle_play_pause)

        self.btn_step_fwd = QPushButton("1 Frame ▶")
        self.btn_step_fwd.clicked.connect(self._step_forward)

        self.lbl_fps_badge = QLabel("FPS: --")
        self.lbl_fps_badge.setStyleSheet("color: #e0af68; font-weight: bold; padding: 0 10px;")

        ctrl_bar.addWidget(self.btn_step_back)
        ctrl_bar.addWidget(self.btn_play_pause)
        ctrl_bar.addWidget(self.btn_step_fwd)
        ctrl_bar.addStretch()
        ctrl_bar.addWidget(self.lbl_fps_badge)
        left_layout.addLayout(ctrl_bar)

        splitter.addWidget(left_widget)

        # ==========================================
        # PANEL PHẢI: ĐỘ TIN CẬY & NỘI DUNG MÃ QR
        # ==========================================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # GROUP 1: BỘ ĐIỀU CHỈNH ĐỘ TIN CẬY (CONFIDENCE THRESHOLD)
        grp_conf = QGroupBox("🎛️ Bộ Điều Chỉnh Độ Tin Cậy & Độ Nhạy")
        grp_conf_layout = QVBoxLayout(grp_conf)

        conf_header = QHBoxLayout()
        self.lbl_conf_val = QLabel("Độ tin cậy: 40% (Độ nhạy trung bình)")
        self.lbl_conf_val.setStyleSheet("color: #2ac3de; font-weight: bold; font-size: 14px;")
        conf_header.addWidget(self.lbl_conf_val)
        grp_conf_layout.addLayout(conf_header)

        # Thanh trượt độ tin cậy từ 5% đến 95%
        self.slider_confidence = QSlider(Qt.Horizontal)
        self.slider_confidence.setRange(5, 95)
        self.slider_confidence.setValue(40)
        self.slider_confidence.valueChanged.connect(self._on_confidence_changed)
        grp_conf_layout.addWidget(self.slider_confidence)

        self.lbl_conf_desc = QLabel("💡 Kéo sang TRÁI để giảm độ tin cậy: Tăng cường quét đa tầng (CLAHE, phân ngưỡng, phóng to) giúp đọc mã mờ, nghiêng, nhỏ hoặc thiếu sáng.")
        self.lbl_conf_desc.setWordWrap(True)
        self.lbl_conf_desc.setStyleSheet("color: #565f89; font-size: 11px; margin-bottom: 6px;")
        grp_conf_layout.addWidget(self.lbl_conf_desc)

        # Các tùy chọn bổ sung
        self.chk_dewarp = QCheckBox("Tự động nắn phối cảnh (Perspective Dewarp)")
        self.chk_dewarp.setChecked(True)
        self.chk_dewarp.toggled.connect(self._reprocess_current_frame)
        grp_conf_layout.addWidget(self.chk_dewarp)

        self.chk_show_candidates = QCheckBox("Hiển thị vùng nghi vấn chưa giải mã (Hộp viền Cam)")
        self.chk_show_candidates.setChecked(True)
        self.chk_show_candidates.toggled.connect(self._reprocess_current_frame)
        grp_conf_layout.addWidget(self.chk_show_candidates)

        self.chk_datamatrix = QCheckBox("Quét thêm định dạng Data Matrix / 2D Barcode")
        self.chk_datamatrix.setChecked(False)
        self.chk_datamatrix.toggled.connect(self._reprocess_current_frame)
        grp_conf_layout.addWidget(self.chk_datamatrix)

        right_layout.addWidget(grp_conf)

        # GROUP 2: NỘI DUNG MÃ QR ĐANG ĐỌC ĐƯỢC
        grp_content = QGroupBox("🎯 Nội Dung Mã QR Đang Phát Hiện")
        grp_content_layout = QVBoxLayout(grp_content)

        # Text edit hiển thị nội dung
        self.txt_qr_content = QTextEdit()
        self.txt_qr_content.setReadOnly(True)
        self.txt_qr_content.setPlaceholderText("Chưa có mã QR nào được giải mã trong khung hình hiện tại...")
        self.txt_qr_content.setMaximumHeight(110)
        grp_content_layout.addWidget(self.txt_qr_content)

        # Nút Copy và xem ảnh patch nắn thẳng
        content_btn_bar = QHBoxLayout()
        self.btn_copy_content = QPushButton("📋 Sao Chép Nội Dung")
        self.btn_copy_content.setObjectName("btnCopy")
        self.btn_copy_content.clicked.connect(self._copy_to_clipboard)

        self.lbl_copy_toast = QLabel("")
        self.lbl_copy_toast.setStyleSheet("color: #9ece6a; font-weight: bold;")

        content_btn_bar.addWidget(self.btn_copy_content)
        content_btn_bar.addWidget(self.lbl_copy_toast)
        content_btn_bar.addStretch()
        grp_content_layout.addLayout(content_btn_bar)

        # Ảnh thu nhỏ của mã QR đã nắn thẳng
        patch_box = QHBoxLayout()
        patch_box.addWidget(QLabel("Ảnh nắn thẳng (Warped):"))
        self.lbl_dewarp_patch = QLabel("Chưa có ảnh")
        self.lbl_dewarp_patch.setFixedSize(90, 90)
        self.lbl_dewarp_patch.setAlignment(Qt.AlignCenter)
        self.lbl_dewarp_patch.setStyleSheet("background-color: #15161e; border: 1px dashed #414868; border-radius: 4px;")
        patch_box.addWidget(self.lbl_dewarp_patch)
        patch_box.addStretch()
        grp_content_layout.addLayout(patch_box)

        right_layout.addWidget(grp_content)

        # GROUP 3: ĐO LƯỜNG KÍCH THƯỚC MÃ QR
        grp_measure = QGroupBox("📐 Kích Thước & Đo Lường Mã QR")
        grp_measure_layout = QVBoxLayout(grp_measure)

        self.lbl_measure_px = QLabel("Kích thước Pixel: -- × -- px")
        self.lbl_measure_px.setStyleSheet("color: #7dcfff; font-weight: bold; font-size: 14px;")
        grp_measure_layout.addWidget(self.lbl_measure_px)

        self.lbl_measure_center = QLabel("🎯 Tâm mã QR: (X: --, Y: --) | Lệch Camera: --")
        self.lbl_measure_center.setStyleSheet("color: #00E676; font-weight: bold; font-size: 13px;")
        grp_measure_layout.addWidget(self.lbl_measure_center)

        self.lbl_measure_details = QLabel(
            "4 Cạnh: Trên --px | Dưới --px | Trái --px | Phải --px\n"
            "Diện tích: -- px² | Tỷ lệ W/H: -- | Góc xoay: --°"
        )
        self.lbl_measure_details.setStyleSheet("color: #a9b1d6; font-size: 11px; font-family: monospace;")
        grp_measure_layout.addWidget(self.lbl_measure_details)

        # Thanh cài đặt kích thước thực tế (mm) để ước tính khoảng cách và mm/px
        calib_layout = QHBoxLayout()
        calib_layout.addWidget(QLabel("Tem QR in thực tế:"))

        self.combo_real_size = QComboBox()
        self.combo_real_size.addItems([
            "50 mm (Tem 5×5 cm)",
            "60 mm (Tem 6×6 cm)",
            "80 mm (Tem 8×8 cm)",
            "100 mm (Tem 10×10 cm)",
            "150 mm (Tem 15×15 cm)",
            "200 mm (Tem 20×20 cm)",
            "Tùy chỉnh (mm)"
        ])
        self.combo_real_size.currentIndexChanged.connect(self._on_real_size_changed)
        calib_layout.addWidget(self.combo_real_size)

        self.spin_real_size = QDoubleSpinBox()
        self.spin_real_size.setRange(5.0, 1000.0)
        self.spin_real_size.setValue(50.0)
        self.spin_real_size.setSuffix(" mm")
        self.spin_real_size.setSingleStep(5.0)
        self.spin_real_size.setEnabled(False)
        self.spin_real_size.valueChanged.connect(self._reprocess_current_frame)
        calib_layout.addWidget(self.spin_real_size)
        grp_measure_layout.addLayout(calib_layout)

        # Nhãn kết quả quy đổi thực tế & khoảng cách
        self.lbl_real_estimation = QLabel("Độ phân giải: -- px/mm | Khoảng cách ước tính (Camera -> QR): --")
        self.lbl_real_estimation.setStyleSheet("color: #bb9af7; font-weight: bold;")
        grp_measure_layout.addWidget(self.lbl_real_estimation)

        right_layout.addWidget(grp_measure)

        # GROUP 4: LỊCH SỬ CÁC MÃ ĐÃ ĐỌC (SCAN HISTORY)
        grp_history = QGroupBox("📜 Lịch Sử Quét Mã (Nhấn đúp để nhảy tới Frame)")
        grp_history_layout = QVBoxLayout(grp_history)

        self.table_history = QTableWidget(0, 5)
        self.table_history.setHorizontalHeaderLabels(["Thời Gian", "Nội Dung Mã QR", "Tâm (X, Y)", "Kích Thước", "Backend"])
        self.table_history.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_history.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_history.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_history.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_history.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table_history.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_history.cellDoubleClicked.connect(self._on_history_row_double_clicked)
        grp_history_layout.addWidget(self.table_history)

        hist_btn_bar = QHBoxLayout()
        btn_clear_hist = QPushButton("🗑️ Xóa Lịch Sử")
        btn_clear_hist.clicked.connect(self._clear_history)
        btn_export_jsonl = QPushButton("💾 Mở File Log JSONL")
        btn_export_jsonl.clicked.connect(self._open_jsonl_log)
        hist_btn_bar.addWidget(btn_clear_hist)
        hist_btn_bar.addWidget(btn_export_jsonl)
        hist_btn_bar.addStretch()
        grp_history_layout.addLayout(hist_btn_bar)

        right_layout.addWidget(grp_history, stretch=1)

        splitter.addWidget(right_widget)
        splitter.setSizes([740, 500])

        # Thanh trạng thái dưới cùng
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Sẵn sàng. Hãy chọn video hoặc camera để bắt đầu.")

    def _choose_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn file video",
            CURRENT_DIR,
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)"
        )
        if file_path:
            self.load_source(file_path)

    def _open_webcam(self):
        """Mở hoặc tắt Webcam phát luồng liên tục thời gian thực (Live Stream)"""
        # Nếu đang mở camera thì bấm lại sẽ TẮT camera
        if self.is_camera and self.cap is not None:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.is_camera = False
            self.is_playing = False
            self.btn_open_cam.setText("📷 Mở Webcam (Live Stream)")
            self.btn_open_cam.setStyleSheet("")
            self.btn_play_pause.setText("▶ Phát Video")
            self.btn_play_pause.setStyleSheet("")
            self.lbl_source_info.setText("Đã tắt Webcam.")
            self.status_bar.showMessage("Đã tắt Webcam.")
            return

        # Dừng timer và giải phóng capture cũ nếu đang mở video
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.timer.stop()
        self.is_playing = False

        available_cams = []
        for idx in range(10):
            test_path = f"/dev/video{idx}"
            if os.path.exists(test_path):
                available_cams.append(idx)

        # Quét lần lượt các cổng camera khả dụng và kiểm tra đọc frame thực tế
        working_cap = None
        working_idx = None
        for idx in available_cams:
            for api in [cv2.CAP_V4L2, cv2.CAP_ANY]:
                try:
                    test_cap = cv2.VideoCapture(idx, api)
                    if test_cap.isOpened():
                        ret, test_frame = test_cap.read()
                        if ret and test_frame is not None:
                            working_cap = test_cap
                            working_idx = idx
                            break
                        test_cap.release()
                    else:
                        test_cap.release()
                except Exception:
                    pass
            if working_cap is not None:
                break

        if working_cap is not None:
            self.cap = working_cap
            self.is_camera = True
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.total_frames = 0
            self.camera_start_time = time.time()
            self.slider_timeline.setEnabled(False)
            self.btn_step_back.setEnabled(False)
            self.btn_step_fwd.setEnabled(False)

            # Tự động bật Live Streaming liên tục
            self.is_playing = True
            self.timer.start(30)
            self.btn_open_cam.setText("⏹ Tắt Webcam")
            self.btn_open_cam.setStyleSheet("background-color: #f7768e; color: #15161e; font-weight: bold;")
            self.btn_play_pause.setText("⏸ Dừng Live Stream")
            self.btn_play_pause.setStyleSheet("background-color: #f7768e; color: #15161e; font-weight: bold;")

            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.lbl_source_info.setText(f"Nguồn: Webcam #{working_idx} ({w}x{h}) [LIVE STREAMING LIÊN TỤC]")
            self.status_bar.showMessage(f"Đã kết nối Webcam #{working_idx} ({w}x{h}) - Đang truyền trực tiếp...")
            return

        # Không tìm thấy camera nào hoạt động -> Giải phóng và hiển thị cảnh báo
        self.is_camera = False
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Không tìm thấy Webcam")
        msg.setText("<b>Không thể mở camera trên máy tính!</b>")
        
        info_text = (
            f"Đã kiểm tra các cổng camera: {available_cams if available_cams else 'Không có'}.<br><br>"
            "<b>Các bước kiểm tra:</b><br>"
            "1. <b>Laptop Lenovo Legion (máy hiện tại)</b>: Có một <u>công tắc gạt vật lý bảo mật camera (E-Shutter)</u> "
            "nằm ở <b>cạnh bên PHẢI</b> thân máy (gần cổng USB/jack 3.5mm). Hãy gạt công tắc này sang vị trí <b>MỞ</b>.<br><br>"
            "2. <b>Phím tắt trên bàn phím</b>: Thử bấm phím <b>F9</b> hoặc tổ hợp phím <b>Fn + F9</b> (biểu tượng camera gạch chéo) để mở khóa.<br><br>"
            "3. <b>Nếu dùng Camera USB / Orbbec</b>: Hãy kiểm tra cắm lại cáp USB.<br><br>"
            "<i>(Sau khi gạt công tắc, hãy bấm lại nút '📷 Mở Webcam').</i>"
        )
        msg.setInformativeText(info_text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        self.lbl_source_info.setText("Chưa kết nối Webcam. Hãy gạt công tắc camera ở cạnh phải máy.")

    def load_source(self, source_path: str):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.timer.stop()
        self.is_playing = False

        if source_path.isdigit():
            cam_idx = int(source_path)
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_ANY)
            self.is_camera = True
            source_title = f"Webcam #{cam_idx}"
            if not self.cap.isOpened():
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                self.is_camera = False
                QMessageBox.warning(
                    self,
                    "Không thể mở Webcam",
                    f"Không thể mở camera /dev/video{cam_idx}.\nVui lòng cắm webcam hoặc chọn file video."
                )
                self.lbl_source_info.setText("Lỗi: Không tìm thấy Webcam")
                return

            # Cài đặt buffer = 1 để hình ảnh luôn mới nhất thời gian thực, không bị giật lag
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.total_frames = 0
            self.camera_start_time = time.time()
            self.slider_timeline.setEnabled(False)
            self.btn_step_back.setEnabled(False)
            self.btn_step_fwd.setEnabled(False)

            # TỰ ĐỘNG BẬT LIVE STREAMING LIÊN TỤC CHO WEBCAM
            self.is_playing = True
            self.timer.start(30)  # Cập nhật liên tục ~33 FPS
            self.btn_open_cam.setText("⏹ Tắt Webcam")
            self.btn_open_cam.setStyleSheet("background-color: #f7768e; color: #15161e; font-weight: bold;")
            self.btn_play_pause.setText("⏸ Dừng Live Stream")
            self.btn_play_pause.setStyleSheet("background-color: #f7768e; color: #15161e; font-weight: bold;")

        else:
            if not os.path.exists(source_path):
                QMessageBox.warning(self, "Lỗi", f"File video không tồn tại:\n{source_path}")
                return
            self.cap = cv2.VideoCapture(source_path)
            self.is_camera = False
            source_title = os.path.basename(source_path)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Lỗi", f"OpenCV không thể giải mã file video:\n{source_path}")
                return

            self.btn_open_cam.setText("📷 Mở Webcam (Live Stream)")
            self.btn_open_cam.setStyleSheet("")
            self.btn_play_pause.setText("▶ Phát Video")
            self.btn_play_pause.setStyleSheet("")

            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.slider_timeline.setEnabled(True)
            self.slider_timeline.setRange(0, max(0, self.total_frames - 1))
            self.slider_timeline.setValue(0)
            self.btn_step_back.setEnabled(True)
            self.btn_step_fwd.setEnabled(True)

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mode_tag = "[LIVE STREAMING LIÊN TỤC]" if self.is_camera else ""
        self.lbl_source_info.setText(f"Nguồn: {source_title} ({w}x{h}) {mode_tag}")

        # Đọc và hiển thị frame đầu tiên
        self.current_frame_idx = 0
        self._read_and_process_frame()

    def _toggle_play_pause(self):
        if not self.cap or not self.cap.isOpened():
            return

        if self.is_playing:
            self.is_playing = False
            self.timer.stop()
            if self.is_camera:
                self.btn_play_pause.setText("▶ Tiếp Tục Live Stream")
                self.btn_play_pause.setStyleSheet("background-color: #2ac3de; color: #15161e; font-weight: bold;")
            else:
                self.btn_play_pause.setText("▶ Phát Video")
                self.btn_play_pause.setStyleSheet("")
        else:
            self.is_playing = True
            interval_ms = 30 if self.is_camera else max(10, int(1000 / self.fps))
            self.timer.start(interval_ms)
            self.btn_play_pause.setText("⏸ Dừng Live Stream" if self.is_camera else "⏸ Tạm Dừng")
            self.btn_play_pause.setStyleSheet("background-color: #f7768e; color: #15161e; font-weight: bold;")

    def _take_snapshot(self):
        """Chụp và lưu ảnh khung hình hiện tại vào thư mục output_qr/snapshots/"""
        if self.last_frame_bgr is None:
            QMessageBox.information(self, "Thông báo", "Chưa có khung hình nào để chụp!")
            return

        snap_dir = os.path.join(CURRENT_DIR, "output_qr", "snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        filename = f"manual_snap_{int(time.time()*1000)}.jpg"
        filepath = os.path.join(snap_dir, filename)
        cv2.imwrite(filepath, self.last_frame_bgr)

        self.status_bar.showMessage(f"✔ Đã chụp và lưu ảnh vào: {filepath}")
        self.lbl_copy_toast.setText("✔ Đã chụp ảnh!")
        QTimer.singleShot(2500, lambda: self.lbl_copy_toast.setText(""))

    def _on_timer_tick(self):
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            if not self.is_camera:
                # Hết video -> dừng lại ở frame cuối
                self._toggle_play_pause()
                self.status_bar.showMessage("Đã phát hết video.")
            return

        self.current_frame_idx += 1
        self._process_frame(frame)

    def _read_and_process_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if ret:
            self._process_frame(frame)

    def _process_frame(self, frame: np.ndarray):
        self.last_frame_bgr = frame.copy()
        t0 = time.time()

        # Lấy giá trị độ tin cậy từ slider (5% -> 95% tương ứng 0.05 -> 0.95)
        conf_val = self.slider_confidence.value() / 100.0
        enable_dewarp = self.chk_dewarp.isChecked()
        show_candidates = self.chk_show_candidates.isChecked()
        include_datamatrix = self.chk_datamatrix.isChecked()

        self.detector.enable_dewarp = enable_dewarp

        # PHÁT HIỆN & GIẢI MÃ VỚI NGƯỠNG ĐỘ TIN CẬY
        results = self.detector.detect_and_decode(
            frame,
            confidence=conf_val,
            include_other_formats=include_datamatrix
        )

        # Lọc hiển thị candidates nếu người dùng bỏ chọn
        if not show_candidates:
            results = [r for r in results if r.is_decoded]

        elapsed_ms = (time.time() - t0) * 1000.0
        fps_current = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
        self.lbl_fps_badge.setText(f"FPS: {fps_current:.1f} ({elapsed_ms:.1f}ms)")

        # Cập nhật thông tin nội dung QR hiển thị
        decoded_results = [r for r in results if r.is_decoded]
        candidates = [r for r in results if not r.is_decoded]

        if decoded_results:
            first = decoded_results[0]
            self.txt_qr_content.setText(first.text)
            self.txt_qr_content.setStyleSheet("background-color: #15161e; border: 2px solid #00E676; color: #00E676; font-weight: bold;")

            # Hiển thị ảnh patch nắn thẳng
            if first.dewarped_patch is not None:
                self._display_patch(first.dewarped_patch)

            # Ghi nhận vào lịch sử
            vid_time = (self.current_frame_idx / self.fps) if not self.is_camera else None
            for dec in decoded_results:
                size_str = f"{dec.width_px:.0f}×{dec.height_px:.0f} px"
                center_str = f"({dec.center[0]:.0f}, {dec.center[1]:.0f})"
                logged = self.logger.log(
                    text=dec.text,
                    corners=dec.points,
                    backend=dec.backend,
                    video_timestamp_sec=vid_time,
                    barcode_format=dec.format_name,
                    geometry=dec.geometry
                )
                if logged:
                    self._add_to_history(dec.text, vid_time, size_str, dec.backend, center_str)

        else:
            if candidates:
                cand = candidates[0]
                self.txt_qr_content.setText(f"[VÙNG NGHI VẤN CHƯA GIẢI MÃ]\nLý do: {cand.error}\n(Mẹo: Hãy thử chỉnh slider độ tin cậy hoặc nắn góc nhìn)")
                self.txt_qr_content.setStyleSheet("background-color: #15161e; border: 2px solid #FF9100; color: #FF9100;")
                if cand.dewarped_patch is not None:
                    self._display_patch(cand.dewarped_patch)
            else:
                self.txt_qr_content.setText("Không phát hiện mã QR trong khung hình này.")
                self.txt_qr_content.setStyleSheet("background-color: #15161e; border: 1px solid #414868; color: #565f89;")

        # --- CẬP NHẬT ĐO LƯỜNG KÍCH THƯỚC VÀ KHOẢNG CÁCH ---
        target_res = decoded_results[0] if decoded_results else (candidates[0] if candidates else None)
        if target_res is not None:
            w_px = target_res.width_px
            h_px = target_res.height_px
            geom = target_res.geometry or {}
            top_e = geom.get("top_edge_px", w_px)
            bot_e = geom.get("bottom_edge_px", w_px)
            left_e = geom.get("left_edge_px", h_px)
            right_e = geom.get("right_edge_px", h_px)
            area = geom.get("area_px", target_res.area_px)
            aspect = geom.get("aspect_ratio", target_res.aspect_ratio)
            orient = geom.get("orientation_deg", target_res.orientation_deg)

            fh, fw = frame.shape[:2]
            pct_frame = (area / max(1, fh * fw)) * 100.0

            cx, cy = target_res.center
            dx = cx - (fw / 2.0)
            dy = cy - (fh / 2.0)
            dir_str = "Lệch Phải" if dx > 5 else ("Lệch Trái" if dx < -5 else "Chuẩn Tâm")

            self.lbl_measure_px.setText(f"Kích thước Pixel: {w_px:.1f} × {h_px:.1f} px")
            self.lbl_measure_center.setText(f"🎯 Tâm QR: ({cx:.1f}, {cy:.1f}) | Lệch Camera: ΔX: {dx:+.1f}px ({dir_str}), ΔY: {dy:+.1f}px")
            self.lbl_measure_details.setText(
                f"4 Cạnh: Trên {top_e:.0f}px | Dưới {bot_e:.0f}px | Trái {left_e:.0f}px | Phải {right_e:.0f}px\n"
                f"Diện tích: {area:,.0f} px² ({pct_frame:.1f}% khung hình) | Tỷ lệ W/H: {aspect:.2f} | Góc xoay: {orient:+.1f}°"
            )

            real_size_mm = self._get_selected_real_size_mm()
            if w_px > 3:
                px_per_mm = w_px / real_size_mm
                mm_per_px = real_size_mm / w_px
                # Mô hình Pinhole ước tính khoảng cách: f ~ 0.95 * chiều rộng khung hình
                focal_px = fw * 0.95
                dist_cm = ((focal_px * real_size_mm) / w_px) / 10.0
                self.lbl_real_estimation.setText(
                    f"Độ phân giải: {px_per_mm:.2f} px/mm ({mm_per_px:.2f} mm/px) | Khoảng cách ước tính: ~ {dist_cm:.1f} cm ({dist_cm/100:.2f} m)"
                )
            else:
                self.lbl_real_estimation.setText("Độ phân giải: -- px/mm | Khoảng cách ước tính: --")
        else:
            self.lbl_measure_px.setText("Kích thước Pixel: -- × -- px")
            self.lbl_measure_center.setText("🎯 Tâm mã QR: (X: --, Y: --) | Lệch Camera: --")
            self.lbl_measure_details.setText(
                "4 Cạnh: Trên --px | Dưới --px | Trái --px | Phải --px\n"
                "Diện tích: -- px² | Tỷ lệ W/H: -- | Góc xoay: --°"
            )
            self.lbl_real_estimation.setText("Độ phân giải: -- px/mm | Khoảng cách ước tính: --")

        # Vẽ kết quả lên ảnh
        vis_frame = self.visualizer.draw_results(frame, results)

        # Hiển thị lên QLabel
        self._display_video_frame(vis_frame)

        # Cập nhật thanh trượt timeline & nhãn thời gian
        if not self.is_camera and self.total_frames > 0:
            self.slider_timeline.blockSignals(True)
            self.slider_timeline.setValue(self.current_frame_idx)
            self.slider_timeline.blockSignals(False)

            cur_sec = self.current_frame_idx / self.fps
            tot_sec = self.total_frames / self.fps
            self.lbl_time_info.setText(
                f"{int(cur_sec//60):02d}:{cur_sec%60:04.1f} / {int(tot_sec//60):02d}:{tot_sec%60:04.1f} "
                f"(Frame {self.current_frame_idx}/{self.total_frames})"
            )
        elif self.is_camera:
            live_sec = (time.time() - self.camera_start_time) if hasattr(self, 'camera_start_time') else 0.0
            self.lbl_time_info.setText(
                f"🔴 LIVE: {int(live_sec//60):02d}:{int(live_sec%60):02d} (Frame #{self.current_frame_idx})"
            )

    def _display_video_frame(self, frame_bgr: np.ndarray):
        rgb_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Scale giữ đúng tỉ lệ khung hình (Aspect Ratio) vừa vặn kích thước hiển thị của QLabel
        lbl_size = self.lbl_video.size()
        pixmap = QPixmap.fromImage(q_img).scaled(
            lbl_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.lbl_video.setPixmap(pixmap)

    def _display_patch(self, patch_bgr: np.ndarray):
        rgb_patch = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        ph, pw, pch = rgb_patch.shape
        q_img = QImage(rgb_patch.data, pw, ph, pch * pw, QImage.Format_RGB888)
        pix = QPixmap.fromImage(q_img).scaled(
            self.lbl_dewarp_patch.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.lbl_dewarp_patch.setPixmap(pix)

    def _on_confidence_changed(self, val: int):
        if val < 25:
            desc = "RẤT THẤP (Quét sâu đa tầng, tăng cường tương phản tối đa)"
            color = "#f7768e"
        elif val < 45:
            desc = "THẤP (Cân bằng tốt giữa bắt mã mờ và tránh báo sai)"
            color = "#e0af68"
        elif val < 70:
            desc = "TRUNG BÌNH (Tiêu chuẩn)"
            color = "#2ac3de"
        else:
            desc = "CAO (Nghiêm ngặt, chỉ nhận mã cực rõ)"
            color = "#9ece6a"

        self.lbl_conf_val.setText(f"Độ tin cậy: {val}% ({desc})")
        self.lbl_conf_val.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 14px;")

        # Nếu video đang tạm dừng, chạy quét lại ngay lập tức frame hiện tại để xem kết quả
        self._reprocess_current_frame()

    def _reprocess_current_frame(self):
        if not self.is_playing and self.last_frame_bgr is not None:
            self._process_frame(self.last_frame_bgr)

    def _step_forward(self):
        if not self.cap or self.is_camera:
            return
        if self.is_playing:
            self._toggle_play_pause()
        self.current_frame_idx = min(self.total_frames - 1, self.current_frame_idx + 1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        self._read_and_process_frame()

    def _step_backward(self):
        if not self.cap or self.is_camera:
            return
        if self.is_playing:
            self._toggle_play_pause()
        self.current_frame_idx = max(0, self.current_frame_idx - 1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        self._read_and_process_frame()

    def _on_timeline_pressed(self):
        if self.is_playing:
            self._toggle_play_pause()

    def _on_timeline_slider_moved(self, pos: int):
        if not self.cap or self.is_camera:
            return
        self.current_frame_idx = pos
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        self._read_and_process_frame()

    def _copy_to_clipboard(self):
        text = self.txt_qr_content.toPlainText().strip()
        if text and not text.startswith("[VÙNG NGHI VẤN") and not text.startswith("Không phát hiện"):
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            self.lbl_copy_toast.setText("✔ Đã sao chép!")
            QTimer.singleShot(2000, lambda: self.lbl_copy_toast.setText(""))
        else:
            self.lbl_copy_toast.setText("⚠️ Chưa có nội dung mã!")
            QTimer.singleShot(2000, lambda: self.lbl_copy_toast.setText(""))

    def _get_selected_real_size_mm(self) -> float:
        """Lấy kích thước tem QR thực tế (mm) để tính toán độ phân giải và khoảng cách"""
        idx = self.combo_real_size.currentIndex()
        presets = [50.0, 60.0, 80.0, 100.0, 150.0, 200.0]
        if idx < len(presets):
            return presets[idx]
        return float(self.spin_real_size.value())

    def _on_real_size_changed(self, idx: int):
        # Nếu chọn 'Tùy chỉnh (mm)' (index 6) thì mở khóa ô nhập số spin_real_size
        is_custom = (idx == 6)
        self.spin_real_size.setEnabled(is_custom)
        self._reprocess_current_frame()

    def _add_to_history(self, text: str, vid_time_sec: Optional[float], size_str: str, backend: str, center_str: str = "--"):
        row_pos = self.table_history.rowCount()
        self.table_history.insertRow(row_pos)

        time_str = f"{vid_time_sec:.2f}s" if vid_time_sec is not None else time.strftime("%H:%M:%S")
        item_time = QTableWidgetItem(time_str)
        item_text = QTableWidgetItem(text)
        item_center = QTableWidgetItem(center_str)
        item_size = QTableWidgetItem(size_str)
        item_backend = QTableWidgetItem(backend)

        # Lưu frame index vào item để nhảy đến khi click
        frame_target = int(vid_time_sec * self.fps) if vid_time_sec is not None else 0
        item_time.setData(Qt.UserRole, frame_target)

        self.table_history.setItem(row_pos, 0, item_time)
        self.table_history.setItem(row_pos, 1, item_text)
        self.table_history.setItem(row_pos, 2, item_center)
        self.table_history.setItem(row_pos, 3, item_size)
        self.table_history.setItem(row_pos, 4, item_backend)
        self.table_history.scrollToBottom()

    def _on_history_row_double_clicked(self, row: int, col: int):
        item_time = self.table_history.item(row, 0)
        if item_time and not self.is_camera:
            target_frame = item_time.data(Qt.UserRole)
            if target_frame is not None:
                if self.is_playing:
                    self._toggle_play_pause()
                self.current_frame_idx = target_frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                self._read_and_process_frame()
                self.status_bar.showMessage(f"Đã chuyển video đến Frame {target_frame}")

    def _clear_history(self):
        self.table_history.setRowCount(0)
        self.status_bar.showMessage("Đã xóa lịch sử quét.")

    def _open_jsonl_log(self):
        log_path = os.path.abspath("output_qr/gui_qr_events.jsonl")
        if os.path.exists(log_path):
            QMessageBox.information(self, "File Nhật Ký JSONL", f"File log được lưu tại:\n{log_path}")
        else:
            QMessageBox.information(self, "Thông báo", "Chưa có sự kiện nào được ghi nhận.")

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.logger.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("QR Scanner GUI")

    initial_vid = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        initial_vid = sys.argv[1]

    gui = QRScannerGUI(initial_video=initial_vid)
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
