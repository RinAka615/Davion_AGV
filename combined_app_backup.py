#!/usr/bin/env python3
"""
Ứng dụng Tích hợp Đa nhiệm: Dò Đường (Line/Path Tracking) & Quét Mã QR (QR Mission Checkpoint)
Phát triển cho Robot Nông Nghiệp trong nhà màng (Ubuntu/Linux).
Sử dụng: AI YOLO Segmentation (PyTorch/CUDA) + OpenCV + ZXing-C++ + PyQt5.
"""

import os
import sys
import time
from typing import Optional, List, Dict, Any, Tuple

# Khắc phục triệt để lỗi Qt xcb conflict do opencv-python chèn plugin cv2/qt/plugins
import cv2
import numpy as np

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
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QCheckBox, QFileDialog, QMessageBox,
    QGroupBox, QFrame, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QTextEdit, QStatusBar, QComboBox, QDoubleSpinBox,
    QSpinBox, QTabWidget, QProgressBar, QScrollArea, QGridLayout
)

# Thêm thư mục hiện tại vào sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Import module Line Tracking
from src.video_reader import find_orbbec_camera_index
from src.yolo_segmenter import YoloSegmenter, is_cuda_available_and_working
from src.path_detector import PathDetector, TrackingState, DetectionResult
from src.main_window import draw_dashed_polyline, draw_dashed_line
from src.uart_controller import UartController
from src.navigation_fusion import (
    FusedNavigationTarget,
    fuse_navigation_target,
    check_qr_exceeds_threshold,
    QRTurnMissionManager,
    QRTurnMissionState
)

# Import module QR Scanner
from qr_scanner import QRDetector, QRJsonLogger, QRVisualizer, QRDetectionResult


COMBINED_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #16161e;
    color: #c0caf5;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #2f354a;
    border-radius: 8px;
    background-color: #1a1b26;
    top: -1px;
}
QTabBar::tab {
    background-color: #1f2335;
    color: #7aa2f7;
    border: 1px solid #2f354a;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 16px;
    margin-right: 4px;
    font-weight: bold;
}
QTabBar::tab:selected {
    background-color: #24283b;
    color: #7dcfff;
    border-color: #7aa2f7;
}
QGroupBox {
    border: 1px solid #2f354a;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
    padding-bottom: 4px;
    font-weight: bold;
    color: #7aa2f7;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    background-color: #1a1b26;
}
QPushButton {
    background-color: #24283b;
    border: 1px solid #414868;
    border-radius: 5px;
    padding: 5px 10px;
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
QComboBox, QDoubleSpinBox {
    background-color: #1f2335;
    border: 1px solid #414868;
    border-radius: 4px;
    padding: 4px 8px;
    color: #c0caf5;
}
QTableWidget {
    background-color: #15161e;
    border: 1px solid #2f354a;
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
QTextEdit {
    background-color: #15161e;
    border: 1px solid #2f354a;
    border-radius: 6px;
    color: #73daca;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 13px;
}
"""


class CombinedRobotApp(QMainWindow):
    def __init__(self, initial_video: Optional[str] = None, initial_uart: bool = False):
        super().__init__()
        self.setWindowTitle("Robot Nông Nghiệp: Hệ Thống Tích Hợp Dò Đường & Quét Mã QR")
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            self.resize(min(avail.width(), 1280), min(avail.height(), 720))
        else:
            self.resize(1200, 680)
        self.setMinimumSize(880, 480)
        self.setStyleSheet(COMBINED_STYLESHEET)

        # Trạng thái capture & video
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_camera = False
        self.total_frames = 0
        self.current_frame_idx = 0
        self.fps = 30.0
        self.is_playing = False
        self.last_frame_bgr: Optional[np.ndarray] = None
        self.camera_start_time = 0.0

        # --- 0. TÍNH NĂNG QUAY / GHI HÌNH VIDEO (VIDEO RECORDER) ---
        self.is_recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.record_start_time = 0.0
        self.record_frame_count = 0
        self.record_filepath: Optional[str] = None

        # --- 1. KHỞI TẠO ĐỘNG CƠ DÒ ĐƯỜNG (LINE TRACKING) ---
        self.yolo = YoloSegmenter()
        self.path_detector = PathDetector()
        self.cuda_available = is_cuda_available_and_working()
        self.device = "cuda" if self.cuda_available else "cpu"

        # Tự động nạp model bạt đen tốt nhất
        self._load_best_yolo_model()

        # --- 2. KHỞI TẠO ĐỘNG CƠ QUÉT MÃ QR ---
        self.qr_detector = QRDetector(backend="auto", enable_dewarp=True)
        self.qr_visualizer = QRVisualizer()
        self.logger = QRJsonLogger(log_path="output_qr/combined_mission_events.jsonl", dedup_cooldown_sec=2.0)

        # --- 3. ĐIỀU KHIỂN TRUYỀN THÔNG UART STM32 ---
        self.uart = UartController(port="/dev/ttySTM32", baudrate=115200, is_enabled=initial_uart)
        if initial_uart:
            self.uart.connect()

        # Timer vòng lặp hiển thị
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_tick)

        # --- 4. MÁY TRẠNG THÁI NHIỆM VỤ CHUYỂN HƯỚNG QR (TURN 1 -> TURN 2) ---
        self.qr_mission = QRTurnMissionManager(
            stop_duration_sec=1.0,
            align_tolerance_px=180.0,
            min_stop_qr_dim=200.0,
            auto_reset_delay_sec=3.0,
            path_seek_tolerance_px=200.0,
            path_seek_max_angle_deg=20.0,
            delay_turn1_sec=1.0,
            delay_turn2_sec=1.0,
            confirm_frames=2
        )

        self._init_ui()

        # Nạp video mẫu
        if initial_video and os.path.exists(initial_video):
            self.load_source(initial_video)
        else:
            default_vid = os.path.join(CURRENT_DIR, "nhamang_20261025_030235.mp4")
            test_vid = os.path.join(CURRENT_DIR, "test_qr_dataset", "test_qr_multiscene.mp4")
            if os.path.exists(default_vid):
                self.load_source(default_vid)
            elif os.path.exists(test_vid):
                self.load_source(test_vid)

    def _load_best_yolo_model(self):
        models = [
            os.path.join(CURRENT_DIR, "models", "yolo_fastestv2_int8.param"),
            os.path.join(CURRENT_DIR, "models", "fast_scnn_int8.param"),
            os.path.join(CURRENT_DIR, "models", "picodet_int8.param"),
            os.path.join(CURRENT_DIR, "models", "model_int8.param"),
            os.path.join(CURRENT_DIR, "models", "nhamang_batden_seg.pt"),
            os.path.join(CURRENT_DIR, "models", "thamden_seg.pt")
        ]
        for m in models:
            if os.path.exists(m):
                success = self.yolo.load_model(m, device=self.device)
                if success:
                    print(f"[COMBINED APP] Đã nạp thành công model YOLO ({self.device}): {os.path.basename(m)}")
                    break

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ==========================================
        # PANEL TRÁI: VIDEO VIEWPORT & PLAYBACK CONTROLS
        # ==========================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # Toolbar nguồn
        src_bar = QHBoxLayout()
        btn_open_file = QPushButton("📂 Chọn File Video...")
        btn_open_file.clicked.connect(self._choose_video_file)

        self.btn_open_cam = QPushButton("📷 Mở Webcam (Live Stream)")
        self.btn_open_cam.clicked.connect(self._open_webcam)

        self.btn_snapshot = QPushButton("📸 Chụp Ảnh Frame")
        self.btn_snapshot.clicked.connect(self._take_snapshot)

        self.btn_record_video = QPushButton("🎥 Bật Ghi Video")
        self.btn_record_video.setToolTip("Bật hoặc Dừng quay video hành trình kèm thông số HUD")
        self.btn_record_video.clicked.connect(self._toggle_record_video)

        self.lbl_record_badge = QLabel("")
        self.lbl_record_badge.setStyleSheet("color: #f7768e; font-weight: bold; padding: 0 4px;")

        self.lbl_source_info = QLabel("Chưa mở nguồn nào")
        self.lbl_source_info.setStyleSheet("color: #bb9af7; font-weight: bold;")

        src_bar.addWidget(btn_open_file)
        src_bar.addWidget(self.btn_open_cam)
        src_bar.addWidget(self.btn_snapshot)
        src_bar.addWidget(self.btn_record_video)
        src_bar.addWidget(self.lbl_record_badge)
        src_bar.addWidget(self.lbl_source_info)
        src_bar.addStretch()
        left_layout.addLayout(src_bar)

        # Màn hình Video
        self.lbl_video = QLabel("Vui lòng mở video hoặc webcam")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setMinimumSize(360, 220)
        self.lbl_video.setStyleSheet("background-color: #101015; border: 2px solid #24283b; border-radius: 8px;")
        left_layout.addWidget(self.lbl_video, stretch=1)

        # Timeline Slider
        timeline_bar = QHBoxLayout()
        timeline_bar.setContentsMargins(0, 2, 0, 2)
        self.slider_timeline = QSlider(Qt.Horizontal)
        self.slider_timeline.setRange(0, 0)
        self.slider_timeline.sliderMoved.connect(self._on_timeline_slider_moved)
        self.slider_timeline.sliderPressed.connect(self._on_timeline_pressed)

        self.lbl_time_info = QLabel("00:00 / 00:00 (Frame 0/0)")
        self.lbl_time_info.setStyleSheet("color: #a9b1d6; font-family: monospace; font-size: 11px;")

        timeline_bar.addWidget(self.slider_timeline)
        timeline_bar.addWidget(self.lbl_time_info)
        left_layout.addLayout(timeline_bar)

        # Điều khiển phát & Nút nhanh UART tích hợp cùng dòng (tiết kiệm không gian chiều dọc)
        ctrl_bar = QHBoxLayout()
        ctrl_bar.setContentsMargins(0, 2, 0, 2)
        ctrl_bar.setSpacing(6)
        self.btn_step_back = QPushButton("◀ 1 Frame")
        self.btn_step_back.clicked.connect(self._step_backward)

        self.btn_play_pause = QPushButton("▶ Phát Video")
        self.btn_play_pause.setObjectName("btnPlay")
        self.btn_play_pause.clicked.connect(self._toggle_play_pause)

        self.btn_step_fwd = QPushButton("1 Frame ▶")
        self.btn_step_fwd.clicked.connect(self._step_forward)

        self.btn_quick_uart = QPushButton("📡 BẬT UART (STM32)")
        self.btn_quick_uart.setCheckable(True)
        self.btn_quick_uart.setChecked(self.uart.is_enabled)
        self.btn_quick_uart.setToolTip("Bấm để BẬT hoặc TẮT truyền tín hiệu điều khiển 5 byte (mỗi 0.1s) xuống STM32")
        self.btn_quick_uart.setStyleSheet("background-color: #37474f; color: #ffb74d; font-weight: bold; font-size: 11px; padding: 4px 8px; border: 1px solid #ff9800; border-radius: 4px;")
        self.btn_quick_uart.clicked.connect(self._toggle_quick_uart)

        self.lbl_fps_badge = QLabel("FPS: --")
        self.lbl_fps_badge.setStyleSheet("color: #e0af68; font-weight: bold; padding: 0 8px; font-size: 11px;")

        ctrl_bar.addWidget(self.btn_step_back)
        ctrl_bar.addWidget(self.btn_play_pause)
        ctrl_bar.addWidget(self.btn_step_fwd)
        ctrl_bar.addWidget(self.btn_quick_uart)
        ctrl_bar.addStretch()
        ctrl_bar.addWidget(self.lbl_fps_badge)
        left_layout.addLayout(ctrl_bar)

        splitter.addWidget(left_widget)

        # ==========================================
        # PANEL PHẢI: TAB CONTROL DÒ ĐƯỜNG & QUÉT MÃ QR (CUỘN THÔNG MINH)
        # ==========================================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        self.tab_widget = QTabWidget()

        def _make_scrollable_tab(widget_content):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll.setStyleSheet("""
                QScrollArea {
                    border: none;
                    background-color: transparent;
                }
                QScrollBar:vertical {
                    border: none;
                    background: #1a1b26;
                    width: 8px;
                    margin: 0px;
                }
                QScrollBar::handle:vertical {
                    background: #3d59a1;
                    min-height: 25px;
                    border-radius: 4px;
                }
                QScrollBar::handle:vertical:hover {
                    background: #7aa2f7;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                }
            """)
            scroll.setWidget(widget_content)
            return scroll

        # TAB 1: 🌿 DÒ ĐƯỜNG (LINE TRACKING)
        tab_line = QWidget()
        tab_line_layout = QVBoxLayout(tab_line)
        tab_line_layout.setContentsMargins(6, 4, 6, 4)
        tab_line_layout.setSpacing(6)

        # Group 1.1: Trạng thái bám đường & Điều hướng
        grp_nav = QGroupBox("🧭 Trạng Thái Bám Đường (Navigation Telemetry)")
        grp_nav_layout = QVBoxLayout(grp_nav)
        grp_nav_layout.setContentsMargins(8, 8, 8, 8)
        grp_nav_layout.setSpacing(4)

        self.lbl_line_state = QLabel("Trạng thái: Đang chờ...")
        self.lbl_line_state.setStyleSheet("color: #00E676; font-weight: bold; font-size: 14px;")
        grp_nav_layout.addWidget(self.lbl_line_state)

        telemetry_row = QHBoxLayout()
        self.lbl_line_offset = QLabel("Độ lệch tâm: 0.0 px (Đúng tâm)")
        self.lbl_line_offset.setStyleSheet("color: #7dcfff; font-size: 12px; font-weight: bold;")
        self.lbl_steering_action = QLabel("Lệnh lái: GIỮ THẲNG")
        self.lbl_steering_action.setStyleSheet("color: #bb9af7; font-weight: bold; font-size: 12px;")
        telemetry_row.addWidget(self.lbl_line_offset)
        telemetry_row.addStretch()
        telemetry_row.addWidget(self.lbl_steering_action)
        grp_nav_layout.addLayout(telemetry_row)

        self.lbl_active_nav_source = QLabel("Mục tiêu hiện tại: 🌿 Tâm lối đi (Path)")
        self.lbl_active_nav_source.setStyleSheet("color: #7aa2f7; font-size: 12px; font-weight: bold;")
        grp_nav_layout.addWidget(self.lbl_active_nav_source)

        uart_row = QHBoxLayout()
        self.chk_uart_enable = QCheckBox("📡 Gửi UART STM32 (/dev/ttySTM32)")
        self.chk_uart_enable.setToolTip("Bật gửi gói 5 byte (Offset + Detect) mỗi 0.1s xuống vi điều khiển STM32")
        self.chk_uart_enable.toggled.connect(self._on_chk_uart_toggled)
        self.lbl_uart_info = QLabel("UART: Tắt")
        self.lbl_uart_info.setStyleSheet("color: #565f89; font-size: 11px;")
        uart_row.addWidget(self.chk_uart_enable)
        uart_row.addStretch()
        uart_row.addWidget(self.lbl_uart_info)
        grp_nav_layout.addLayout(uart_row)

        tab_line_layout.addWidget(grp_nav)

        # Group 1.2: Cài đặt mô hình AI Line
        grp_line_cfg = QGroupBox("⚙️ Cấu Hình Mô Hình Dò Đường")
        grp_line_cfg_layout = QVBoxLayout(grp_line_cfg)
        grp_line_cfg_layout.setContentsMargins(8, 8, 8, 8)
        grp_line_cfg_layout.setSpacing(4)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.combo_model = QComboBox()
        self.combo_model.addItems([
            "YOLO-Fastest V2 (NCNN INT8)",
            "Fast-SCNN (NCNN INT8)",
            "PicoDet (NCNN INT8)",
            "MobileNetV3-UNet (NCNN INT8)",
            "YOLOv8 Seg (.pt)",
            "Thảm đen (.pt)"
        ])
        self.combo_model.currentIndexChanged.connect(self._on_model_combo_changed)
        model_row.addWidget(self.combo_model, stretch=1)
        self.lbl_dev_badge = QLabel(f"{'CUDA GPU' if self.cuda_available else 'CPU'}")
        self.lbl_dev_badge.setStyleSheet("color: #9ece6a; font-weight: bold; padding-left: 6px;")
        model_row.addWidget(self.lbl_dev_badge)
        grp_line_cfg_layout.addLayout(model_row)

        conf_row = QHBoxLayout()
        self.lbl_line_conf = QLabel("Ngưỡng Line: 25%")
        self.lbl_line_conf.setStyleSheet("font-size: 11px;")
        self.slider_line_conf = QSlider(Qt.Horizontal)
        self.slider_line_conf.setRange(10, 90)
        self.slider_line_conf.setValue(25)
        self.slider_line_conf.valueChanged.connect(self._on_line_conf_changed)
        conf_row.addWidget(self.lbl_line_conf)
        conf_row.addWidget(self.slider_line_conf, stretch=1)
        grp_line_cfg_layout.addLayout(conf_row)

        # Tùy chọn hiển thị (Xếp lưới 2x2 gọn gàng)
        self.chk_show_mask = QCheckBox("Mask màu")
        self.chk_show_mask.setChecked(True)
        self.chk_show_mask.toggled.connect(self._reprocess_current_frame)

        self.chk_show_centerline = QCheckBox("Đường tim")
        self.chk_show_centerline.setChecked(True)
        self.chk_show_centerline.toggled.connect(self._reprocess_current_frame)

        self.chk_show_bbox = QCheckBox("Hộp bao")
        self.chk_show_bbox.setChecked(True)
        self.chk_show_bbox.toggled.connect(self._reprocess_current_frame)

        self.chk_show_offset = QCheckBox("Vectơ lệch")
        self.chk_show_offset.setChecked(True)
        self.chk_show_offset.toggled.connect(self._reprocess_current_frame)

        grid_checks = QGridLayout()
        grid_checks.setHorizontalSpacing(10)
        grid_checks.setVerticalSpacing(2)
        grid_checks.addWidget(self.chk_show_mask, 0, 0)
        grid_checks.addWidget(self.chk_show_centerline, 0, 1)
        grid_checks.addWidget(self.chk_show_bbox, 1, 0)
        grid_checks.addWidget(self.chk_show_offset, 1, 1)
        grp_line_cfg_layout.addLayout(grid_checks)

        tab_line_layout.addWidget(grp_line_cfg)

        # Group 1.3: Dẫn hướng ưu tiên theo Tâm QR
        grp_fusion = QGroupBox("🎯 Tự Động Bám Tâm QR (QR-Center Alignment)")
        grp_fusion_layout = QVBoxLayout(grp_fusion)
        grp_fusion_layout.setContentsMargins(8, 8, 8, 8)
        grp_fusion_layout.setSpacing(4)

        self.chk_auto_qr_nav = QCheckBox("BẬT chuyển bám Tâm QR khi đường bao > 100×100 px")
        self.chk_auto_qr_nav.setChecked(True)
        self.chk_auto_qr_nav.setStyleSheet("color: #bb9af7; font-weight: bold;")
        self.chk_auto_qr_nav.toggled.connect(self._reprocess_current_frame)
        grp_fusion_layout.addWidget(self.chk_auto_qr_nav)

        qr_sizes_row = QHBoxLayout()
        qr_sizes_row.addWidget(QLabel("Bám tâm:"))
        self.spin_qr_nav_thresh = QSpinBox()
        self.spin_qr_nav_thresh.setRange(20, 500)
        self.spin_qr_nav_thresh.setValue(100)
        self.spin_qr_nav_thresh.setSuffix(" px")
        self.spin_qr_nav_thresh.valueChanged.connect(self._reprocess_current_frame)
        qr_sizes_row.addWidget(self.spin_qr_nav_thresh)

        qr_sizes_row.addSpacing(8)
        qr_sizes_row.addWidget(QLabel("Trạm dừng (>px):"))
        self.spin_qr_stop_thresh = QSpinBox()
        self.spin_qr_stop_thresh.setRange(50, 1000)
        self.spin_qr_stop_thresh.setValue(200)
        self.spin_qr_stop_thresh.setSuffix(" px")
        self.spin_qr_stop_thresh.valueChanged.connect(self._reprocess_current_frame)
        qr_sizes_row.addWidget(self.spin_qr_stop_thresh)
        qr_sizes_row.addStretch()
        grp_fusion_layout.addLayout(qr_sizes_row)

        self.lbl_fusion_status = QLabel("Nguồn tính sai số: 🌿 Tâm lối đi (Chờ QR > 100x100, Dừng > 200x200)")
        self.lbl_fusion_status.setStyleSheet("color: #7dcfff; font-size: 11px; font-weight: bold;")
        grp_fusion_layout.addWidget(self.lbl_fusion_status)

        tab_line_layout.addWidget(grp_fusion)

        # Group 1.4: Nhiệm vụ tự động chuyển hướng QR (Turn 1 -> Turn 2)
        grp_mission = QGroupBox("🚀 Nhiệm Vụ Tự Động: TURN 1 -> Rẽ Trái -> Dừng TURN 2")
        grp_mission_layout = QVBoxLayout(grp_mission)
        grp_mission_layout.setContentsMargins(8, 8, 8, 8)
        grp_mission_layout.setSpacing(4)

        self.chk_mission_enabled = QCheckBox("BẬT Chu Trình: Dừng Turn 1 -> Rẽ Trái -> Dừng Tâm Turn 2")
        self.chk_mission_enabled.setChecked(True)
        self.chk_mission_enabled.setStyleSheet("color: #7aa2f7; font-weight: bold;")
        grp_mission_layout.addWidget(self.chk_mission_enabled)

        self.lbl_mission_status = QLabel("Nhiệm vụ QR: Sẵn sàng (Chờ tiếp cận mã TURN_1)")
        self.lbl_mission_status.setStyleSheet("color: #7dcfff; font-size: 11px; font-weight: bold;")
        grp_mission_layout.addWidget(self.lbl_mission_status)

        # Các thông số nhiệm vụ xếp trên 1 dòng ngang gọn gàng
        row_mission_params = QHBoxLayout()
        row_mission_params.setSpacing(6)

        lbl_align_tol = QLabel("🎯 Tâm TURN_2:")
        lbl_align_tol.setStyleSheet("color: #a9b1d6; font-size: 11px;")
        self.spin_align_tol = QSpinBox()
        self.spin_align_tol.setRange(20, 300)
        self.spin_align_tol.setValue(180)
        self.spin_align_tol.setSingleStep(5)
        self.spin_align_tol.setSuffix(" px")
        self.spin_align_tol.valueChanged.connect(self._on_align_tol_changed)

        lbl_path_tol = QLabel("🌱 Bắt luống:")
        lbl_path_tol.setStyleSheet("color: #a9b1d6; font-size: 11px;")
        self.spin_path_seek_tol = QSpinBox()
        self.spin_path_seek_tol.setRange(50, 500)
        self.spin_path_seek_tol.setValue(200)
        self.spin_path_seek_tol.setSingleStep(10)
        self.spin_path_seek_tol.setSuffix(" px")
        self.spin_path_seek_tol.valueChanged.connect(self._on_path_seek_tol_changed)

        lbl_auto_reset = QLabel("⏱️ Tự Reset:")
        lbl_auto_reset.setStyleSheet("color: #a9b1d6; font-size: 11px;")
        self.spin_auto_reset = QDoubleSpinBox()
        self.spin_auto_reset.setRange(0.0, 30.0)
        self.spin_auto_reset.setValue(3.0)
        self.spin_auto_reset.setSingleStep(0.5)
        self.spin_auto_reset.setSuffix(" s")
        self.spin_auto_reset.valueChanged.connect(self._on_auto_reset_changed)

        row_mission_params.addWidget(lbl_align_tol)
        row_mission_params.addWidget(self.spin_align_tol)
        row_mission_params.addWidget(lbl_path_tol)
        row_mission_params.addWidget(self.spin_path_seek_tol)
        row_mission_params.addWidget(lbl_auto_reset)
        row_mission_params.addWidget(self.spin_auto_reset)

        grp_mission_layout.addLayout(row_mission_params)

        # Hàng 2: Trễ xoay Turn 1, Trễ xoay Turn 2, Số frame xác nhận (Debounce)
        row_mission_params_2 = QHBoxLayout()
        row_mission_params_2.setSpacing(6)

        lbl_delay_turn1 = QLabel("⏱️ Trễ T1:")
        lbl_delay_turn1.setStyleSheet("color: #a9b1d6; font-size: 11px;")
        self.spin_delay_turn1 = QDoubleSpinBox()
        self.spin_delay_turn1.setRange(0.0, 5.0)
        self.spin_delay_turn1.setValue(1.0)
        self.spin_delay_turn1.setSingleStep(0.1)
        self.spin_delay_turn1.setSuffix(" s")
        self.spin_delay_turn1.valueChanged.connect(self._on_delay_turn1_changed)

        lbl_delay_turn2 = QLabel("⏱️ Trễ T2:")
        lbl_delay_turn2.setStyleSheet("color: #a9b1d6; font-size: 11px;")
        self.spin_delay_turn2 = QDoubleSpinBox()
        self.spin_delay_turn2.setRange(0.0, 5.0)
        self.spin_delay_turn2.setValue(1.0)
        self.spin_delay_turn2.setSingleStep(0.1)
        self.spin_delay_turn2.setSuffix(" s")
        self.spin_delay_turn2.valueChanged.connect(self._on_delay_turn2_changed)

        lbl_confirm_frames = QLabel("🎯 Frame:")
        lbl_confirm_frames.setStyleSheet("color: #a9b1d6; font-size: 11px;")
        self.spin_confirm_frames = QSpinBox()
        self.spin_confirm_frames.setRange(1, 10)
        self.spin_confirm_frames.setValue(2)
        self.spin_confirm_frames.setSingleStep(1)
        self.spin_confirm_frames.setSuffix(" f")
        self.spin_confirm_frames.valueChanged.connect(self._on_confirm_frames_changed)

        row_mission_params_2.addWidget(lbl_delay_turn1)
        row_mission_params_2.addWidget(self.spin_delay_turn1)
        row_mission_params_2.addWidget(lbl_delay_turn2)
        row_mission_params_2.addWidget(self.spin_delay_turn2)
        row_mission_params_2.addWidget(lbl_confirm_frames)
        row_mission_params_2.addWidget(self.spin_confirm_frames)

        grp_mission_layout.addLayout(row_mission_params_2)

        self.btn_reset_mission = QPushButton("🔄 Reset Thủ Công (Dự phòng)")
        self.btn_reset_mission.setStyleSheet("""
            QPushButton {
                background-color: #24283b;
                color: #7aa2f7;
                border: 1px solid #3d59a1;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3d59a1;
                color: #ffffff;
            }
        """)
        self.btn_reset_mission.clicked.connect(self._reset_qr_mission)
        grp_mission_layout.addWidget(self.btn_reset_mission)

        tab_line_layout.addWidget(grp_mission)
        tab_line_layout.addStretch()
        self.tab_widget.addTab(_make_scrollable_tab(tab_line), "🌿 Dò Đường (Line)")

        # TAB 2: 🎯 QUÉT MÃ QR (QR CHECKPOINT & SIZING)
        tab_qr = QWidget()
        tab_qr_layout = QVBoxLayout(tab_qr)

        # Group 2.1: Độ tin cậy QR
        grp_qr_conf = QGroupBox("🎛️ Bộ Điều Chỉnh Độ Tin Cậy QR")
        grp_qr_conf_layout = QVBoxLayout(grp_qr_conf)

        self.lbl_qr_conf_val = QLabel("Độ tin cậy QR: 40% (Độ nhạy trung bình)")
        self.lbl_qr_conf_val.setStyleSheet("color: #2ac3de; font-weight: bold;")
        grp_qr_conf_layout.addWidget(self.lbl_qr_conf_val)

        self.slider_qr_conf = QSlider(Qt.Horizontal)
        self.slider_qr_conf.setRange(5, 95)
        self.slider_qr_conf.setValue(40)
        self.slider_qr_conf.valueChanged.connect(self._on_qr_conf_changed)
        grp_qr_conf_layout.addWidget(self.slider_qr_conf)

        qr_opt_row = QHBoxLayout()
        self.chk_qr_dewarp = QCheckBox("Nắn phối cảnh (Dewarp)")
        self.chk_qr_dewarp.setChecked(True)
        self.chk_qr_dewarp.toggled.connect(self._reprocess_current_frame)

        self.chk_qr_candidates = QCheckBox("Hiện vùng nghi vấn (Cam)")
        self.chk_qr_candidates.setChecked(True)
        self.chk_qr_candidates.toggled.connect(self._reprocess_current_frame)

        qr_opt_row.addWidget(self.chk_qr_dewarp)
        qr_opt_row.addWidget(self.chk_qr_candidates)
        grp_qr_conf_layout.addLayout(qr_opt_row)
        tab_qr_layout.addWidget(grp_qr_conf)

        # Group 2.2: Nội dung QR
        grp_qr_res = QGroupBox("🎯 Nội Dung Mã QR Gần Nhất")
        grp_qr_res_layout = QVBoxLayout(grp_qr_res)

        self.txt_qr_content = QTextEdit()
        self.txt_qr_content.setReadOnly(True)
        self.txt_qr_content.setPlaceholderText("Chưa có mã QR nào được giải mã...")
        self.txt_qr_content.setMaximumHeight(85)
        grp_qr_res_layout.addWidget(self.txt_qr_content)

        qr_btn_bar = QHBoxLayout()
        self.btn_copy_qr = QPushButton("📋 Sao Chép")
        self.btn_copy_qr.setObjectName("btnCopy")
        self.btn_copy_qr.clicked.connect(self._copy_qr_content)
        self.lbl_copy_toast = QLabel("")
        self.lbl_copy_toast.setStyleSheet("color: #9ece6a; font-weight: bold;")
        qr_btn_bar.addWidget(self.btn_copy_qr)
        qr_btn_bar.addWidget(self.lbl_copy_toast)
        qr_btn_bar.addStretch()

        # Thumbnail patch nắn thẳng
        self.lbl_dewarp_patch = QLabel("Chưa có ảnh")
        self.lbl_dewarp_patch.setFixedSize(70, 70)
        self.lbl_dewarp_patch.setAlignment(Qt.AlignCenter)
        self.lbl_dewarp_patch.setStyleSheet("background-color: #101015; border: 1px dashed #414868; border-radius: 4px;")
        qr_btn_bar.addWidget(self.lbl_dewarp_patch)

        grp_qr_res_layout.addLayout(qr_btn_bar)
        tab_qr_layout.addWidget(grp_qr_res)

        # Group 2.3: Đo kích thước QR
        grp_qr_size = QGroupBox("📐 Kích Thước & Khoảng Cách")
        grp_qr_size_layout = QVBoxLayout(grp_qr_size)

        self.lbl_qr_size_px = QLabel("Kích thước: -- × -- px")
        self.lbl_qr_size_px.setStyleSheet("color: #7dcfff; font-weight: bold;")
        grp_qr_size_layout.addWidget(self.lbl_qr_size_px)

        self.lbl_qr_center = QLabel("🎯 Tâm QR: (X: --, Y: --) | Lệch Robot: --")
        self.lbl_qr_center.setStyleSheet("color: #00E676; font-weight: bold;")
        grp_qr_size_layout.addWidget(self.lbl_qr_center)

        self.lbl_qr_size_detail = QLabel("Diện tích: -- px² | Góc xoay: --°")
        self.lbl_qr_size_detail.setStyleSheet("color: #a9b1d6; font-size: 11px;")
        grp_qr_size_layout.addWidget(self.lbl_qr_size_detail)

        real_row = QHBoxLayout()
        real_row.addWidget(QLabel("Tem in:"))
        self.combo_real_size = QComboBox()
        self.combo_real_size.addItems(["50 mm (Tem 5×5cm)", "80 mm", "100 mm", "150 mm", "Tùy chỉnh (mm)"])
        self.combo_real_size.currentIndexChanged.connect(self._on_real_size_changed)
        real_row.addWidget(self.combo_real_size)

        self.spin_real_size = QDoubleSpinBox()
        self.spin_real_size.setRange(5.0, 1000.0)
        self.spin_real_size.setValue(50.0)
        self.spin_real_size.setSuffix(" mm")
        self.spin_real_size.setEnabled(False)
        self.spin_real_size.valueChanged.connect(self._reprocess_current_frame)
        real_row.addWidget(self.spin_real_size)
        grp_qr_size_layout.addLayout(real_row)

        self.lbl_dist_est = QLabel("Độ phân giải: -- | Khoảng cách: --")
        self.lbl_dist_est.setStyleSheet("color: #bb9af7; font-weight: bold;")
        grp_qr_size_layout.addWidget(self.lbl_dist_est)

        tab_qr_layout.addWidget(grp_qr_size)
        tab_qr_layout.addStretch()
        self.tab_widget.addTab(_make_scrollable_tab(tab_qr), "🎯 Quét Mã QR")

        # TAB 3: 📜 LỊCH SỬ & NHIỆM VỤ (MISSION LOG)
        tab_log = QWidget()
        tab_log_layout = QVBoxLayout(tab_log)

        self.table_history = QTableWidget(0, 5)
        self.table_history.setHorizontalHeaderLabels(["Thời Gian", "Mã QR Checkpoint", "Kích Thước", "Lệch Line (px)", "Backend"])
        self.table_history.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_history.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_history.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table_history.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_history.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table_history.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_history.cellDoubleClicked.connect(self._on_history_row_double_clicked)
        tab_log_layout.addWidget(self.table_history)

        log_btn_bar = QHBoxLayout()
        btn_clear_hist = QPushButton("🗑️ Xóa Lịch Sử")
        btn_clear_hist.clicked.connect(self._clear_history)
        btn_export_jsonl = QPushButton("💾 Mở File Log JSONL")
        btn_export_jsonl.clicked.connect(self._open_jsonl_log)
        btn_open_rec = QPushButton("🎬 Mở Thư Mục Video")
        btn_open_rec.clicked.connect(self._open_recordings_folder)
        log_btn_bar.addWidget(btn_clear_hist)
        log_btn_bar.addWidget(btn_export_jsonl)
        log_btn_bar.addWidget(btn_open_rec)
        log_btn_bar.addStretch()
        tab_log_layout.addLayout(log_btn_bar)

        self.tab_widget.addTab(tab_log, "📜 Nhiệm Vụ & Lịch Sử")

        right_layout.addWidget(self.tab_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([780, 560])

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Hệ thống tích hợp Line + QR đã sẵn sàng.")

    def _choose_video_file(self):
        if self.is_recording:
            self._stop_recording()
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
        if self.is_recording:
            self._stop_recording()
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

        # Dừng timer và giải phóng capture cũ
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.timer.stop()
        self.is_playing = False

        orbbec_idx = find_orbbec_camera_index(default=2)
        candidate_cams = [orbbec_idx]
        for idx in range(10):
            if idx not in candidate_cams and os.path.exists(f"/dev/video{idx}"):
                candidate_cams.append(idx)

        # Quét lần lượt các cổng camera khả dụng (ưu tiên Orbbec Astra Pro) và kiểm tra đọc frame thực tế
        working_cap = None
        working_idx = None
        for idx in candidate_cams:
            for api in [cv2.CAP_V4L2, cv2.CAP_ANY]:
                try:
                    test_cap = cv2.VideoCapture(idx, api)
                    if test_cap.isOpened():
                        # Cấu hình định dạng HD nếu là Orbbec
                        test_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                        test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        test_cap.set(cv2.CAP_PROP_FPS, 30)
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

            # Tự động live stream
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

        self.is_camera = False
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Không tìm thấy Webcam")
        msg.setText("<b>Không thể mở camera trên máy tính!</b>")
        info_text = (
            f"Đã kiểm tra các cổng camera: {available_cams if available_cams else 'Không có'}.<br><br>"
            "1. <b>Laptop Lenovo Legion</b>: Gạt <u>công tắc vật lý (Camera E-Shutter)</u> ở <b>cạnh bên PHẢI</b> sang vị trí <b>BẬT</b>.<br>"
            "2. <b>Phím tắt</b>: Bấm <b>F9</b> hoặc <b>Fn + F9</b> để mở khóa camera.<br>"
            "3. <b>Camera rời / Orbbec</b>: Kiểm tra cắm lại cáp USB."
        )
        msg.setInformativeText(info_text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        self.lbl_source_info.setText("Chưa tìm thấy Webcam. Hãy gạt công tắc camera ở cạnh phải máy.")

    def load_source(self, source_path: str):
        if self.is_recording:
            self._stop_recording()
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
                QMessageBox.warning(self, "Lỗi", f"Không thể mở camera /dev/video{cam_idx}")
                return
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.total_frames = 0
            self.camera_start_time = time.time()
            self.slider_timeline.setEnabled(False)
            self.btn_step_back.setEnabled(False)
            self.btn_step_fwd.setEnabled(False)

            # Tự động live stream
            self.is_playing = True
            self.timer.start(30)
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
                QMessageBox.critical(self, "Lỗi", f"Không thể mở file video:\n{source_path}")
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
        self.lbl_source_info.setText(f"Nguồn: {source_title} ({w}x{h})")

        self.current_frame_idx = 0
        self._read_and_process_frame()

        # Tự động phát video ngay khi nạp để truyền tín hiệu UART liên tục
        if not self.is_camera and not self.is_playing:
            self._toggle_play_pause()

    def _toggle_play_pause(self):
        if not self.cap or not self.cap.isOpened():
            return
        if self.is_playing:
            self.is_playing = False
            self.timer.stop()
            self.btn_play_pause.setText("▶ Phát Video" if not self.is_camera else "▶ Tiếp Tục Live Stream")
            self.btn_play_pause.setStyleSheet("")
        else:
            self.is_playing = True
            interval_ms = 30 if self.is_camera else max(10, int(1000 / self.fps))
            self.timer.start(interval_ms)
            self.btn_play_pause.setText("⏸ Tạm Dừng" if not self.is_camera else "⏸ Dừng Live Stream")
            self.btn_play_pause.setStyleSheet("background-color: #f7768e; color: #15161e; font-weight: bold;")

    def _on_timer_tick(self):
        if not self.cap or not self.cap.isOpened():
            return
        ret, frame = self.cap.read()
        if not ret:
            if not self.is_camera:
                if self.is_recording:
                    self._stop_recording()
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
        fh, fw = frame.shape[:2]

        # -------------------------------------------------------------
        # 1. XỬ LÝ LINE / PATH TRACKING (YOLO SEGMENTATION)
        # -------------------------------------------------------------
        line_conf = self.slider_line_conf.value() / 100.0
        yolo_res = None
        path_res = None

        if self.yolo.is_loaded:
            target_sz = getattr(self.yolo, "fixed_imgsz", None) or 416
            yolo_res = self.yolo.predict(frame, conf=line_conf, imgsz=target_sz)
            if yolo_res.is_valid and yolo_res.chosen_mask is not None:
                path_res = self.path_detector.process(frame, yolo_res.chosen_mask, is_source_valid=True)
            else:
                path_res = self.path_detector.process(frame, None, is_source_valid=False)
        else:
            path_res = self.path_detector.process(frame, None, is_source_valid=False)

        # -------------------------------------------------------------
        # 2. XỬ LÝ QUÉT MÃ QR (QR CHECKPOINT SCANNER)
        # -------------------------------------------------------------
        qr_conf = self.slider_qr_conf.value() / 100.0
        self.qr_detector.enable_dewarp = self.chk_qr_dewarp.isChecked()

        # Tối ưu hóa nhịp quét thông minh để giữ FPS cao 35-50 FPS trên CPU Pi 5:
        # - Nếu AI YOLO phát hiện có hộp QR (khi đang quay/tiếp cận) HOẶC đang bám mã QR: Quét liên tục 100% mỗi frame.
        # - Nếu không có mã QR: Quét xen kẽ mỗi 2 frame để giữ FPS mượt mà 45-50 FPS.
        if not hasattr(self, "_qr_frame_count"):
            self._qr_frame_count = 0
            self._cached_qr_results = []
        self._qr_frame_count += 1

        ai_qr_boxes = getattr(yolo_res, "qr_boxes", None) if yolo_res is not None else None
        should_scan_qr = (
            bool(ai_qr_boxes and len(ai_qr_boxes) > 0) or
            bool(len(self._cached_qr_results) > 0) or
            (self._qr_frame_count % 2 == 0)
        )

        if should_scan_qr:
            qr_results = self.qr_detector.detect_and_decode(frame, confidence=qr_conf, ai_qr_boxes=ai_qr_boxes)
            if not self.chk_qr_candidates.isChecked():
                qr_results = [r for r in qr_results if r.is_decoded]
            self._cached_qr_results = qr_results
        else:
            qr_results = self._cached_qr_results

        elapsed_ms = (time.time() - t0) * 1000.0
        fps_current = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
        self.lbl_fps_badge.setText(f"FPS: {fps_current:.1f} ({elapsed_ms:.1f}ms)")

        # -------------------------------------------------------------
        # 3. KẾT HỢP VẼ OVERLAY LÊN ẢNH
        # -------------------------------------------------------------
        vis_frame = frame.copy()

        # 3.1 Vẽ lớp phủ Line Tracking
        if path_res is not None:
            # Vẽ Mask màu bán trong suốt
            if self.chk_show_mask.isChecked() and yolo_res and yolo_res.chosen_mask is not None:
                color_mask = np.zeros_like(frame)
                color_mask[yolo_res.chosen_mask > 128] = (255, 180, 0)  # Cyan/Blue
                cv2.addWeighted(color_mask, 0.35, vis_frame, 1.0, 0, vis_frame)

            # Vẽ Bounding Box của bạt đen
            if self.chk_show_bbox.isChecked() and path_res.bounding_box:
                bx, by, bw, bh = path_res.bounding_box
                cv2.rectangle(vis_frame, (bx, by), (bx + bw, by + bh), (0, 200, 255), 2)
                cv2.putText(vis_frame, "[LỐI ĐI BẠT ĐEN]", (bx + 5, by + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

            # Vẽ Đường tim dẫn hướng (Centerline)
            if self.chk_show_centerline.isChecked():
                if path_res.centerline_points:
                    pts = np.array(path_res.centerline_points, dtype=np.int32)
                    cv2.polylines(vis_frame, [pts], isClosed=False, color=(0, 255, 255), thickness=3, lineType=cv2.LINE_AA)
                if path_res.extrapolated_points:
                    draw_dashed_polyline(vis_frame, path_res.extrapolated_points, color=(0, 255, 255), thickness=2)

        # 3.2 HỢP NHẤT DẪN HƯỚNG: TÍNH SAI SỐ THEO TÂM QR NẾU DIỆN TÍCH ĐƯỜNG BAO > 100x100
        min_thresh = float(self.spin_qr_nav_thresh.value()) if hasattr(self, "spin_qr_nav_thresh") else 100.0
        stop_thresh = float(self.spin_qr_stop_thresh.value()) if hasattr(self, "spin_qr_stop_thresh") else 200.0
        override_enabled = self.chk_auto_qr_nav.isChecked() if hasattr(self, "chk_auto_qr_nav") else True
        fused_nav = fuse_navigation_target(
            path_res=path_res,
            qr_results=qr_results,
            frame_shape=frame.shape,
            min_qr_size=min_thresh,
            qr_stop_thresh=stop_thresh,
            enable_qr_override=override_enabled
        )

        # 3.2.1 XỬ LÝ MÁY TRẠNG THÁI NHIỆM VỤ QR TURN 1 -> TURN 2
        if hasattr(self, "qr_mission"):
            # ĐỌC PHẢN HỒI TELEMETRY TỪ STM32 QUA UART ($RC,Auto_Forward,stm_qr_state)
            uart_inst = getattr(self, "uart", None) or getattr(self, "uart_ctrl", None)
            if uart_inst and uart_inst.is_enabled:
                telemetry = uart_inst.read_telemetry()
                if telemetry:
                    # NẾU CÔNG TẮC KÊNH 5 BỊ GẠT XUỐNG LÁI TAY (Auto_Forward == 0):
                    if telemetry.get("auto_forward") == 0:
                        if self.qr_mission.state != QRTurnMissionState.IDLE:
                            self.qr_mission.reset()
                            self.lbl_mission_status.setText("Nhiệm vụ QR: Sẵn sàng (Đã reset từ công tắc tay cầm Kênh 5)")
                            self.lbl_mission_status.setStyleSheet("color: #7dcfff; font-size: 12px; font-weight: bold;")
                            self.statusBar().showMessage("🔄 Đã reset nhiệm vụ từ công tắc tay cầm Kênh 5 (Auto_Forward OFF)", 3000)

            self.qr_mission.is_enabled = self.chk_mission_enabled.isChecked() if hasattr(self, "chk_mission_enabled") else True
            if hasattr(self, "spin_qr_stop_thresh"):
                self.qr_mission.min_stop_qr_dim = float(self.spin_qr_stop_thresh.value())
            if hasattr(self, "spin_align_tol"):
                self.qr_mission.align_tolerance_px = float(self.spin_align_tol.value())
            if hasattr(self, "spin_path_seek_tol"):
                self.qr_mission.path_seek_tolerance_px = float(self.spin_path_seek_tol.value())
            if hasattr(self, "spin_path_seek_angle"):
                self.qr_mission.path_seek_max_angle_deg = float(self.spin_path_seek_angle.value())
            if hasattr(self, "spin_delay_turn1"):
                self.qr_mission.delay_turn1_sec = float(self.spin_delay_turn1.value())
            if hasattr(self, "spin_delay_turn2"):
                self.qr_mission.delay_turn2_sec = float(self.spin_delay_turn2.value())
            if hasattr(self, "spin_confirm_frames"):
                self.qr_mission.confirm_frames = int(self.spin_confirm_frames.value())
                self.qr_mission.min_path_confirm_frames = int(self.spin_confirm_frames.value())
            if self.qr_mission.state == QRTurnMissionState.TURNING_LEFT_SEEK_TURN2:
                if hasattr(self, "qr_detector") and hasattr(self.qr_detector, "last_decoded_text"):
                    if self.qr_detector.last_decoded_text and "TURN_1" in self.qr_detector.last_decoded_text.upper():
                        self.qr_detector.last_decoded_text = "TURN_2"
            fused_nav = self.qr_mission.process(
                base_nav=fused_nav,
                qr_results=qr_results,
                frame_shape=frame.shape,
                path_res=path_res
            )

        current_offset_val = fused_nav.lateral_offset_px
        cam_cx = fw // 2

        # 3.3 Vẽ Vectơ lệch tâm và điểm mục tiêu lái
        if fused_nav.target_pt:
            tx, ty = fused_nav.target_pt
            if fused_nav.is_qr_mode:
                # Chế độ dẫn hướng theo tâm QR (> 100x100 hoặc nhiệm vụ QR)
                cv2.circle(vis_frame, (tx, ty), 10, (255, 0, 255), -1)
                cv2.circle(vis_frame, (tx, ty), 18, (0, 255, 255), 2)

                if self.chk_show_offset.isChecked():
                    cv2.line(vis_frame, (cam_cx, ty), (tx, ty), (255, 0, 255), 3, cv2.LINE_AA)
                    cv2.circle(vis_frame, (cam_cx, ty), 5, (255, 255, 255), -1)
                    badge_str = f"🎯 [{fused_nav.mode_name}] {fused_nav.offset_direction}"
                    cv2.putText(vis_frame, badge_str, (max(10, cam_cx - 160), max(25, ty - 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
            else:
                # Chế độ dẫn hướng theo tâm lối đi bình thường
                cv2.circle(vis_frame, (tx, ty), 8, (0, 255, 0), -1)
                cv2.circle(vis_frame, (tx, ty), 14, (0, 255, 0), 2)

                if self.chk_show_offset.isChecked():
                    cv2.line(vis_frame, (cam_cx, ty), (tx, ty), (0, 0, 255), 3, cv2.LINE_AA)
                    cv2.circle(vis_frame, (cam_cx, ty), 4, (255, 255, 255), -1)
                    offset_txt = f"{fused_nav.offset_direction}"
                    cv2.putText(vis_frame, offset_txt, (cam_cx - 50, ty - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # Cập nhật Trạng thái Nhiệm Vụ QR (Turn 1 -> Turn 2)
        if hasattr(self, "lbl_mission_status") and fused_nav.mission_status_desc:
            self.lbl_mission_status.setText(f"Nhiệm vụ: {fused_nav.mission_status_desc}")
            if fused_nav.mission_state == QRTurnMissionState.STOP_AT_TURN1.value:
                self.lbl_mission_status.setStyleSheet("color: #ff9e64; font-weight: bold; font-size: 13px;")
            elif fused_nav.mission_state == QRTurnMissionState.TURNING_LEFT_SEEK_TURN2.value:
                self.lbl_mission_status.setStyleSheet("color: #bb9af7; font-weight: bold; font-size: 13px;")
            elif fused_nav.mission_state == QRTurnMissionState.APPROACHING_TURN2.value:
                self.lbl_mission_status.setStyleSheet("color: #7dcfff; font-weight: bold; font-size: 13px;")
            elif fused_nav.mission_state == QRTurnMissionState.COMPLETED_STOPPED.value:
                self.lbl_mission_status.setStyleSheet("color: #00E676; font-weight: bold; font-size: 13px;")
            else:
                self.lbl_mission_status.setStyleSheet("color: #7aa2f7; font-size: 12px;")

        # Cập nhật Telemetry trên giao diện
        if fused_nav.is_qr_mode:
            self.lbl_active_nav_source.setText(f"Mục tiêu hiện tại: 🎯 TÂM MÃ QR ({fused_nav.mode_name})")
            self.lbl_active_nav_source.setStyleSheet("color: #bb9af7; font-size: 13px; font-weight: bold;")
            
            if fused_nav.mission_state == QRTurnMissionState.COMPLETED_STOPPED.value:
                self.lbl_fusion_status.setText(f"🛑 HOÀN TẤT ĐẾN TRẠM TURN_2 [BYTE 4 = 1] ({fused_nav.offset_direction})")
                self.lbl_fusion_status.setStyleSheet("color: #00E676; font-size: 12px; font-weight: bold;")
            elif fused_nav.mission_state == QRTurnMissionState.APPROACHING_TURN2.value:
                if fused_nav.uart_flag_byte == 2:
                    self.lbl_fusion_status.setText(f"⏸️ TẠM DỪNG 500MS TRIỆT TIÊU QUÁN TÍNH [BYTE 4 = 2] ({fused_nav.offset_direction})")
                    self.lbl_fusion_status.setStyleSheet("color: #ff9e64; font-size: 12px; font-weight: bold;")
                else:
                    self.lbl_fusion_status.setText(f"🚀 TIẾN THEO TÂM TURN_2 [BYTE 4 = 0]: {fused_nav.qr_bbox_dim[0]:.0f}×{fused_nav.qr_bbox_dim[1]:.0f} px")
                    self.lbl_fusion_status.setStyleSheet("color: #7dcfff; font-size: 12px; font-weight: bold;")
            elif fused_nav.qr_stop_triggered:
                if fused_nav.uart_flag_byte == 2:
                    self.lbl_fusion_status.setText(f"🎉 ĐÃ BẮT TÂM TURN_2 [BYTE 4 = 2] ({fused_nav.offset_direction})")
                    self.lbl_fusion_status.setStyleSheet("color: #00E676; font-size: 12px; font-weight: bold;")
                elif fused_nav.uart_flag_byte == 1:
                    self.lbl_fusion_status.setText(f"🛑 TRẠM DỪNG TURN_1 [BYTE 4 = 1] ({fused_nav.offset_direction})")
                    self.lbl_fusion_status.setStyleSheet("color: #ff5555; font-size: 12px; font-weight: bold;")
                else:
                    self.lbl_fusion_status.setText(f"🛑 TRẠM DỪNG QR [BYTE 4 = 1] ({fused_nav.offset_direction})")
                    self.lbl_fusion_status.setStyleSheet("color: #ff5555; font-size: 12px; font-weight: bold;")
            elif fused_nav.uart_flag_byte == 2:
                self.lbl_fusion_status.setText("🎉 ĐÃ BẮT TÂM TURN_2 [BYTE 4 = 2] (Dừng xe)")
                self.lbl_fusion_status.setStyleSheet("color: #00E676; font-size: 12px; font-weight: bold;")
            else:
                self.lbl_fusion_status.setText(f"🎯 Đang bám Tâm QR [BYTE 4 = 0]: {fused_nav.qr_bbox_dim[0]:.0f}×{fused_nav.qr_bbox_dim[1]:.0f} px")
                self.lbl_fusion_status.setStyleSheet("color: #bb9af7; font-size: 12px; font-weight: bold;")

            self.lbl_line_state.setText(f"Trạng thái: 🎯 {fused_nav.mode_name}")
            self.lbl_line_state.setStyleSheet("color: #bb9af7; font-weight: bold; font-size: 15px;")
            self.lbl_line_offset.setText(f"Độ lệch tâm: {fused_nav.offset_direction}")

            # Tính toán hướng lái đề xuất
            if fused_nav.mission_state == QRTurnMissionState.COMPLETED_STOPPED.value:
                self.lbl_steering_action.setText("🛑 LỆNH LÁI: DỪNG HOÀN TẤT (Đã đến trạm TURN_2 ≥ 200px)")
                self.lbl_steering_action.setStyleSheet("color: #00E676; font-weight: bold; font-size: 13px;")
            elif fused_nav.mission_state == QRTurnMissionState.APPROACHING_TURN2.value:
                if fused_nav.uart_flag_byte == 2:
                    self.lbl_steering_action.setText("⏸️ LỆNH LÁI: TẠM DỪNG 500MS TRIỆT TIÊU QUÁN TÍNH")
                    self.lbl_steering_action.setStyleSheet("color: #ff9e64; font-weight: bold; font-size: 13px;")
                else:
                    self.lbl_steering_action.setText(f"🚀 LỆNH LÁI: TIẾN THEO TÂM TURN_2 ({fused_nav.offset_direction})")
                    self.lbl_steering_action.setStyleSheet("color: #7dcfff; font-weight: bold; font-size: 13px;")
            elif fused_nav.mission_state == QRTurnMissionState.STOP_AT_TURN1.value or (fused_nav.uart_flag_byte == 1 and fused_nav.mode_name == "TURN1_STATION"):
                self.lbl_steering_action.setText("🛑 LỆNH LÁI: GỬI B4=1 ĐẾN TRẠM TURN_1 (STM32 tự dừng 1s & xoay)")
                self.lbl_steering_action.setStyleSheet("color: #ff5555; font-weight: bold; font-size: 13px;")
            elif fused_nav.mission_state == QRTurnMissionState.TURNING_LEFT_SEEK_TURN2.value and fused_nav.uart_flag_byte != 2:
                self.lbl_steering_action.setText(f"↩️ LỆNH LÁI: STM32 ĐANG QUAY TRÁI TÌM TURN_2 ({fused_nav.offset_direction})")
                self.lbl_steering_action.setStyleSheet("color: #bb9af7; font-weight: bold; font-size: 13px;")
            elif fused_nav.qr_stop_triggered:
                self.lbl_steering_action.setText("🛑 LỆNH LÁI: DỪNG ROBOT (Trạm dừng QR > 200 px)")
                self.lbl_steering_action.setStyleSheet("color: #ff5555; font-weight: bold; font-size: 13px;")
            else:
                deg = abs(fused_nav.lateral_offset_px) / (fw * 0.5) * 15.0 if fused_nav.lateral_offset_px else 0.0
                if fused_nav.lateral_offset_px and fused_nav.lateral_offset_px < -5:
                    self.lbl_steering_action.setText(f"Lệnh lái: BẺ LÁI PHẢI (+{deg:.1f}°) [Căn tâm QR]")
                    self.lbl_steering_action.setStyleSheet("color: #7dcfff; font-weight: bold; font-size: 13px;")
                elif fused_nav.lateral_offset_px and fused_nav.lateral_offset_px > 5:
                    self.lbl_steering_action.setText(f"Lệnh lái: BẺ LÁI TRÁI (-{deg:.1f}°) [Căn tâm QR]")
                    self.lbl_steering_action.setStyleSheet("color: #bb9af7; font-weight: bold; font-size: 13px;")
                else:
                    self.lbl_steering_action.setText("Lệnh lái: GIỮ THẲNG [Đã đúng tâm QR]")
                    self.lbl_steering_action.setStyleSheet("color: #00E676; font-weight: bold; font-size: 13px;")

        elif path_res is not None and path_res.state != TrackingState.LOST:
            self.lbl_active_nav_source.setText("Mục tiêu hiện tại: 🌿 Tâm lối đi (Path Center)")
            self.lbl_active_nav_source.setStyleSheet("color: #7aa2f7; font-size: 12px; font-weight: bold;")
            if fused_nav.qr_stop_triggered:
                self.lbl_fusion_status.setText("🛑 TRẠM DỪNG QR > 200×200 [BYTE 4 = 1] | Lái theo lối đi")
                self.lbl_fusion_status.setStyleSheet("color: #ff5555; font-size: 12px; font-weight: bold;")
                self.lbl_steering_action.setText("🛑 LỆNH LÁI: DỪNG ROBOT (Trạm dừng QR > 200 px)")
                self.lbl_steering_action.setStyleSheet("color: #ff5555; font-weight: bold; font-size: 13px;")
            else:
                self.lbl_fusion_status.setText("Nguồn tính sai số: 🌿 Tâm lối đi [BYTE 4 = 0]")
                self.lbl_fusion_status.setStyleSheet("color: #7dcfff; font-size: 12px;")

            state_text = path_res.state.value
            color_state = "#00E676" if path_res.state == TrackingState.TRACKING else "#FF9100"
            self.lbl_line_state.setText(f"Trạng thái: {state_text}")
            self.lbl_line_state.setStyleSheet(f"color: {color_state}; font-weight: bold; font-size: 15px;")
            self.lbl_line_offset.setText(f"Độ lệch tâm: {path_res.offset_direction}")

            if path_res.lateral_offset_px is not None:
                deg = abs(path_res.lateral_offset_px) / (fw * 0.5) * 15.0
                if path_res.lateral_offset_px < -5:
                    self.lbl_steering_action.setText(f"Lệnh lái: BẺ LÁI PHẢI (+{deg:.1f}°)")
                    self.lbl_steering_action.setStyleSheet("color: #7dcfff; font-weight: bold; font-size: 13px;")
                elif path_res.lateral_offset_px > 5:
                    self.lbl_steering_action.setText(f"Lệnh lái: BẺ LÁI TRÁI (-{deg:.1f}°)")
                    self.lbl_steering_action.setStyleSheet("color: #bb9af7; font-weight: bold; font-size: 13px;")
                else:
                    self.lbl_steering_action.setText("Lệnh lái: GIỮ THẲNG (Đúng tâm)")
                    self.lbl_steering_action.setStyleSheet("color: #9ece6a; font-weight: bold; font-size: 13px;")
            else:
                self.lbl_steering_action.setText("Lệnh lái: CHỜ TÍN HIỆU LINE")
                self.lbl_steering_action.setStyleSheet("color: #565f89; font-weight: bold; font-size: 13px;")
        else:
            self.lbl_active_nav_source.setText("Mục tiêu hiện tại: ⚪ Mất tín hiệu")
            self.lbl_active_nav_source.setStyleSheet("color: #565f89; font-size: 12px;")
            self.lbl_fusion_status.setText("Chế độ hiện tại: ⚪ Mất tín hiệu cả lối đi và QR [BYTE 4 = 0]")
            self.lbl_fusion_status.setStyleSheet("color: #565f89; font-size: 12px;")
            self.lbl_line_state.setText("Trạng thái: Mất đường / Chờ tín hiệu")
            self.lbl_line_state.setStyleSheet("color: #ff5555; font-weight: bold; font-size: 15px;")
            self.lbl_line_offset.setText("Độ lệch tâm: N/A")
            self.lbl_steering_action.setText("Lệnh lái: DỪNG / CHỜ TÍN HIỆU")
            self.lbl_steering_action.setStyleSheet("color: #565f89; font-weight: bold; font-size: 13px;")

        # Gửi dữ liệu điều khiển qua UART nếu được bật
        if hasattr(self, "uart") and self.uart and self.uart.is_enabled:
            # Byte 4: Giá trị số nguyên từ fused_nav (0: Chạy, 1: Dừng, 2: Rẽ trái, 3: Căn tâm)
            flag_val = int(fused_nav.uart_flag_byte)
            conf_val = 98.0 if fused_nav.is_qr_mode else (path_res.confidence if path_res else 0.0)
            tx = fused_nav.target_pt[0] if fused_nav.target_pt else None
            ty = fused_nav.target_pt[1] if fused_nav.target_pt else None
            state_val = self.uart.STATE_TRACKING if (fused_nav.target_pt is not None) else self.uart.STATE_LOST

            self.uart.send_nav_update(
                state_code=state_val,
                offset_px=fused_nav.lateral_offset_px,
                confidence_pct=conf_val,
                target_x=tx,
                target_y=ty,
                is_detected=flag_val
            )
            pkt_txt = self.uart.last_sent_packet or ""
            if hasattr(self, "lbl_uart_info"):
                flag_str_map = {0: "0 (TIẾN)", 1: "1 (TRẠM DỪNG)", 2: "2 (TÂM DỪNG 500MS)"}
                flag_str = flag_str_map.get(flag_val, f"{flag_val}")
                self.lbl_uart_info.setText(f"UART [B4={flag_str}]: {pkt_txt.strip()[:24]}")

        # 3.4 Vẽ mã QR trực quan
        vis_frame = self.qr_visualizer.draw_results(vis_frame, qr_results)

        # 3.5 Vẽ HUD Telemetry tích hợp trên cùng
        hud_h = 42
        hud_overlay = vis_frame.copy()
        cv2.rectangle(hud_overlay, (0, 0), (fw, hud_h), (20, 20, 20), -1)
        cv2.addWeighted(hud_overlay, 0.7, vis_frame, 0.3, 0, vis_frame)
        if fused_nav.is_qr_mode:
            line_stat_short = f"🎯 QR-CENTER ({fused_nav.qr_bbox_dim[0]:.0f}x{fused_nav.qr_bbox_dim[1]:.0f})"
        else:
            line_stat_short = f"🌿 LINE: {path_res.state.name if path_res else 'N/A'}"
        qr_count = len([r for r in qr_results if r.is_decoded])
        qr_stat_short = f"QR: {qr_count} mã"
        stop_badge_map = {
            0: "⚪ B4=0 (CHẠY)",
            1: "🛑 B4=1 (TRẠM TURN_1)",
            2: "🎉 B4=2 (TÂM TURN_2)",
            3: "↩️ B4=3 (XOAY TRÁI TÌM ĐƯỜNG)"
        }
        stop_badge = stop_badge_map.get(fused_nav.uart_flag_byte, f"B4={fused_nav.uart_flag_byte}")
        hud_txt = f"ROBOT AGV HUD | {line_stat_short} ({fused_nav.offset_direction}) | {stop_badge} | {qr_stat_short} | FPS: {fps_current:.1f}"
        cv2.putText(vis_frame, hud_txt, (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # HUD hiển thị trạng thái nhiệm vụ QR Turn 1 -> Turn 2 nếu đang hoạt động
        if hasattr(fused_nav, "mission_state") and fused_nav.mission_state != QRTurnMissionState.IDLE.value and fused_nav.mission_status_desc:
            overlay_h = 36
            banner_bg = (30, 20, 60)
            if fused_nav.mission_state == QRTurnMissionState.STOP_AT_TURN1.value:
                banner_bg = (20, 40, 80)
            elif fused_nav.mission_state == QRTurnMissionState.TURNING_RIGHT_SEEK_PATH.value:
                banner_bg = (60, 40, 20)
            elif fused_nav.mission_state == QRTurnMissionState.COMPLETED_STOPPED.value:
                banner_bg = (20, 80, 40)
            cv2.rectangle(vis_frame, (10, 48), (fw - 10, 48 + overlay_h), banner_bg, -1)
            cv2.rectangle(vis_frame, (10, 48), (fw - 10, 48 + overlay_h), (187, 154, 247), 2)
            cv2.putText(vis_frame, f"[NHIỆM VỤ QR] {fused_nav.mission_status_desc}", (20, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)

        # 3.6 Nếu đang Ghi Video: Vẽ ký hiệu REC nhấp nháy và lưu frame vào file MP4
        if self.is_recording:
            rec_sec = int(time.time() - self.record_start_time)
            mins, secs = divmod(rec_sec, 60)
            rec_time_str = f"{mins:02d}:{secs:02d}"
            # Hiệu ứng nhấp nháy chấm tròn đỏ mỗi 0.5s
            blink_on = (int((time.time() - self.record_start_time) * 2) % 2) == 0
            dot_color = (0, 0, 255) if blink_on else (80, 80, 80)
            rec_x = fw - 165
            if rec_x > 250:
                cv2.circle(vis_frame, (rec_x, 21), 6, dot_color, -1)
                cv2.putText(vis_frame, f"REC {rec_time_str}", (rec_x + 12, 27),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
            self.btn_record_video.setText(f"🔴 Dừng Ghi ({rec_time_str})")
            self.lbl_record_badge.setText(f"● REC {rec_time_str} ({self.record_frame_count}f)")
            # Khởi tạo VideoWriter với kích thước chuẩn của frame
            if self.video_writer is None and self.record_filepath:
                rec_fps = self.fps if (not self.is_camera and self.fps > 0) else 30.0
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.record_filepath, fourcc, float(rec_fps), (fw, fh))
            if self.video_writer is not None and self.video_writer.isOpened():
                if vis_frame.shape[1] == fw and vis_frame.shape[0] == fh:
                    self.video_writer.write(vis_frame)
                    self.record_frame_count += 1

        # Hiển thị lên QLabel
        self._display_video_frame(vis_frame)

        # -------------------------------------------------------------
        # 4. CẬP NHẬT KẾT QUẢ QR VÀ ĐO KÍCH THƯỚC TRÊN GIAO DIỆN
        # -------------------------------------------------------------
        decoded_qrs = [r for r in qr_results if r.is_decoded]
        candidate_qrs = [r for r in qr_results if not r.is_decoded]

        if decoded_qrs:
            first_qr = decoded_qrs[0]
            self.txt_qr_content.setText(first_qr.text)
            self.txt_qr_content.setStyleSheet("background-color: #101015; border: 2px solid #00E676; color: #00E676; font-weight: bold;")
            if first_qr.dewarped_patch is not None:
                self._display_patch(first_qr.dewarped_patch)

            # Ghi nhận vào bảng Lịch sử
            vid_time = (self.current_frame_idx / self.fps) if not self.is_camera else None
            for dec in decoded_qrs:
                size_str = f"{dec.width_px:.0f}×{dec.height_px:.0f} px"
                logged = self.logger.log(
                    text=dec.text,
                    corners=dec.points,
                    backend=dec.backend,
                    video_timestamp_sec=vid_time,
                    barcode_format=dec.format_name,
                    geometry=dec.geometry
                )
                if logged:
                    self._add_to_history(dec.text, vid_time, size_str, current_offset_val, dec.backend, is_qr_mode=fused_nav.is_qr_mode)

        else:
            if candidate_qrs:
                cand = candidate_qrs[0]
                self.txt_qr_content.setText(f"[VÙNG NGHI VẤN CHƯA GIẢI MÃ]\nLý do: {cand.error}")
                self.txt_qr_content.setStyleSheet("background-color: #101015; border: 2px solid #FF9100; color: #FF9100;")
                if cand.dewarped_patch is not None:
                    self._display_patch(cand.dewarped_patch)
            else:
                self.txt_qr_content.setText("Không phát hiện mã QR trong khung hình này.")
                self.txt_qr_content.setStyleSheet("background-color: #101015; border: 1px solid #2f354a; color: #565f89;")

        # Đo kích thước QR
        target_qr = decoded_qrs[0] if decoded_qrs else (candidate_qrs[0] if candidate_qrs else None)
        if target_qr is not None:
            w_px, h_px = target_qr.width_px, target_qr.height_px
            geom = target_qr.geometry or {}
            cx, cy = target_qr.center
            dx = cx - (fw / 2.0)
            dy = cy - (fh / 2.0)
            dir_str = "Lệch Phải" if dx > 5 else ("Lệch Trái" if dx < -5 else "Chuẩn Tâm")

            self.lbl_qr_size_px.setText(f"Kích thước Pixel: {w_px:.1f} × {h_px:.1f} px")
            self.lbl_qr_center.setText(f"🎯 Tâm QR: ({cx:.1f}, {cy:.1f}) | Lệch: ΔX: {dx:+.1f}px ({dir_str}), ΔY: {dy:+.1f}px")
            self.lbl_qr_size_detail.setText(f"Diện tích: {target_qr.area_px:,.0f} px² | Tỷ lệ: {target_qr.aspect_ratio:.2f} | Góc xoay: {target_qr.orientation_deg:+.1f}°")

            real_size_mm = self._get_selected_real_size_mm()
            if w_px > 3:
                px_per_mm = w_px / real_size_mm
                focal_px = fw * 0.95
                dist_cm = ((focal_px * real_size_mm) / w_px) / 10.0
                self.lbl_dist_est.setText(f"Độ phân giải: {px_per_mm:.2f} px/mm | Khoảng cách: ~ {dist_cm:.1f} cm ({dist_cm/100:.2f} m)")
        else:
            self.lbl_qr_size_px.setText("Kích thước Pixel: -- × -- px")
            self.lbl_qr_center.setText("🎯 Tâm QR: (X: --, Y: --) | Lệch Robot: --")
            self.lbl_qr_size_detail.setText("Diện tích: -- px² | Góc xoay: --°")
            self.lbl_dist_est.setText("Độ phân giải: -- | Khoảng cách: --")

        # Cập nhật timeline slider
        if not self.is_camera and self.total_frames > 0:
            self.slider_timeline.blockSignals(True)
            self.slider_timeline.setValue(self.current_frame_idx)
            self.slider_timeline.blockSignals(False)

            cur_sec = self.current_frame_idx / self.fps
            tot_sec = self.total_frames / self.fps
            self.lbl_time_info.setText(
                f"{int(cur_sec//60):02d}:{cur_sec%60:04.1f} / {int(tot_sec//60):02d}:{tot_sec%60:04.1f} (Frame {self.current_frame_idx}/{self.total_frames})"
            )
        elif self.is_camera:
            live_sec = (time.time() - self.camera_start_time)
            self.lbl_time_info.setText(f"🔴 LIVE: {int(live_sec//60):02d}:{int(live_sec%60):02d} (Frame #{self.current_frame_idx})")

    def _display_video_frame(self, frame_bgr: np.ndarray):
        rgb_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        lbl_size = self.lbl_video.size()
        pixmap = QPixmap.fromImage(q_img).scaled(lbl_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_video.setPixmap(pixmap)

    def _display_patch(self, patch_bgr: np.ndarray):
        rgb_patch = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        ph, pw, pch = rgb_patch.shape
        q_img = QImage(rgb_patch.data, pw, ph, pch * pw, QImage.Format_RGB888)
        pix = QPixmap.fromImage(q_img).scaled(self.lbl_dewarp_patch.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.lbl_dewarp_patch.setPixmap(pix)

    def _on_model_combo_changed(self, idx: int):
        model_names = [
            "yolo_fastestv2_int8.param",
            "fast_scnn_int8.param",
            "picodet_int8.param",
            "model_int8.param",
            "nhamang_batden_seg.pt",
            "thamden_seg.pt"
        ]
        if idx < len(model_names):
            m_path = os.path.join(CURRENT_DIR, "models", model_names[idx])
            if os.path.exists(m_path):
                self.yolo.load_model(m_path, device=self.device)
                self.status_bar.showMessage(f"Đã chuyển model Line sang: {model_names[idx]}")
                self._reprocess_current_frame()

    def _on_line_conf_changed(self, val: int):
        self.lbl_line_conf.setText(f"Ngưỡng tin cậy Line: {val}%")
        self._reprocess_current_frame()

    def _on_qr_conf_changed(self, val: int):
        self.lbl_qr_conf_val.setText(f"Độ tin cậy QR: {val}%")
        self._reprocess_current_frame()

    def _get_selected_real_size_mm(self) -> float:
        idx = self.combo_real_size.currentIndex()
        presets = [50.0, 80.0, 100.0, 150.0]
        if idx < len(presets):
            return presets[idx]
        return float(self.spin_real_size.value())

    def _on_real_size_changed(self, idx: int):
        is_custom = (idx == 4)
        self.spin_real_size.setEnabled(is_custom)
        self._reprocess_current_frame()

    def _reprocess_current_frame(self):
        if not self.is_playing and self.last_frame_bgr is not None:
            self._process_frame(self.last_frame_bgr)

    def _reset_qr_mission(self):
        if hasattr(self, "qr_mission"):
            self.qr_mission.reset()
        if hasattr(self, "qr_detector") and hasattr(self.qr_detector, "reset_last_decoded_text"):
            self.qr_detector.reset_last_decoded_text("TURN_2")
        if hasattr(self, "lbl_mission_status"):
            self.lbl_mission_status.setText("Nhiệm vụ QR: Đã reset (Chờ tiếp cận mã TURN_1)")
            self.lbl_mission_status.setStyleSheet("color: #7dcfff; font-size: 12px; font-weight: bold;")
        self.statusBar().showMessage("🔄 Đã reset nhiệm vụ chuyển hướng QR (Turn 1 -> Turn 2)", 3000)

    def _on_auto_reset_changed(self, val: float):
        if hasattr(self, "qr_mission"):
            self.qr_mission.auto_reset_delay_sec = float(val)

    def _on_align_tol_changed(self, val: int):
        if hasattr(self, "qr_mission") and self.qr_mission:
            self.qr_mission.align_tolerance_px = float(val)
        self._reprocess_current_frame()

    def _on_path_seek_tol_changed(self, val: int):
        if hasattr(self, "qr_mission") and self.qr_mission:
            self.qr_mission.path_seek_tolerance_px = float(val)
        self._reprocess_current_frame()

    def _on_path_seek_angle_changed(self, val: int):
        if hasattr(self, "qr_mission") and self.qr_mission:
            self.qr_mission.path_seek_max_angle_deg = float(val)
        self._reprocess_current_frame()

    def _on_delay_turn1_changed(self, val: float):
        if hasattr(self, "qr_mission") and self.qr_mission:
            self.qr_mission.delay_turn1_sec = float(val)

    def _on_delay_turn2_changed(self, val: float):
        if hasattr(self, "qr_mission") and self.qr_mission:
            self.qr_mission.delay_turn2_sec = float(val)

    def _on_confirm_frames_changed(self, val: int):
        if hasattr(self, "qr_mission") and self.qr_mission:
            self.qr_mission.confirm_frames = int(val)
            self.qr_mission.min_path_confirm_frames = int(val)

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

    def _copy_qr_content(self):
        text = self.txt_qr_content.toPlainText().strip()
        if text and not text.startswith("[VÙNG NGHI VẤN") and not text.startswith("Không phát hiện"):
            QApplication.clipboard().setText(text)
            self.lbl_copy_toast.setText("✔ Đã sao chép!")
            QTimer.singleShot(2000, lambda: self.lbl_copy_toast.setText(""))
        else:
            self.lbl_copy_toast.setText("⚠️ Chưa có nội dung mã!")
            QTimer.singleShot(2000, lambda: self.lbl_copy_toast.setText(""))

    def _take_snapshot(self):
        if self.last_frame_bgr is None:
            return
        snap_dir = os.path.join(CURRENT_DIR, "output_qr", "snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        filepath = os.path.join(snap_dir, f"combined_snap_{int(time.time()*1000)}.jpg")
        cv2.imwrite(filepath, self.last_frame_bgr)
        self.status_bar.showMessage(f"✔ Đã chụp và lưu ảnh vào: {filepath}")

    def _toggle_record_video(self):
        """Bật hoặc Dừng quay video hành trình kèm HUD."""
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if not self.cap or not self.cap.isOpened():
            QMessageBox.warning(self, "Thông báo", "Vui lòng mở video hoặc webcam trước khi quay video!")
            return

        rec_dir = os.path.join(CURRENT_DIR, "output_qr", "recordings")
        os.makedirs(rec_dir, exist_ok=True)
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        self.record_filepath = os.path.join(rec_dir, f"robot_record_{timestamp_str}.mp4")

        self.video_writer = None
        self.record_start_time = time.time()
        self.record_frame_count = 0
        self.is_recording = True

        self.btn_record_video.setText("🔴 Dừng Ghi (00:00)")
        self.btn_record_video.setStyleSheet("background-color: #f7768e; color: #15161e; font-weight: bold;")
        self.lbl_record_badge.setText("● REC")
        self.status_bar.showMessage(f"🔴 Bắt đầu ghi hình video: {os.path.basename(self.record_filepath)}")

    def _stop_recording(self):
        if not self.is_recording and self.video_writer is None:
            return

        self.is_recording = False
        duration_sec = time.time() - self.record_start_time if self.record_start_time > 0 else 0.0
        saved_path = self.record_filepath
        frames_saved = self.record_frame_count

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        self.btn_record_video.setText("🎥 Bật Ghi Video")
        self.btn_record_video.setStyleSheet("")
        self.lbl_record_badge.setText("")

        if saved_path and os.path.exists(saved_path) and frames_saved > 0:
            fname = os.path.basename(saved_path)
            self.status_bar.showMessage(f"✔ Đã lưu video: {fname} ({frames_saved} frames, {duration_sec:.1f}s)")
            self.lbl_record_badge.setText(f"✔ Đã lưu: {fname}")
            self.lbl_record_badge.setStyleSheet("color: #7aa2f7; font-weight: bold; padding: 0 4px;")
            QTimer.singleShot(4000, lambda: self.lbl_record_badge.setText("") if not self.is_recording else None)
        else:
            self.status_bar.showMessage("Đã dừng ghi video.")

    def _open_recordings_folder(self):
        rec_dir = os.path.abspath(os.path.join(CURRENT_DIR, "output_qr", "recordings"))
        os.makedirs(rec_dir, exist_ok=True)
        try:
            import subprocess
            subprocess.Popen(["xdg-open", rec_dir])
        except Exception:
            QMessageBox.information(self, "Thư Mục Video", f"Video được lưu tại:\n{rec_dir}")

    def _add_to_history(self, text: str, vid_time_sec: Optional[float], size_str: str, line_offset: Optional[float], backend: str, is_qr_mode: bool = False):
        row_pos = self.table_history.rowCount()
        self.table_history.insertRow(row_pos)

        time_str = f"{vid_time_sec:.2f}s" if vid_time_sec is not None else time.strftime("%H:%M:%S")
        if line_offset is not None:
            tag = " (QR)" if is_qr_mode else " (Line)"
            offset_str = f"{line_offset:+.1f} px{tag}"
        else:
            offset_str = "N/A"

        item_time = QTableWidgetItem(time_str)
        item_text = QTableWidgetItem(text)
        item_size = QTableWidgetItem(size_str)
        item_offset = QTableWidgetItem(offset_str)
        item_backend = QTableWidgetItem(backend)

        frame_target = int(vid_time_sec * self.fps) if vid_time_sec is not None else 0
        item_time.setData(Qt.UserRole, frame_target)

        self.table_history.setItem(row_pos, 0, item_time)
        self.table_history.setItem(row_pos, 1, item_text)
        self.table_history.setItem(row_pos, 2, item_size)
        self.table_history.setItem(row_pos, 3, item_offset)
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
                self.status_bar.showMessage(f"Đã chuyển đến Frame {target_frame}")

    def _clear_history(self):
        self.table_history.setRowCount(0)

    def _open_jsonl_log(self):
        log_path = os.path.abspath("output_qr/combined_mission_events.jsonl")
        if os.path.exists(log_path):
            QMessageBox.information(self, "Nhật Ký", f"File log được lưu tại:\n{log_path}")
        else:
            QMessageBox.information(self, "Thông báo", "Chưa có sự kiện nào được ghi nhận.")

    def _toggle_quick_uart(self):
        new_state = not getattr(self.uart, "is_enabled", False)
        self._set_uart_enabled(new_state)

    def _on_chk_uart_toggled(self, checked: bool):
        self._set_uart_enabled(checked)

    def _set_uart_enabled(self, enabled: bool):
        if not hasattr(self, "uart") or not self.uart:
            return

        self.uart.is_enabled = enabled
        if enabled:
            ok = self.uart.connect()
        else:
            self.uart.disconnect()
            ok = False

        self._update_uart_ui_state()
        status_msg = f"Truyền UART STM32: {'BẬT (Gửi 5 byte mỗi 0.1s)' if enabled else 'TẮT'}"
        self.statusBar().showMessage(status_msg, 3000)

    def _update_uart_ui_state(self):
        if not hasattr(self, "uart") or not self.uart:
            return

        is_on = getattr(self.uart, "is_enabled", False)
        is_conn = getattr(self.uart, "is_connected", False)

        # 1. Đồng bộ Checkbox ở Tab Dẫn hướng bên phải
        if hasattr(self, "chk_uart_enable"):
            self.chk_uart_enable.blockSignals(True)
            self.chk_uart_enable.setChecked(is_on)
            self.chk_uart_enable.blockSignals(False)

        # 2. Cập nhật nút nhấn nhanh nổi bật ở thanh điều khiển (tiết kiệm không gian)
        if hasattr(self, "btn_quick_uart"):
            self.btn_quick_uart.blockSignals(True)
            self.btn_quick_uart.setChecked(is_on)
            if is_on and is_conn:
                self.btn_quick_uart.setText(f"🟢 GỬI UART ({self.uart.port}) - BẬT")
                self.btn_quick_uart.setStyleSheet(
                    "background-color: #2e7d32; color: #ffffff; font-weight: bold; font-size: 11px; "
                    "padding: 4px 8px; border-radius: 4px; border: 1px solid #81c784;"
                )
            elif is_on and not is_conn:
                self.btn_quick_uart.setText(f"🔴 LỖI UART ({self.uart.port})")
                self.btn_quick_uart.setStyleSheet(
                    "background-color: #b71c1c; color: #ffffff; font-weight: bold; font-size: 11px; "
                    "padding: 4px 8px; border-radius: 4px; border: 1px solid #ef5350;"
                )
            else:
                self.btn_quick_uart.setText("📡 BẬT UART (STM32)")
                self.btn_quick_uart.setStyleSheet(
                    "background-color: #37474f; color: #ffb74d; font-weight: bold; font-size: 11px; "
                    "padding: 4px 8px; border-radius: 4px; border: 1px solid #ff9800;"
                )
            self.btn_quick_uart.blockSignals(False)

        # 3. Cập nhật Label trạng thái ở Panel Phải
        if hasattr(self, "lbl_uart_info"):
            if is_on and is_conn:
                self.lbl_uart_info.setText(f"🟢 {self.uart.port} (115200)")
                self.lbl_uart_info.setStyleSheet("color: #00E676; font-weight: bold;")
            elif is_on and not is_conn:
                self.lbl_uart_info.setText(f"🔴 Lỗi: {self.uart.port}")
                self.lbl_uart_info.setStyleSheet("color: #ff5555; font-weight: bold;")
            else:
                self.lbl_uart_info.setText("UART: Tắt")
                self.lbl_uart_info.setStyleSheet("color: #565f89;")

    def closeEvent(self, event):
        self.timer.stop()
        if self.is_recording:
            self._stop_recording()
        if hasattr(self, "uart") and self.uart:
            self.uart.disconnect()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.logger.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Robot Line + QR Navigation")

    initial_vid = None
    initial_uart = False
    for arg in sys.argv[1:]:
        if arg == "--uart":
            initial_uart = True
        elif not arg.startswith("-") and initial_vid is None:
            initial_vid = arg

    gui = CombinedRobotApp(initial_video=initial_vid, initial_uart=initial_uart)
    # Tự động mở cửa sổ vừa khít toàn màn hình (Fit to screen)
    gui.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
