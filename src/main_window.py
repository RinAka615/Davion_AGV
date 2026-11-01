"""
Desktop Graphical User Interface for Agricultural Greenhouse Path Detection.
Supports DUAL MODES:
  1. AI Mode: YOLOv8/YOLO26 segmentation (bat_loi_di class)
  2. Image Processing Mode: CIELAB color distribution model & interactive sampling
Features 4-view layout, boundary clipping detection, robust centerline fitting,
and background non-blocking video processing worker thread.
"""

from dataclasses import dataclass
import json
import os
import threading
import time
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from .ui_compat import (
    QtWidgets, QtCore, QtGui, Qt, QThread, Signal, Slot, QTimer,
    QRect, QPoint, QSize, QImage, QPixmap, QPainter, QPen, QColor, QFont,
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QLabel, QPushButton, QSlider, QCheckBox, QComboBox, QFileDialog, QMessageBox,
    QGroupBox, QFrame, QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
    QLineEdit, QStatusBar, QT_BINDING
)

from .video_reader import VideoReader
from .yolo_segmenter import (
    YoloSegmenter, YoloResult, InstanceDetection, is_cuda_available_and_working,
    get_cuda_status, CudaDeviceInfo, get_cpu_info
)
from .color_model import ColorModel, ColorPatch
from .path_detector import PathDetector, DetectionResult, TrackingState, ScanlineResult
from .profile_manager import ProfileManager, PROJECT_ROOT
from .uart_controller import UartController


def draw_dashed_line(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                     color: Tuple[int, int, int], thickness: int = 2,
                     dash_len: int = 8, gap_len: int = 6):
    """Draws a dashed line segment between pt1 and pt2 using OpenCV."""
    p1 = np.array(pt1, dtype=float)
    p2 = np.array(pt2, dtype=float)
    dist = np.linalg.norm(p2 - p1)
    if dist <= 0:
        return
    vec = (p2 - p1) / dist
    curr = 0.0
    drawing = True
    while curr < dist:
        seg = dash_len if drawing else gap_len
        nxt = min(dist, curr + seg)
        if drawing:
            s_pt = tuple(np.round(p1 + vec * curr).astype(int))
            e_pt = tuple(np.round(p1 + vec * nxt).astype(int))
            cv2.line(img, s_pt, e_pt, color, thickness, cv2.LINE_AA)
        curr = nxt
        drawing = not drawing


def draw_dashed_polyline(img: np.ndarray, pts: List[Tuple[int, int]],
                          color: Tuple[int, int, int], thickness: int = 2,
                          dash_len: int = 8, gap_len: int = 6):
    """Draws a dashed polyline through a sequence of points."""
    if len(pts) < 2:
        return
    for i in range(len(pts) - 1):
        draw_dashed_line(img, pts[i], pts[i + 1], color, thickness, dash_len, gap_len)


class InteractiveImageLabel(QLabel):
    """Custom QLabel supporting interactive rectangle drag selection with aspect-ratio mapping."""

    rectSelected = Signal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333;")
        self.setMinimumSize(320, 240)

        self.current_pixmap: Optional[QPixmap] = None
        self.original_image_size: QSize = QSize(0, 0)

        self.is_dragging: bool = False
        self.drag_start: QPoint = QPoint()
        self.drag_current: QPoint = QPoint()

    def set_display_pixmap(self, pixmap: QPixmap, orig_size: Optional[Tuple[int, int]] = None):
        """Sets pixmap and records underlying image dimensions."""
        self.current_pixmap = pixmap
        if orig_size:
            self.original_image_size = QSize(orig_size[0], orig_size[1])
        elif not pixmap.isNull():
            self.original_image_size = pixmap.size()
        self.update()

    def get_pixmap_render_rect(self) -> QRect:
        """Calculates geometry of the scaled pixmap within the label widget."""
        if not self.current_pixmap or self.current_pixmap.isNull():
            return QRect()

        scaled = self.current_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        return QRect(x, y, scaled.width(), scaled.height())

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == Qt.LeftButton and self.current_pixmap and not self.current_pixmap.isNull():
            render_rect = self.get_pixmap_render_rect()
            if render_rect.contains(event.pos()):
                self.is_dragging = True
                self.drag_start = event.pos()
                self.drag_current = event.pos()
                self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self.is_dragging:
            self.drag_current = event.pos()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == Qt.LeftButton and self.is_dragging:
            self.is_dragging = False
            render_rect = self.get_pixmap_render_rect()
            if render_rect.width() > 0 and render_rect.height() > 0 and self.original_image_size.width() > 0:
                x1 = max(render_rect.left(), min(self.drag_start.x(), self.drag_current.x()))
                y1 = max(render_rect.top(), min(self.drag_start.y(), self.drag_current.y()))
                x2 = min(render_rect.right(), max(self.drag_start.x(), self.drag_current.x()))
                y2 = min(render_rect.bottom(), max(self.drag_start.y(), self.drag_current.y()))

                sel_w = x2 - x1
                sel_h = y2 - y1

                if sel_w > 5 and sel_h > 5:
                    scale_x = self.original_image_size.width() / float(render_rect.width())
                    scale_y = self.original_image_size.height() / float(render_rect.height())

                    img_x = int((x1 - render_rect.left()) * scale_x)
                    img_y = int((y1 - render_rect.top()) * scale_y)
                    img_w = int(sel_w * scale_x)
                    img_h = int(sel_h * scale_y)

                    self.rectSelected.emit(QRect(img_x, img_y, img_w, img_h))
            self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event: QtGui.QPaintEvent):
        super().paintEvent(event)
        if not self.current_pixmap or self.current_pixmap.isNull():
            painter = QPainter(self)
            painter.setPen(QColor("#777"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Chưa có hình ảnh\nVui lòng mở video để bắt đầu")
            return

        render_rect = self.get_pixmap_render_rect()
        painter = QPainter(self)
        scaled_pixmap = self.current_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        painter.drawPixmap(render_rect.topLeft(), scaled_pixmap)

        if self.is_dragging:
            sel_rect = QRect(self.drag_start, self.drag_current).normalized()
            painter.setPen(QPen(QColor(0, 255, 255), 2, Qt.DashLine))
            painter.setBrush(QtGui.QBrush(QColor(0, 255, 255, 40)))
            painter.drawRect(sel_rect)


@dataclass
class ProcessedPacket:
    """Atomic bundle containing video frame, detection results, and timing telemetry."""
    frame: np.ndarray
    yolo_res: Optional[YoloResult]
    det_res: DetectionResult
    score_map: Optional[np.ndarray]
    frame_idx: int
    timestamp: float
    session_id: int
    read_time_ms: float = 0.0
    yolo_time_ms: float = 0.0
    detect_time_ms: float = 0.0
    pipeline_time_ms: float = 0.0
    dropped_frames_total: int = 0


class ProcessingWorker(QThread):
    """Background worker thread supporting both AI (YOLO) and Image Processing (Color) modes.
    Features atomic single-slot flow control to prevent UI freezing, latency drift, and memory backlog.
    """

    frameProcessed = Signal(np.ndarray, object, object, object, int, float, int)
    packetReady = Signal(int)
    finishedPlayback = Signal()

    def __init__(
        self,
        reader: VideoReader,
        segmenter: YoloSegmenter,
        detector: PathDetector,
        color_model: ColorModel,
        mode: str = "ai",
        conf: float = 0.25,
        imgsz: int = 416,
        session_id: int = 0,
        uart: Optional[UartController] = None
    ):
        super().__init__()
        self.reader = reader
        self.segmenter = segmenter
        self.detector = detector
        self.color_model = color_model
        self.uart = uart
        self.mode = mode  # "ai" or "color"
        self.conf = conf
        self.imgsz = imgsz
        self.session_id = session_id
        self.is_running: bool = False
        self.speed_multiplier: float = 1.0

        # Flow control & atomic packet buffer
        self.packet_lock = threading.Lock()
        self.latest_packet: Optional[ProcessedPacket] = None
        self.dropped_ui_frames: int = 0
        self.dropped_read_frames: int = 0

    def get_latest_packet(self) -> Optional[ProcessedPacket]:
        """Atomically retrieves and clears the latest packet waiting for the UI."""
        with self.packet_lock:
            pkt = self.latest_packet
            self.latest_packet = None
            return pkt

    def run(self):
        self.is_running = True
        target_fps = max(1.0, self.reader.fps)
        wall_start_time = time.perf_counter()
        playback_start_frame = self.reader.current_frame_idx

        while self.is_running:
            # 1. Timeline synchronization (only for video files, live cameras pace themselves)
            if not getattr(self.reader, "is_live", False):
                now = time.perf_counter()
                elapsed_wall = now - wall_start_time
                expected_frame_idx = playback_start_frame + int(elapsed_wall * target_fps * self.speed_multiplier)
                lag = expected_frame_idx - self.reader.current_frame_idx

                if lag > 1:
                    # Fast forward without decoding to keep playback in sync
                    skipped = self.reader.skip_frames(min(lag - 1, 10))
                    self.dropped_read_frames += skipped

            # 2. Read frame
            t_read_start = time.perf_counter()
            ret, frame, frame_idx, timestamp = self.reader.read_frame()
            t_read_ms = (time.perf_counter() - t_read_start) * 1000.0

            if not ret or frame is None:
                self.is_running = False
                self.finishedPlayback.emit()
                break

            # 3. Process frame
            t_pipe_start = time.perf_counter()
            roi_rect = self.detector.compute_roi(frame.shape[:2])
            yolo_ms = 0.0

            if self.mode == "ai":
                yolo_res = self.segmenter.predict(
                    frame_bgr=frame,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    roi_rect=roi_rect
                )
                yolo_ms = yolo_res.inference_time_ms if yolo_res else 0.0
                t_det_start = time.perf_counter()
                det_res = self.detector.process(
                    frame_bgr=frame,
                    mask_or_score=yolo_res.chosen_mask,
                    is_source_valid=yolo_res.is_valid
                )
                detect_ms = (time.perf_counter() - t_det_start) * 1000.0
                score_map = None
            else:
                yolo_res = None
                if self.color_model.is_trained:
                    score_map = self.color_model.score_frame(frame, roi_rect)
                    is_valid = True
                else:
                    rx, ry, rw, rh = roi_rect
                    score_map = np.zeros((rh, rw), dtype=np.uint8)
                    is_valid = False

                t_det_start = time.perf_counter()
                det_res = self.detector.process(
                    frame_bgr=frame,
                    mask_or_score=score_map,
                    is_source_valid=is_valid
                )
                detect_ms = (time.perf_counter() - t_det_start) * 1000.0

                if (self.color_model.online_update_enabled and
                    det_res.state == TrackingState.TRACKING and
                    det_res.confidence >= 70.0 and
                    det_res.safe_interior_mask is not None):
                    self.color_model.update_online(frame, det_res.safe_interior_mask, (roi_rect[0], roi_rect[1]))

            pipeline_ms = (time.perf_counter() - t_pipe_start) * 1000.0

            # 4. UART transmission to STM32 (Non-blocking, rate-limited at 10 Hz / 0.1s)
            if self.uart and self.uart.is_enabled:
                state_code = UartController.STATE_LOST
                if det_res.state == TrackingState.TRACKING:
                    state_code = UartController.STATE_TRACKING
                elif det_res.state == TrackingState.LOW_CONFIDENCE:
                    state_code = UartController.STATE_LOW_CONF
                elif det_res.state == TrackingState.NO_MODEL:
                    state_code = UartController.STATE_NO_MODEL

                target_pt = det_res.path_center if (det_res.center_method == "path" and det_res.path_center is not None) else det_res.box_center
                tx = int(target_pt[0]) if target_pt else None
                ty = int(target_pt[1]) if target_pt else None

                is_detected = bool(
                    (det_res.state == TrackingState.TRACKING or 
                     (det_res.state == TrackingState.LOW_CONFIDENCE and det_res.lateral_offset_px is not None))
                )

                self.uart.send_nav_update(
                    state_code=state_code,
                    offset_px=det_res.lateral_offset_px,
                    confidence_pct=det_res.confidence,
                    target_x=tx,
                    target_y=ty,
                    is_detected=is_detected
                )

            # 5. Atomic packet delivery (single-slot buffer, replaces unread packet)
            total_dropped = self.dropped_read_frames + self.dropped_ui_frames
            pkt = ProcessedPacket(
                frame=frame,
                yolo_res=yolo_res,
                det_res=det_res,
                score_map=score_map,
                frame_idx=frame_idx,
                timestamp=timestamp,
                session_id=self.session_id,
                read_time_ms=t_read_ms,
                yolo_time_ms=yolo_ms,
                detect_time_ms=detect_ms,
                pipeline_time_ms=pipeline_ms,
                dropped_frames_total=total_dropped
            )

            with self.packet_lock:
                if self.latest_packet is not None:
                    self.dropped_ui_frames += 1
                self.latest_packet = pkt

            self.packetReady.emit(self.session_id)
            self.frameProcessed.emit(frame, yolo_res, det_res, score_map, frame_idx, timestamp, self.session_id)

            # 5. Pacing sleep if faster than video FPS
            target_period = 1.0 / (target_fps * self.speed_multiplier)
            cycle_time = time.perf_counter() - t_read_start
            sleep_time = target_period - cycle_time
            if sleep_time > 0.002:
                time.sleep(sleep_time)
            else:
                # Trên CPU/ARM: Thêm khoảng nghỉ ngắn 10ms để CPU không bị ghim 100% gây sụt áp nguồn (Brownout)
                time.sleep(0.01)

    def stop(self):
        self.is_running = False
        self.wait(1500)


class MainWindow(QMainWindow):
    """Main application window supporting Dual Modes: AI (YOLO) and Image Processing (Color)."""

    MODE_AI = "ai"
    MODE_COLOR = "color"

    def __init__(
        self,
        initial_video_path: Optional[str] = None,
        initial_model_path: Optional[str] = None,
        initial_mode: str = "ai",
        initial_center_method: str = "box",
        initial_device: Optional[str] = None,
        initial_uart: bool = False,
        initial_uart_port: str = "/dev/ttySTM32",
        initial_uart_baud: int = 115200,
        initial_uart_protocol: str = UartController.PROTOCOL_UNION_5B,
        initial_uart_union_type: str = UartController.UNION_TYPE_FLOAT
    ):
        super().__init__()
        self.setWindowTitle(f"RobotNongNghiep - Nhận diện lối đi nhà màng [Dual-Engine: AI & Xử lý ảnh] [{QT_BINDING}]")
        self.resize(1420, 930)

        # Core logic modules
        self.reader = VideoReader()
        self.detector = PathDetector()
        self.detector.center_method = initial_center_method
        self.color_model = ColorModel()

        # UART Controller for STM32 (10 Hz = 0.1s rate limit)
        self.uart = UartController(
            port=initial_uart_port,
            baudrate=initial_uart_baud,
            send_rate_hz=10.0,
            protocol=initial_uart_protocol,
            union_type=initial_uart_union_type
        )
        if initial_uart:
            self.uart.is_enabled = True
            self.uart.connect()

        # Detection Mode
        self.active_mode: str = self.MODE_AI if initial_mode.lower() in ("ai", "yolo") else self.MODE_COLOR

        # Device & YOLO Segmenter
        self._explicit_initial_model = initial_model_path if (initial_model_path and os.path.exists(initial_model_path)) else None
        pico_ncnn = os.path.join(PROJECT_ROOT, "models", "nhamang_pico_seg_ncnn_model")
        nano_ncnn = os.path.join(PROJECT_ROOT, "models", "nhamang_batden_seg_ncnn_model")
        nhamang_model = os.path.join(PROJECT_ROOT, "models", "nhamang_batden_seg.pt")
        if os.path.exists(pico_ncnn):
            fallback_model = pico_ncnn
        elif os.path.exists(nano_ncnn):
            fallback_model = nano_ncnn
        elif os.path.exists(nhamang_model):
            fallback_model = nhamang_model
        else:
            fallback_model = os.path.join(PROJECT_ROOT, "models", "farm_path_bootstrap.pt")
        default_model = self._explicit_initial_model or fallback_model
        self.yolo_conf: float = 0.25
        self.yolo_imgsz: int = 320 if ("ncnn" in default_model.lower() or "pico" in default_model.lower()) else 416
        self.cuda_status = get_cuda_status()
        if initial_device:
            self.yolo_device = initial_device
        else:
            self.yolo_device = "cuda" if self.cuda_status.is_available else "cpu"

        self.segmenter = YoloSegmenter(
            model_path=default_model if os.path.exists(default_model) else None,
            conf_threshold=self.yolo_conf,
            imgsz=self.yolo_imgsz,
            device=self.yolo_device
        )
        if self.segmenter.fixed_imgsz is not None:
            self.yolo_imgsz = self.segmenter.fixed_imgsz

        # Playback & state
        self.worker: Optional[ProcessingWorker] = None
        self.is_paused: bool = True
        self.current_session_id: int = 0
        self.current_frame_bgr: Optional[np.ndarray] = None
        self.last_yolo_result: Optional[YoloResult] = None
        self.last_detection_result: Optional[DetectionResult] = None
        self.last_score_map: Optional[np.ndarray] = None
        self.active_sample_category: str = "path"

        # Telemetry & Pacing
        self.display_fps: float = 0.0
        self.last_render_time: float = 0.0
        self._last_v4_render_time: float = 0.0
        self._is_rendering_frame: bool = False

        # Debounce timer for slider adjustments when paused
        self.yolo_debounce_timer = QTimer(self)
        self.yolo_debounce_timer.setSingleShot(True)
        self.yolo_debounce_timer.timeout.connect(self._on_yolo_debounce_timeout)

        # Build UI layout
        self._init_ui()

        # Update labels and view titles according to active mode
        self._apply_mode_ui_state()
        self._update_uart_buttons_ui()

        # Try to load default profile if exists
        default_prof_path = os.path.join(ProfileManager.DEFAULT_PROFILE_DIR, "default_profile.json")
        if os.path.exists(default_prof_path):
            self._load_profile_path(default_prof_path)

        # Open initial video if provided
        if initial_video_path and os.path.exists(initial_video_path):
            self.open_video(initial_video_path)

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Left: 4 Display Views (Grid 2x2)
        views_widget = QWidget()
        views_layout = QGridLayout(views_widget)
        views_layout.setContentsMargins(0, 0, 0, 0)
        views_layout.setSpacing(6)

        # View 1: Original + ROI
        self.view_original = InteractiveImageLabel()
        self.view_original.rectSelected.connect(self._on_sample_rect_selected)
        self.v1_box = self._create_view_container("1. Khung hình gốc + Vùng ROI", self.view_original)
        views_layout.addWidget(self.v1_box, 0, 0)

        # View 2: Detection Overlay
        self.view_overlay = InteractiveImageLabel()
        self.v2_box = self._create_view_container("2. Kết quả nhận diện (Overlay, Bounding Box & Trục tâm)", self.view_overlay)
        views_layout.addWidget(self.v2_box, 0, 1)

        # View 3: Binary Mask
        self.view_mask = InteractiveImageLabel()
        self.v3_box = self._create_view_container("3. Mặt nạ nhị phân lối đi (Binary Mask)", self.view_mask)
        views_layout.addWidget(self.v3_box, 1, 0)

        # View 4: Mode-dependent View (AI Debug / Color Heatmap)
        self.view_debug = InteractiveImageLabel()
        v4_container = QWidget()
        v4_layout = QVBoxLayout(v4_container)
        v4_layout.setContentsMargins(0, 0, 0, 0)
        v4_layout.setSpacing(4)
        self.chk_view4_enable = QCheckBox("Bật View 4 (Debug/Heatmap) [Giới hạn 15 FPS]")
        self.chk_view4_enable.setChecked(True)
        self.chk_view4_enable.setStyleSheet("color: #8cf; font-size: 11px;")
        v4_layout.addWidget(self.chk_view4_enable)
        v4_layout.addWidget(self.view_debug)
        self.v4_box = self._create_view_container("4. Debug / Heatmap", v4_container)
        views_layout.addWidget(self.v4_box, 1, 1)

        # Alias for backward compatibility
        self.view_heatmap = self.view_debug

        # Right: Control Panel (Scrollable)
        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setMinimumWidth(440)
        control_scroll.setMaximumWidth(480)

        control_content = QWidget()
        self.control_layout = QVBoxLayout(control_content)
        self.control_layout.setContentsMargins(6, 6, 6, 6)
        self.control_layout.setSpacing(8)

        # Sections
        self._init_mode_selector_group()
        self._init_playback_group()
        self._init_yolo_group()
        self._init_sampling_group()
        self._init_parameters_group()
        self._init_status_group()
        self._init_uart_group()
        self._init_profile_group()

        self.control_layout.addStretch()
        control_scroll.setWidget(control_content)

        # Splitter between Views and Controls
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(views_widget)
        splitter.addWidget(control_scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Sẵn sàng. Chọn chế độ AI hoặc Xử lý ảnh để bắt đầu.")

    def _create_view_container(self, title: str, label_widget: QWidget) -> QGroupBox:
        box = QGroupBox(title)
        box.setStyleSheet("QGroupBox { font-weight: bold; color: #ddd; border: 1px solid #444; margin-top: 6px; padding-top: 10px; }")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(label_widget)
        return box

    def _init_mode_selector_group(self):
        box = QGroupBox("⚙️ Lựa chọn Chế độ Nhận diện")
        box.setStyleSheet("QGroupBox { font-weight: bold; color: #8cf; border: 1px solid #0088cc; padding-top: 10px; }")
        layout = QVBoxLayout(box)

        self.combo_mode = QComboBox()
        self.combo_mode.setStyleSheet("font-size: 13px; font-weight: bold; padding: 6px;")
        self.combo_mode.addItems([
            "🤖 Chế độ AI (YOLO Segmentation)",
            "🎨 Chế độ Xử lý ảnh (CIELAB Color Model)"
        ])
        if self.active_mode == self.MODE_COLOR:
            self.combo_mode.setCurrentIndex(1)
        else:
            self.combo_mode.setCurrentIndex(0)
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        layout.addWidget(self.combo_mode)

        self.control_layout.addWidget(box)

    def _on_mode_changed(self, index: int):
        if index == 0:
            self.active_mode = self.MODE_AI
        else:
            self.active_mode = self.MODE_COLOR

        if self.worker:
            self.worker.mode = self.active_mode

        self.detector.reset_tracking()
        self._apply_mode_ui_state()
        self._reprocess_current_frame()
        self.status_bar.showMessage(f"Đã chuyển sang: {self.combo_mode.currentText()}")

    def _apply_mode_ui_state(self):
        is_ai = (self.active_mode == self.MODE_AI)

        # Toggle parameter sections visibility
        self.group_yolo.setVisible(is_ai)
        self.group_color.setVisible(not is_ai)

        # Threshold slider only visible in Color mode
        self.row_threshold_widget.setVisible(not is_ai)

        # Update view titles
        if is_ai:
            self.v2_box.setTitle("2. Kết quả nhận diện (Overlay YOLO, Bounding Box & Trục tâm)")
            self.v4_box.setTitle("4. Debug (Tất cả Instance & Phân tích Lát cắt)")
            self._update_model_status_label()
        else:
            self.v2_box.setTitle("2. Kết quả nhận diện (Overlay Xử lý ảnh, Bounding Box & Trục tâm)")
            self.v4_box.setTitle("4. Bản đồ điểm tương đồng màu (CIELAB Heatmap)")

    def _init_playback_group(self):
        box = QGroupBox("Điều khiển Video")
        layout = QVBoxLayout(box)

        # File & Camera open buttons
        top_row = QHBoxLayout()
        self.btn_open = QPushButton("📂 Mở Video")
        self.btn_open.clicked.connect(self._on_open_video_clicked)
        self.btn_open_cam = QPushButton("📷 Mở Camera Orbbec")
        self.btn_open_cam.setStyleSheet("font-weight: bold; color: #81c784;")
        self.btn_open_cam.clicked.connect(self._on_open_camera_clicked)
        self.lbl_file_info = QLabel("Chưa mở file")
        self.lbl_file_info.setStyleSheet("color: #aaa;")
        top_row.addWidget(self.btn_open)
        top_row.addWidget(self.btn_open_cam)
        top_row.addWidget(self.lbl_file_info, 1)
        layout.addLayout(top_row)

        # Row 1b: Big prominent button to Toggle UART transmission to STM32
        uart_quick_row = QHBoxLayout()
        self.btn_quick_uart = QPushButton("📡 BẬT GỬI LỆNH UART (STM32)")
        self.btn_quick_uart.setCheckable(True)
        self.btn_quick_uart.setChecked(self.uart.is_enabled)
        self.btn_quick_uart.setToolTip("Bấm để BẬT hoặc TẮT truyền tín hiệu điều khiển 5 byte (mỗi 0.1s) xuống STM32")
        self.btn_quick_uart.clicked.connect(self._toggle_uart_transmission)
        uart_quick_row.addWidget(self.btn_quick_uart)
        layout.addLayout(uart_quick_row)

        # Play / Step buttons
        btn_row = QHBoxLayout()
        self.btn_play = QPushButton("▶ Phát (Play)")
        self.btn_play.clicked.connect(self._toggle_play_pause)
        self.btn_step_prev = QPushButton("◀ Lùi 1F")
        self.btn_step_prev.clicked.connect(lambda: self._step_frame(-1))
        self.btn_step_next = QPushButton("Tiến 1F ▶")
        self.btn_step_next.clicked.connect(lambda: self._step_frame(1))

        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["0.25x", "0.5x", "1.0x", "2.0x"])
        self.combo_speed.setCurrentText("1.0x")
        self.combo_speed.currentTextChanged.connect(self._on_speed_changed)

        btn_row.addWidget(self.btn_play)
        btn_row.addWidget(self.btn_step_prev)
        btn_row.addWidget(self.btn_step_next)
        btn_row.addWidget(QLabel("Tốc độ:"))
        btn_row.addWidget(self.combo_speed)
        layout.addLayout(btn_row)

        # Seek slider
        slider_row = QHBoxLayout()
        self.slider_seek = QSlider(Qt.Horizontal)
        self.slider_seek.setRange(0, 0)
        self.slider_seek.sliderMoved.connect(self._on_seek_slider_moved)
        self.lbl_time = QLabel("00:00 / 00:00 (F: 0/0)")
        self.lbl_time.setStyleSheet("font-family: monospace; color: #8cf;")
        slider_row.addWidget(self.slider_seek)
        slider_row.addWidget(self.lbl_time)
        layout.addLayout(slider_row)

        self.control_layout.addWidget(box)

    def _init_yolo_group(self):
        self.group_yolo = QGroupBox("🤖 Mô hình AI / NCNN Quantized")
        layout = QVBoxLayout(self.group_yolo)

        # Quick Model Preset Dropdown + File Browser
        row_preset = QHBoxLayout()
        row_preset.addWidget(QLabel("Chọn nhanh:"))
        self.combo_model_preset = QComboBox()
        self._populate_model_presets()
        self.combo_model_preset.currentIndexChanged.connect(self._on_model_preset_changed)
        row_preset.addWidget(self.combo_model_preset, 1)
        layout.addLayout(row_preset)

        # Model file selector row
        row_model = QHBoxLayout()
        self.btn_select_model = QPushButton("📂 Chọn File (.pt / NCNN)")
        self.btn_select_model.clicked.connect(self._on_select_model_clicked)
        self.lbl_model_name = QLabel(os.path.basename(self.segmenter.model_path or "Chưa chọn"))
        self.lbl_model_name.setStyleSheet("color: #81c784; font-weight: bold;")
        row_model.addWidget(self.btn_select_model)
        row_model.addWidget(self.lbl_model_name, 1)
        layout.addLayout(row_model)

        # Model Status Badge
        self.lbl_model_status = QLabel("Trạng thái model...")
        self.lbl_model_status.setStyleSheet("color: #bbb; font-size: 11px;")
        layout.addWidget(self.lbl_model_status)

        # Confidence slider
        self.slider_yolo_conf, self.lbl_yolo_conf_val = self._add_slider_row(
            layout, "Ngưỡng tự tin (Conf):", 5, 95, int(self.yolo_conf * 100), "%"
        )
        self.slider_yolo_conf.valueChanged.connect(self._on_yolo_conf_changed)

        # Image size and device row
        row_cfg = QHBoxLayout()
        row_cfg.addWidget(QLabel("Img Size:"))
        self.combo_imgsz = QComboBox()
        self.combo_imgsz.addItems(["320", "416", "512", "640"])
        self.combo_imgsz.setCurrentText(str(self.yolo_imgsz))
        self.combo_imgsz.currentTextChanged.connect(self._on_yolo_imgsz_changed)
        row_cfg.addWidget(self.combo_imgsz)

        row_cfg.addWidget(QLabel("Thiết bị:"))
        self.combo_device = QComboBox()
        self.combo_device.addItem("CPU", "cpu")
        self.combo_device.addItem("Auto (Ưu tiên CUDA)", "auto")
        if self.cuda_status.is_available:
            self.combo_device.addItem(f"CUDA: {self.cuda_status.device_name}", "cuda")
        else:
            self.combo_device.addItem("CUDA (Lỗi: EIO / Driver)", "cuda")
        default_dev_idx = self.combo_device.findData(self.yolo_device)
        self.combo_device.setCurrentIndex(default_dev_idx if default_dev_idx >= 0 else 0)
        self.combo_device.currentIndexChanged.connect(self._on_yolo_device_changed)
        row_cfg.addWidget(self.combo_device)

        self.btn_reload_model = QPushButton("🔄 Nạp lại")
        self.btn_reload_model.clicked.connect(self._reload_model)
        row_cfg.addWidget(self.btn_reload_model)

        layout.addLayout(row_cfg)

        # Real Device Live Status Badge
        self.lbl_device_real = QLabel(f"Thiết bị thực tế: {self.segmenter.device_display_name}")
        if self.segmenter.fallback_reason:
            self.lbl_device_real.setStyleSheet("color: #ffb74d; font-size: 11px;")
            self.lbl_device_real.setToolTip(self.segmenter.fallback_reason)
        else:
            self.lbl_device_real.setStyleSheet("color: #81c784; font-size: 11px;")
        layout.addWidget(self.lbl_device_real)

        self.control_layout.addWidget(self.group_yolo)

    def _init_sampling_group(self):
        self.group_color = QGroupBox("🎨 Học màu (Lấy mẫu & Phân bố CIELAB)")
        layout = QVBoxLayout(self.group_color)

        lbl_target = QLabel("🎯 Mục tiêu: Bạt / Lối đi (PATH) trong khung ROI")
        lbl_target.setStyleSheet("background-color: #1b5e20; color: #ffffff; padding: 5px 8px; border-radius: 4px; font-weight: bold; font-size: 11px;")
        layout.addWidget(lbl_target)

        lbl_hint = QLabel("💡 Hướng dẫn: Kéo chuột chọn mẫu trên ảnh số 1 hoặc bấm 'Lấy mẫu từ ROI'.")
        lbl_hint.setWordWrap(True)
        lbl_hint.setStyleSheet("color: #bbb; font-style: italic; font-size: 11px;")
        layout.addWidget(lbl_hint)

        # Patches list table
        self.table_patches = QTableWidget(0, 4)
        self.table_patches.setHorizontalHeaderLabels(["ID", "Mục tiêu", "Tọa độ (X,Y,W,H)", "Thao tác"])
        self.table_patches.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_patches.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table_patches.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table_patches.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table_patches.setMaximumHeight(120)
        layout.addWidget(self.table_patches)

        # Action buttons row
        action_row = QHBoxLayout()
        self.btn_sample_roi = QPushButton(" Lấy mẫu từ ROI")
        self.btn_sample_roi.clicked.connect(self._sample_from_roi)
        self.btn_retrain = QPushButton(" Học lại màu")
        self.btn_retrain.clicked.connect(self._rebuild_color_model)
        self.btn_clear_samples = QPushButton(" Xóa tất cả mẫu")
        self.btn_clear_samples.clicked.connect(self._clear_all_samples)
        action_row.addWidget(self.btn_sample_roi)
        action_row.addWidget(self.btn_retrain)
        action_row.addWidget(self.btn_clear_samples)
        layout.addLayout(action_row)

        # Online update checkbox
        self.chk_online_update = QCheckBox("Cập nhật màu chậm trong lúc chạy (Online)")
        self.chk_online_update.setChecked(False)
        self.chk_online_update.toggled.connect(self._on_online_update_toggled)
        layout.addWidget(self.chk_online_update)

        self.control_layout.addWidget(self.group_color)

    def _init_parameters_group(self):
        box = QGroupBox("Cân chỉnh Tham số Thị giác & ROI")
        layout = QVBoxLayout(box)

        # Centering Method Selector (Bounding Box Midpoint vs Generated Centerline Path)
        row_method = QHBoxLayout()
        row_method.addWidget(QLabel("🎯 Xác định tâm:"))
        self.combo_center_method = QComboBox()
        self.combo_center_method.addItem("📦 Tâm Bounding Box (Bắt vùng detect)", "box")
        self.combo_center_method.addItem("📈 Tâm Đường dẫn (Sinh path centerline)", "path")
        self.combo_center_method.setCurrentIndex(0 if self.detector.center_method == "box" else 1)
        self.combo_center_method.currentIndexChanged.connect(self._on_center_method_changed)
        row_method.addWidget(self.combo_center_method)
        layout.addLayout(row_method)

        # ROI Top
        self.slider_roi_top, self.lbl_roi_top = self._add_slider_row(
            layout, "Vị trí đỉnh ROI (%):", 10, 85, int(self.detector.roi_top_pct * 100), "%"
        )
        self.slider_roi_top.valueChanged.connect(self._on_roi_changed)

        # ROI Height
        self.slider_roi_height, self.lbl_roi_height = self._add_slider_row(
            layout, "Chiều cao ROI (%):", 15, 90, int(self.detector.roi_height_pct * 100), "%"
        )
        self.slider_roi_height.valueChanged.connect(self._on_roi_changed)

        # ROI Width
        self.slider_roi_width, self.lbl_roi_width = self._add_slider_row(
            layout, "Chiều rộng ROI (%):", 25, 100, int(self.detector.roi_width_pct * 100), "%"
        )
        self.slider_roi_width.valueChanged.connect(self._on_roi_changed)

        # ROI Center X
        self.slider_roi_center_x, self.lbl_roi_center_x = self._add_slider_row(
            layout, "Vị trí tâm ngang X (%):", 15, 85, int(self.detector.roi_center_x_pct * 100), "%"
        )
        self.slider_roi_center_x.valueChanged.connect(self._on_roi_changed)

        # Threshold slider (Container widget so it can be shown/hidden depending on mode)
        self.row_threshold_widget = QWidget()
        row_thresh_layout = QHBoxLayout(self.row_threshold_widget)
        row_thresh_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_threshold_title = QLabel("Ngưỡng phân loại bạt:")
        self.lbl_threshold = QLabel(f"{self.detector.threshold}")
        self.lbl_threshold.setStyleSheet("font-weight: bold; color: #8cf; min-width: 45px;")
        self.slider_threshold = QSlider(Qt.Horizontal)
        self.slider_threshold.setRange(10, 245)
        self.slider_threshold.setValue(self.detector.threshold)
        self.slider_threshold.valueChanged.connect(lambda v: self.lbl_threshold.setText(str(v)))
        self.slider_threshold.valueChanged.connect(self._on_detector_param_changed)
        row_thresh_layout.addWidget(self.lbl_threshold_title)
        row_thresh_layout.addWidget(self.slider_threshold, 1)
        row_thresh_layout.addWidget(self.lbl_threshold)
        layout.addWidget(self.row_threshold_widget)

        # Smoothing factor
        self.slider_smooth, self.lbl_smooth = self._add_slider_row(
            layout, "Mức làm mượt đường tâm (%):", 5, 95, int(self.detector.smoothing_factor * 100), "%"
        )
        self.slider_smooth.valueChanged.connect(self._on_detector_param_changed)

        # Scanlines
        self.slider_scanlines, self.lbl_scanlines = self._add_slider_row(
            layout, "Số lát cắt quét (Scanlines):", 8, 48, self.detector.num_scanlines, ""
        )
        self.slider_scanlines.valueChanged.connect(self._on_detector_param_changed)

        # Morph Opening Kernel
        self.slider_open, self.lbl_open = self._add_slider_row(
            layout, "Lọc nhiễu Opening (px):", 1, 21, self.detector.k_open, "px"
        )
        self.slider_open.valueChanged.connect(self._on_detector_param_changed)

        # Morph Closing Kernel
        self.slider_close, self.lbl_close = self._add_slider_row(
            layout, "Lấp khoảng trống Closing (px):", 1, 35, self.detector.k_close, "px"
        )
        self.slider_close.valueChanged.connect(self._on_detector_param_changed)

        self.control_layout.addWidget(box)

    def _add_slider_row(self, parent_layout, label_text: str, min_v: int, max_v: int, cur_v: int, unit: str):
        row = QHBoxLayout()
        lbl_name = QLabel(label_text)
        lbl_val = QLabel(f"{cur_v} {unit}")
        lbl_val.setStyleSheet("font-weight: bold; color: #8cf; min-width: 45px;")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(cur_v)

        def update_val(v):
            lbl_val.setText(f"{v} {unit}")
        slider.valueChanged.connect(update_val)

        row.addWidget(lbl_name)
        row.addWidget(slider, 1)
        row.addWidget(lbl_val)
        parent_layout.addLayout(row)
        return slider, lbl_val

    def _init_status_group(self):
        box = QGroupBox("Trạng thái Nhận diện & Dẫn hướng")
        layout = QVBoxLayout(box)

        # State Badge
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Trạng thái:"))
        self.lbl_status_badge = QLabel("Chưa khởi tạo")
        self.lbl_status_badge.setStyleSheet(
            "background-color: #616161; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;"
        )
        row1.addWidget(self.lbl_status_badge)
        row1.addStretch()
        layout.addLayout(row1)

        # Path Geometry Confidence
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Độ tin cậy hình học lối đi:"))
        self.lbl_confidence = QLabel("0.0%")
        self.lbl_confidence.setStyleSheet("font-size: 13px; font-weight: bold; color: #aaa;")
        row2.addWidget(self.lbl_confidence)
        row2.addStretch()
        layout.addLayout(row2)

        # Mode-specific sub-metric
        row2b = QHBoxLayout()
        self.lbl_sub_metric_title = QLabel("Độ tự tin YOLO instance:")
        self.lbl_sub_metric_val = QLabel("N/A")
        self.lbl_sub_metric_val.setStyleSheet("font-size: 13px; font-weight: bold; color: #81c784;")
        row2b.addWidget(self.lbl_sub_metric_title)
        row2b.addWidget(self.lbl_sub_metric_val)
        row2b.addStretch()
        layout.addLayout(row2b)

        # Lateral Offset
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Độ lệch pixel tại lát cắt tham chiếu:"))
        self.lbl_offset = QLabel("N/A")
        self.lbl_offset.setStyleSheet("font-size: 13px; font-weight: bold; color: #ffb74d;")
        row3.addWidget(self.lbl_offset)
        row3.addStretch()
        layout.addLayout(row3)

        # Reference Scanline Height
        row3b = QHBoxLayout()
        row3b.addWidget(QLabel("Cao độ lát cắt tham chiếu:"))
        self.lbl_ref_slice = QLabel("N/A")
        self.lbl_ref_slice.setStyleSheet("font-size: 12px; color: #80cbc4;")
        row3b.addWidget(self.lbl_ref_slice)
        row3b.addStretch()
        layout.addLayout(row3b)

        lbl_note = QLabel("ℹ️ Lưu ý: Mốc giữa ảnh là trục camera. Độ lệch tính theo pixel trên ảnh, chưa qua hiệu chuẩn camera sang mét.")
        lbl_note.setWordWrap(True)
        lbl_note.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(lbl_note)

        # Latency breakdown row
        row_lat = QHBoxLayout()
        row_lat.addWidget(QLabel("Độ trễ thành phần:"))
        self.lbl_latency_breakdown = QLabel("Read: 0.0ms | YOLO: 0.0ms | Detect: 0.0ms | UI: 0.0ms")
        self.lbl_latency_breakdown.setStyleSheet("color: #b0bec5; font-size: 11px;")
        row_lat.addWidget(self.lbl_latency_breakdown)
        row_lat.addStretch()
        layout.addLayout(row_lat)

        # Speed / FPS
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Tốc độ FPS & Frame:"))
        self.lbl_fps = QLabel("Xử lý: 0 FPS | Hiển thị: 0 FPS (Bỏ qua: 0)")
        self.lbl_fps.setStyleSheet("color: #81c784; font-weight: bold;")
        row4.addWidget(self.lbl_fps)
        row4.addStretch()
        layout.addLayout(row4)

        self.control_layout.addWidget(box)

    def _init_uart_group(self):
        box = QGroupBox("Giao tiếp UART (STM32 Controller)")
        box.setStyleSheet("QGroupBox { font-weight: bold; color: #ffb74d; border: 1px solid #e65100; margin-top: 6px; padding-top: 10px; }")
        layout = QVBoxLayout(box)

        # Row 1: Big toggle button + Enable checkbox
        row1 = QHBoxLayout()
        self.btn_uart_toggle_stream = QPushButton("📡 Bật gửi lệnh UART xuống STM32")
        self.btn_uart_toggle_stream.setCheckable(True)
        self.btn_uart_toggle_stream.setChecked(self.uart.is_enabled)
        self.btn_uart_toggle_stream.clicked.connect(self._toggle_uart_transmission)
        row1.addWidget(self.btn_uart_toggle_stream, 1)

        self.chk_uart_enable = QCheckBox("Kích hoạt")
        self.chk_uart_enable.setStyleSheet("font-weight: bold; color: #fff;")
        self.chk_uart_enable.setChecked(self.uart.is_enabled)
        self.chk_uart_enable.toggled.connect(self._on_uart_enable_toggled)
        row1.addWidget(self.chk_uart_enable)
        layout.addLayout(row1)

        # Row 2: Port & Baud
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Cổng:"))
        self.combo_uart_port = QComboBox()
        self.combo_uart_port.setEditable(True)
        self._refresh_uart_ports()
        row2.addWidget(self.combo_uart_port, 1)

        self.btn_uart_refresh = QPushButton("🔄")
        self.btn_uart_refresh.setToolTip("Quét lại danh sách cổng serial")
        self.btn_uart_refresh.setFixedWidth(36)
        self.btn_uart_refresh.clicked.connect(self._refresh_uart_ports)
        row2.addWidget(self.btn_uart_refresh)

        row2.addWidget(QLabel("Baud:"))
        self.combo_uart_baud = QComboBox()
        self.combo_uart_baud.addItems(["9600", "57600", "115200", "230400", "460800", "921600"])
        self.combo_uart_baud.setCurrentText(str(self.uart.baudrate))
        self.combo_uart_baud.currentTextChanged.connect(self._on_uart_baud_changed)
        row2.addWidget(self.combo_uart_baud)
        layout.addLayout(row2)

        # Row 2b: Protocol & Union Type
        row_proto = QHBoxLayout()
        row_proto.addWidget(QLabel("Định dạng:"))
        self.combo_uart_protocol = QComboBox()
        self.combo_uart_protocol.addItems([
            "📦 Union 5 Byte (4B Lệch + 1B Detect)",
            "📝 NMEA Text ($NAV,...)"
        ])
        if self.uart.protocol == UartController.PROTOCOL_NMEA_TEXT:
            self.combo_uart_protocol.setCurrentIndex(1)
        else:
            self.combo_uart_protocol.setCurrentIndex(0)
        self.combo_uart_protocol.currentIndexChanged.connect(self._on_uart_protocol_changed)
        row_proto.addWidget(self.combo_uart_protocol, 1)

        row_proto.addWidget(QLabel("Kiểu Union:"))
        self.combo_uart_union_type = QComboBox()
        self.combo_uart_union_type.addItems(["Float 32-bit", "Int 32-bit"])
        if self.uart.union_type == UartController.UNION_TYPE_INT32:
            self.combo_uart_union_type.setCurrentIndex(1)
        else:
            self.combo_uart_union_type.setCurrentIndex(0)
        self.combo_uart_union_type.currentIndexChanged.connect(self._on_uart_union_type_changed)
        row_proto.addWidget(self.combo_uart_union_type)
        layout.addLayout(row_proto)

        # Rate info note
        lbl_rate_info = QLabel("⏱️ Chu kỳ gửi: 0.1s (10 Hz) | 5 dữ kiện: 4B union lệch + 1B detect")
        lbl_rate_info.setStyleSheet("color: #ffa726; font-size: 11px;")
        layout.addWidget(lbl_rate_info)

        # Row 3: Connect button & Status
        row3 = QHBoxLayout()
        self.btn_uart_connect = QPushButton("🔌 Kết nối UART")
        self.btn_uart_connect.clicked.connect(self._on_uart_connect_clicked)
        row3.addWidget(self.btn_uart_connect)

        self.lbl_uart_status = QLabel("⚪ Chưa kết nối")
        self.lbl_uart_status.setStyleSheet("color: #aaa; font-weight: bold;")
        row3.addWidget(self.lbl_uart_status, 1)
        layout.addLayout(row3)

        # Row 4: Last sent packet monitor
        row4 = QHBoxLayout()
        self.lbl_uart_packet = QLabel("Gói tin: (chưa gửi)")
        self.lbl_uart_packet.setStyleSheet("font-family: monospace; color: #8cf; font-size: 11px;")
        row4.addWidget(self.lbl_uart_packet, 1)
        layout.addLayout(row4)

        self._update_uart_status_ui()
        self._update_uart_buttons_ui()
        self.control_layout.addWidget(box)

    def _refresh_uart_ports(self):
        curr = self.combo_uart_port.currentText() if hasattr(self, "combo_uart_port") else ""
        ports = UartController.list_available_ports()
        if hasattr(self, "combo_uart_port"):
            self.combo_uart_port.blockSignals(True)
            self.combo_uart_port.clear()
            self.combo_uart_port.addItems(ports)
            if curr and curr in ports:
                self.combo_uart_port.setCurrentText(curr)
            elif "/dev/ttySTM32" in ports:
                self.combo_uart_port.setCurrentText("/dev/ttySTM32")
            self.combo_uart_port.blockSignals(False)

    def _toggle_uart_transmission(self):
        new_state = not getattr(self.uart, "is_enabled", False)
        self._set_uart_enabled(new_state)

    def _on_uart_enable_toggled(self, checked: bool):
        self._set_uart_enabled(checked)

    def _set_uart_enabled(self, enabled: bool):
        self.uart.is_enabled = enabled
        if enabled and not self.uart.is_connected:
            self._connect_uart()

        self._update_uart_buttons_ui()
        status_text = "BẬT (Gửi 5 byte mỗi 0.1s)" if enabled else "TẮT"
        self.status_bar.showMessage(f"Truyền tín hiệu UART STM32: {status_text}")

    def _update_uart_buttons_ui(self):
        is_on = getattr(self.uart, "is_enabled", False)

        # 1. Update quick button on main playback panel
        if hasattr(self, "btn_quick_uart"):
            self.btn_quick_uart.blockSignals(True)
            self.btn_quick_uart.setChecked(is_on)
            if is_on:
                self.btn_quick_uart.setText("🟢 ĐANG GỬI UART STM32 (BẤM ĐỂ TẮT)")
                self.btn_quick_uart.setStyleSheet(
                    "background-color: #2e7d32; color: #ffffff; font-weight: bold; font-size: 13px; "
                    "padding: 8px; border-radius: 5px; border: 2px solid #81c784;"
                )
            else:
                self.btn_quick_uart.setText("📡 BẬT GỬI LỆNH UART (STM32)")
                self.btn_quick_uart.setStyleSheet(
                    "background-color: #37474f; color: #ffb74d; font-weight: bold; font-size: 13px; "
                    "padding: 8px; border-radius: 5px; border: 1px solid #ff9800;"
                )
            self.btn_quick_uart.blockSignals(False)

        # 2. Update toggle button in UART group
        if hasattr(self, "btn_uart_toggle_stream"):
            self.btn_uart_toggle_stream.blockSignals(True)
            self.btn_uart_toggle_stream.setChecked(is_on)
            if is_on:
                self.btn_uart_toggle_stream.setText("🟢 Đang gửi UART xuống STM32")
                self.btn_uart_toggle_stream.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; padding: 6px;")
            else:
                self.btn_uart_toggle_stream.setText("📡 Bật gửi lệnh UART xuống STM32")
                self.btn_uart_toggle_stream.setStyleSheet("background-color: #424242; color: #ffb74d; font-weight: bold; padding: 6px;")
            self.btn_uart_toggle_stream.blockSignals(False)

        # 3. Update checkbox in UART group
        if hasattr(self, "chk_uart_enable"):
            self.chk_uart_enable.blockSignals(True)
            self.chk_uart_enable.setChecked(is_on)
            self.chk_uart_enable.blockSignals(False)

        # 4. Update monitor label if turned off
        if hasattr(self, "lbl_uart_packet") and not is_on:
            self.lbl_uart_packet.setText("Gói tin: (chế độ gửi UART đang tắt)")

    def _on_uart_baud_changed(self, text: str):
        try:
            baud = int(text)
            self.uart.baudrate = baud
            if self.uart.is_connected:
                self._connect_uart()
        except ValueError:
            pass

    def _on_uart_protocol_changed(self, idx: int):
        if idx == 0:
            self.uart.protocol = UartController.PROTOCOL_UNION_5B
        else:
            self.uart.protocol = UartController.PROTOCOL_NMEA_TEXT
        if hasattr(self, "combo_uart_union_type"):
            self.combo_uart_union_type.setEnabled(idx == 0)

    def _on_uart_union_type_changed(self, idx: int):
        if idx == 0:
            self.uart.union_type = UartController.UNION_TYPE_FLOAT
        else:
            self.uart.union_type = UartController.UNION_TYPE_INT32

    def _on_uart_connect_clicked(self):
        if self.uart.is_connected:
            self.uart.disconnect()
            self._update_uart_status_ui()
            self.status_bar.showMessage("Đã ngắt kết nối UART.")
        else:
            self._connect_uart()

    def _connect_uart(self):
        port = self.combo_uart_port.currentText().strip()
        baud = int(self.combo_uart_baud.currentText())
        ok = self.uart.connect(port=port, baudrate=baud)
        self._update_uart_status_ui()
        if ok:
            self.status_bar.showMessage(f"Đã kết nối UART: {port} ({baud} bps)")
        else:
            self.status_bar.showMessage(f"Lỗi kết nối UART: {self.uart.last_error}")

    def _update_uart_status_ui(self):
        if not hasattr(self, "btn_uart_connect") or not hasattr(self, "lbl_uart_status"):
            return
        if self.uart.is_connected:
            self.btn_uart_connect.setText("❌ Ngắt kết nối")
            self.lbl_uart_status.setText(f"🟢 Đã kết nối ({self.uart.port})")
            self.lbl_uart_status.setStyleSheet("color: #81c784; font-weight: bold;")
            self.lbl_uart_status.setToolTip("")
        else:
            self.btn_uart_connect.setText("🔌 Kết nối UART")
            if self.uart.last_error:
                err_short = self.uart.last_error[:22] + "..." if len(self.uart.last_error) > 25 else self.uart.last_error
                self.lbl_uart_status.setText(f"🔴 Lỗi: {err_short}")
                self.lbl_uart_status.setStyleSheet("color: #e57373; font-weight: bold;")
                self.lbl_uart_status.setToolTip(self.uart.last_error)
            else:
                self.lbl_uart_status.setText("⚪ Chưa kết nối")
                self.lbl_uart_status.setStyleSheet("color: #aaa; font-weight: bold;")
                self.lbl_uart_status.setToolTip("")

    def _init_profile_group(self):
        box = QGroupBox("Quản lý Profile & Xuất Dữ liệu")
        layout = QVBoxLayout(box)

        row_name = QHBoxLayout()
        row_name.addWidget(QLabel("Tên Farm:"))
        self.edit_farm_name = QLineEdit("Farm_NhaMang_01")
        row_name.addWidget(self.edit_farm_name)
        layout.addLayout(row_name)

        btn_row = QHBoxLayout()
        self.btn_save_profile = QPushButton("💾 Lưu Profile")
        self.btn_save_profile.clicked.connect(self._on_save_profile_clicked)
        self.btn_load_profile = QPushButton("📂 Nạp Profile")
        self.btn_load_profile.clicked.connect(self._on_load_profile_clicked)
        btn_row.addWidget(self.btn_save_profile)
        btn_row.addWidget(self.btn_load_profile)
        layout.addLayout(btn_row)

        self.btn_export_snapshot = QPushButton("📷 Xuất Frame minh chứng & Telemetry JSON")
        self.btn_export_snapshot.clicked.connect(self._on_export_snapshot_clicked)
        layout.addWidget(self.btn_export_snapshot)

        self.control_layout.addWidget(box)

    # --- YOLO Model Handling ---

    def _update_model_status_label(self):
        if self.segmenter.is_loaded:
            c_str = ", ".join(f"{k}:{v}" for k, v in self.segmenter.class_names.items())
            self.lbl_model_status.setText(f"✓ Đã nạp thành công (Lớp: {c_str})")
            self.lbl_model_status.setStyleSheet("color: #81c784; font-size: 11px;")
        else:
            err = self.segmenter.load_error or "Chưa nạp model"
            self.lbl_model_status.setText(f"⚠️ {err}")
            self.lbl_model_status.setStyleSheet("color: #e57373; font-size: 11px;")

        if hasattr(self, "lbl_device_real"):
            self.lbl_device_real.setText(f"Thiết bị thực tế: {self.segmenter.device_display_name}")
            if self.segmenter.fallback_reason:
                self.lbl_device_real.setStyleSheet("color: #ffb74d; font-size: 11px;")
                self.lbl_device_real.setToolTip(self.segmenter.fallback_reason)
            else:
                self.lbl_device_real.setStyleSheet("color: #81c784; font-size: 11px;")
                self.lbl_device_real.setToolTip("")

    def _populate_model_presets(self):
        if not hasattr(self, "combo_model_preset"):
            return
        self.combo_model_preset.blockSignals(True)
        self.combo_model_preset.clear()
        models_dir = os.path.join(PROJECT_ROOT, "models")
        presets = [
            ("⚡ Nano NCNN Quantized (256px - Siêu chính xác, ~30 FPS)", os.path.join(models_dir, "nhamang_nano_256_ncnn_model")),
            ("🚀 Micro NCNN Quantized (256px - Cân bằng, ~40 FPS)", os.path.join(models_dir, "nhamang_micro_256_ncnn_model")),
            ("⚡ Pico NCNN Quantized (320px - Siêu nhẹ, ~48 FPS)", os.path.join(models_dir, "nhamang_pico_seg_ncnn_model")),
            ("🚀 Nano NCNN Quantized (320px - Chuẩn 320px, ~15 FPS)", os.path.join(models_dir, "nhamang_batden_seg_ncnn_model")),
            ("🎯 Nano PyTorch Gốc (.pt - 416px)", os.path.join(models_dir, "nhamang_batden_seg.pt")),
            ("⚡ Pico PyTorch (.pt - 320px)", os.path.join(models_dir, "nhamang_pico_seg.pt")),
            ("🎯 Thảm đen PyTorch (.pt)", os.path.join(models_dir, "thamden_seg.pt")),
            ("🎯 Bootstrap PyTorch (.pt)", os.path.join(models_dir, "farm_path_bootstrap.pt")),
        ]
        cur_abs = os.path.abspath(self.segmenter.model_path) if self.segmenter.model_path else ""
        sel_idx = -1
        for label, p in presets:
            if os.path.exists(p):
                self.combo_model_preset.addItem(label, p)
                if os.path.abspath(p) == cur_abs:
                    sel_idx = self.combo_model_preset.count() - 1
        if sel_idx >= 0:
            self.combo_model_preset.setCurrentIndex(sel_idx)
        self.combo_model_preset.blockSignals(False)

    def _on_model_preset_changed(self, idx: int):
        if idx < 0:
            return
        path = self.combo_model_preset.itemData(idx)
        if path and os.path.exists(path):
            self._load_yolo_model(path)

    def _on_select_model_clicked(self):
        default_dir = os.path.join(PROJECT_ROOT, "models")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn model AI (.pt hoặc file model.ncnn.param trong thư mục NCNN)",
            default_dir,
            "Tất cả Model AI (*.pt *.param *.onnx);;NCNN Quantized (*.param);;PyTorch Model (*.pt);;Tất cả tệp (*)"
        )
        if path:
            self._load_yolo_model(path)

    def _load_yolo_model(self, path: str):
        self._stop_worker()
        if path.endswith(".param") or path.endswith(".bin") or os.path.basename(path) == "metadata.yaml":
            path = os.path.dirname(path)

        dev = self._get_selected_device()
        ok = self.segmenter.load_model(path, device=dev)
        self.lbl_model_name.setText(os.path.basename(path))
        self._populate_model_presets()

        # Tự động đồng bộ Img Size theo model lượng tử hóa / pico (320px)
        if self.segmenter.fixed_imgsz is not None:
            target_sz = str(self.segmenter.fixed_imgsz)
            self.yolo_imgsz = self.segmenter.fixed_imgsz
            if hasattr(self, "combo_imgsz"):
                if self.combo_imgsz.findText(target_sz) >= 0:
                    self.combo_imgsz.blockSignals(True)
                    self.combo_imgsz.setCurrentText(target_sz)
                    self.combo_imgsz.blockSignals(False)
                self.combo_imgsz.setEnabled(False)
                self.combo_imgsz.setToolTip(f"Model NCNN cố định độ phân giải đầu vào imgsz={target_sz}")
        else:
            if hasattr(self, "combo_imgsz"):
                self.combo_imgsz.setEnabled(True)
                self.combo_imgsz.setToolTip("")
            if "pico" in os.path.basename(path).lower() or "320" in os.path.basename(path).lower():
                self.yolo_imgsz = 320
                self.segmenter.imgsz = 320
                if hasattr(self, "combo_imgsz") and self.combo_imgsz.findText("320") >= 0:
                    self.combo_imgsz.blockSignals(True)
                    self.combo_imgsz.setCurrentText("320")
                    self.combo_imgsz.blockSignals(False)

        self._update_model_status_label()
        if ok:
            self.detector.reset_tracking()
            if self.is_paused and self.current_frame_bgr is not None:
                self._process_single_frame(
                    self.current_frame_bgr,
                    self.reader.current_frame_idx,
                    self.reader.get_timestamp_sec()
                )
            self.status_bar.showMessage(f"Đã nạp model: {os.path.basename(path)} ({self.segmenter.device_display_name})")
        else:
            QMessageBox.critical(self, "Lỗi nạp Model", self.segmenter.load_error)

    def _reload_model(self):
        if self.segmenter.model_path:
            self._load_yolo_model(self.segmenter.model_path)

    def _get_selected_device(self) -> str:
        dev_data = self.combo_device.currentData()
        if dev_data:
            return dev_data
        dev_text = self.combo_device.currentText()
        if "CUDA" in dev_text:
            return "cuda"
        elif "Auto" in dev_text:
            return "auto"
        return "cpu"

    def _on_yolo_conf_changed(self):
        self.yolo_conf = self.slider_yolo_conf.value() / 100.0
        self.segmenter.conf_threshold = self.yolo_conf
        if self.worker:
            self.worker.conf = self.yolo_conf
        if self.is_paused:
            self.yolo_debounce_timer.start(150)

    def _on_yolo_imgsz_changed(self, text: str):
        try:
            self.yolo_imgsz = int(text)
            self.segmenter.imgsz = self.yolo_imgsz
            if self.worker:
                self.worker.imgsz = self.yolo_imgsz
            if self.is_paused:
                self.yolo_debounce_timer.start(150)
        except ValueError:
            pass

    def _on_yolo_device_changed(self, idx: int):
        dev = self._get_selected_device()
        self.yolo_device = dev
        if self.segmenter.model_path:
            self._load_yolo_model(self.segmenter.model_path)

    # --- Video & Playback Handling ---

    def _on_open_camera_clicked(self):
        """Connects directly to the connected Orbbec Astra Pro HD Camera (device 0)."""
        self.open_video("0")

    def _on_open_video_clicked(self):
        start_dir = PROJECT_ROOT
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn file video", start_dir, "Video files (*.mp4 *.avi *.mkv *.mov)"
        )
        if path:
            self.open_video(path)

    def open_video(self, video_path: str):
        self._stop_worker()
        if not self.reader.open(video_path):
            QMessageBox.critical(self, "Lỗi mở file / camera", f"Không thể mở file video hoặc camera:\n{video_path}")
            return

        if self.reader.is_live:
            self.lbl_file_info.setText(f"Camera Orbbec Live ({self.reader.width}x{self.reader.height}, {self.reader.fps:.1f} FPS)")
            self.slider_seek.setEnabled(False)
            self.btn_step_prev.setEnabled(False)
            self.btn_step_next.setEnabled(False)
            self.combo_speed.setEnabled(False)
            self.is_paused = False
            self.btn_play.setText("⏸ Tạm dừng Stream")
            self._start_worker()
            self.status_bar.showMessage("Đang phát trực tiếp từ Camera Orbbec Astra Pro.")
        else:
            filename = os.path.basename(video_path)
            self.lbl_file_info.setText(f"{filename} ({self.reader.width}x{self.reader.height}, {self.reader.fps:.1f} FPS)")
            self.slider_seek.setEnabled(True)
            self.btn_step_prev.setEnabled(True)
            self.btn_step_next.setEnabled(True)
            self.combo_speed.setEnabled(True)
            self.slider_seek.setRange(0, max(0, self.reader.total_frames - 1))
            self.slider_seek.setValue(0)
            self.is_paused = True
            self.btn_play.setText("▶ Phát (Play)")
            # Read and display initial frame
            self._seek_to_frame(0)
            self.status_bar.showMessage(f"Đã mở: {filename}. Tạm dừng tại frame 0.")

    def _toggle_play_pause(self):
        if not self.reader.is_opened:
            QMessageBox.information(self, "Thông báo", "Vui lòng mở file video hoặc kết nối camera trước khi phát.")
            return

        if self.is_paused:
            self.is_paused = False
            self.btn_play.setText("⏸ Tạm dừng" if not self.reader.is_live else "⏸ Tạm dừng Stream")
            self._start_worker()
        else:
            self.is_paused = True
            self.btn_play.setText("▶ Phát (Play)" if not self.reader.is_live else "▶ Tiếp tục Stream")
            self._stop_worker()

    def _start_worker(self):
        self._stop_worker()
        self.current_session_id += 1
        self.worker = ProcessingWorker(
            reader=self.reader,
            segmenter=self.segmenter,
            detector=self.detector,
            color_model=self.color_model,
            mode=self.active_mode,
            conf=self.yolo_conf,
            imgsz=self.yolo_imgsz,
            session_id=self.current_session_id,
            uart=self.uart
        )
        speed_str = self.combo_speed.currentText().replace("x", "")
        self.worker.speed_multiplier = float(speed_str)
        self.worker.packetReady.connect(self._on_packet_ready)
        self.worker.finishedPlayback.connect(self._on_playback_finished)
        self.worker.start()

    def _stop_worker(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

    def _on_playback_finished(self):
        self.is_paused = True
        self.btn_play.setText("▶ Phát (Play)")
        self.status_bar.showMessage("Đã kết thúc video.")

    def _on_speed_changed(self, text: str):
        speed = float(text.replace("x", ""))
        if self.worker:
            self.worker.speed_multiplier = speed

    def _on_seek_slider_moved(self, value: int):
        self._seek_to_frame(value)

    def _step_frame(self, delta: int):
        if not self.reader.is_opened:
            return
        if not self.is_paused:
            self._toggle_play_pause()
        target = self.reader.current_frame_idx + delta
        self._seek_to_frame(target)

    def _seek_to_frame(self, frame_idx: int):
        self.current_session_id += 1
        ret, frame = self.reader.seek(frame_idx)
        if ret and frame is not None:
            # Resets tracking state upon seek
            self.detector.reset_tracking()
            self.segmenter.reset_tracking()
            self._process_single_frame(frame, frame_idx, self.reader.get_timestamp_sec())
            self.slider_seek.blockSignals(True)
            self.slider_seek.setValue(frame_idx)
            self.slider_seek.blockSignals(False)

    @Slot(int)
    def _on_packet_ready(self, session_id: int):
        if session_id != self.current_session_id or self.worker is None:
            return
        pkt = self.worker.get_latest_packet()
        if pkt is None:
            return
        self._consume_packet(pkt)

    @Slot(np.ndarray, object, object, object, int, float, int)
    def _on_frame_processed(
        self,
        frame: np.ndarray,
        yolo_res: Optional[YoloResult],
        det_res: DetectionResult,
        score_map: Optional[np.ndarray],
        frame_idx: int,
        timestamp: float,
        session_id: int
    ):
        if session_id != self.current_session_id:
            return
        pkt = ProcessedPacket(
            frame=frame,
            yolo_res=yolo_res,
            det_res=det_res,
            score_map=score_map,
            frame_idx=frame_idx,
            timestamp=timestamp,
            session_id=session_id
        )
        self._consume_packet(pkt)

    def _consume_packet(self, pkt: ProcessedPacket):
        if getattr(self, "_is_rendering_frame", False):
            return
        self._is_rendering_frame = True

        try:
            self.current_frame_bgr = pkt.frame
            self.last_yolo_result = pkt.yolo_res
            self.last_detection_result = pkt.det_res
            self.last_score_map = pkt.score_map

            self.slider_seek.blockSignals(True)
            self.slider_seek.setValue(pkt.frame_idx)
            self.slider_seek.blockSignals(False)

            self._update_time_label(pkt.timestamp, pkt.frame_idx)

            t_render_start = time.perf_counter()
            self._render_views(pkt.frame, pkt.yolo_res, pkt.det_res, pkt.score_map)
            render_ms = (time.perf_counter() - t_render_start) * 1000.0

            # Calculate actual UI Display FPS
            now = time.perf_counter()
            if self.last_render_time > 0:
                dt = now - self.last_render_time
                if dt > 0:
                    inst_fps = 1.0 / dt
                    self.display_fps = 0.85 * self.display_fps + 0.15 * inst_fps
            else:
                self.display_fps = 30.0
            self.last_render_time = now

            self._update_status_ui(pkt.yolo_res, pkt.det_res)
            self._update_telemetry(pkt, render_ms)
        finally:
            self._is_rendering_frame = False

    def _update_telemetry(self, pkt: ProcessedPacket, render_ms: float):
        total_pipeline = pkt.read_time_ms + pkt.pipeline_time_ms
        pipe_fps = (1000.0 / total_pipeline) if total_pipeline > 0 else 0.0

        if self.active_mode == self.MODE_AI:
            dev_str = self.segmenter.device_display_name
            self.lbl_device_real.setText(f"Thiết bị: {dev_str}")
            if self.segmenter.fallback_reason:
                self.lbl_device_real.setToolTip(self.segmenter.fallback_reason)
                self.lbl_device_real.setStyleSheet("color: #ffb74d; font-size: 11px;")
            else:
                self.lbl_device_real.setStyleSheet("color: #81c784; font-size: 11px;")

            self.lbl_latency_breakdown.setText(
                f"Đọc: {pkt.read_time_ms:.1f}ms | YOLO: {pkt.yolo_time_ms:.1f}ms | Tìm biên: {pkt.detect_time_ms:.1f}ms | Render UI: {render_ms:.1f}ms"
            )
        else:
            self.lbl_device_real.setText("Thiết bị: CPU (CIELAB Color)")
            self.lbl_device_real.setStyleSheet("color: #80cbc4; font-size: 11px;")
            cielab_ms = max(0.0, pkt.pipeline_time_ms - pkt.detect_time_ms)
            self.lbl_latency_breakdown.setText(
                f"Đọc: {pkt.read_time_ms:.1f}ms | CIELAB: {cielab_ms:.1f}ms | Tìm biên: {pkt.detect_time_ms:.1f}ms | Render UI: {render_ms:.1f}ms"
            )

        self.lbl_fps.setText(
            f"Xử lý: {pipe_fps:.0f} FPS | Hiển thị: {self.display_fps:.0f} FPS (Bỏ qua: {pkt.dropped_frames_total} frame)"
        )

        if hasattr(self, "lbl_uart_packet") and hasattr(self, "uart"):
            if self.uart.is_enabled and self.uart.is_connected:
                pkt_str = self.uart.last_sent_packet or "(đang chờ gửi)"
                self.lbl_uart_packet.setText(f"Gói tin: {pkt_str.strip()}")
            elif self.uart.is_enabled and not self.uart.is_connected:
                self.lbl_uart_packet.setText("Gói tin: (chưa kết nối cổng serial)")

    def _update_time_label(self, timestamp: float, frame_idx: int):
        total_time_sec = (self.reader.total_frames / self.reader.fps) if self.reader.fps > 0 else 0.0
        cur_min, cur_sec = divmod(int(timestamp), 60)
        tot_min, tot_sec = divmod(int(total_time_sec), 60)
        self.lbl_time.setText(f"{cur_min:02d}:{cur_sec:02d} / {tot_min:02d}:{tot_sec:02d} (F: {frame_idx}/{self.reader.total_frames})")

    def _fast_reprocess_detector(self):
        """Immediately re-processes detector on current paused frame without heavy YOLO re-inference."""
        if not self.is_paused or self.current_frame_bgr is None:
            return

        t_start = time.perf_counter()
        if self.active_mode == self.MODE_AI:
            if self.last_yolo_result is not None:
                mask = self.last_yolo_result.chosen_mask
                valid = self.last_yolo_result.is_valid
            else:
                mask = None
                valid = False
            score_map = None
        else:
            h_img, w_img = self.current_frame_bgr.shape[:2]
            roi_rect = self.detector.compute_roi((h_img, w_img))
            if self.color_model.is_trained:
                score_map = self.color_model.score_frame(self.current_frame_bgr, roi_rect)
                valid = True
            else:
                rx, ry, rw, rh = roi_rect
                score_map = np.zeros((rh, rw), dtype=np.uint8)
                valid = False
            self.last_score_map = score_map
            mask = score_map

        det_res = self.detector.process(
            frame_bgr=self.current_frame_bgr,
            mask_or_score=mask,
            is_source_valid=valid
        )
        self.last_detection_result = det_res
        t_detect_ms = (time.perf_counter() - t_start) * 1000.0

        t_render_start = time.perf_counter()
        self._render_views(self.current_frame_bgr, self.last_yolo_result, det_res, score_map)
        render_ms = (time.perf_counter() - t_render_start) * 1000.0
        self._update_status_ui(self.last_yolo_result, det_res)

        self.lbl_latency_breakdown.setText(f"Tìm biên: {t_detect_ms:.1f}ms | Render UI: {render_ms:.1f}ms")

    def _on_yolo_debounce_timeout(self):
        if self.is_paused and self.current_frame_bgr is not None:
            self._process_single_frame(
                self.current_frame_bgr,
                self.reader.current_frame_idx,
                self.reader.get_timestamp_sec()
            )

    # --- Pipeline Processing for Paused Frames ---

    def _process_single_frame(self, frame_bgr: np.ndarray, frame_idx: int, timestamp: float):
        t_start = time.perf_counter()
        self.current_frame_bgr = frame_bgr.copy()
        self._update_time_label(timestamp, frame_idx)

        roi_rect = self.detector.compute_roi(frame_bgr.shape[:2])
        yolo_ms = 0.0

        if self.active_mode == self.MODE_AI:
            yolo_res = self.segmenter.predict(
                frame_bgr=frame_bgr,
                conf=self.yolo_conf,
                imgsz=self.yolo_imgsz,
                roi_rect=roi_rect
            )
            self.last_yolo_result = yolo_res
            yolo_ms = yolo_res.inference_time_ms if yolo_res else 0.0

            t_det_start = time.perf_counter()
            det_res = self.detector.process(
                frame_bgr=frame_bgr,
                mask_or_score=yolo_res.chosen_mask,
                is_source_valid=yolo_res.is_valid
            )
            detect_ms = (time.perf_counter() - t_det_start) * 1000.0
            score_map = None
            self.last_score_map = None
        else:
            yolo_res = None
            self.last_yolo_result = None
            if self.color_model.is_trained:
                score_map = self.color_model.score_frame(frame_bgr, roi_rect)
                is_valid = True
            else:
                rx, ry, rw, rh = roi_rect
                score_map = np.zeros((rh, rw), dtype=np.uint8)
                is_valid = False

            self.last_score_map = score_map
            t_det_start = time.perf_counter()
            det_res = self.detector.process(
                frame_bgr=frame_bgr,
                mask_or_score=score_map,
                is_source_valid=is_valid
            )
            detect_ms = (time.perf_counter() - t_det_start) * 1000.0

        self.last_detection_result = det_res

        # UART transmission if enabled
        if hasattr(self, "uart") and self.uart and self.uart.is_enabled:
            state_code = UartController.STATE_LOST
            if det_res.state == TrackingState.TRACKING:
                state_code = UartController.STATE_TRACKING
            elif det_res.state == TrackingState.LOW_CONFIDENCE:
                state_code = UartController.STATE_LOW_CONF
            elif det_res.state == TrackingState.NO_MODEL:
                state_code = UartController.STATE_NO_MODEL

            target_pt = det_res.path_center if (det_res.center_method == "path" and det_res.path_center is not None) else det_res.box_center
            tx = int(target_pt[0]) if target_pt else None
            ty = int(target_pt[1]) if target_pt else None

            is_detected = bool(
                (det_res.state == TrackingState.TRACKING or 
                 (det_res.state == TrackingState.LOW_CONFIDENCE and det_res.lateral_offset_px is not None))
            )

            self.uart.send_nav_update(
                state_code=state_code,
                offset_px=det_res.lateral_offset_px,
                confidence_pct=det_res.confidence,
                target_x=tx,
                target_y=ty,
                is_detected=is_detected
            )

        t_render_start = time.perf_counter()
        self._render_views(frame_bgr, yolo_res, det_res, score_map)
        render_ms = (time.perf_counter() - t_render_start) * 1000.0

        self._update_status_ui(yolo_res, det_res)

        if hasattr(self, "lbl_uart_packet") and hasattr(self, "uart"):
            if self.uart.is_enabled and self.uart.is_connected:
                pkt_str = self.uart.last_sent_packet or "(đang chờ gửi)"
                self.lbl_uart_packet.setText(f"Gói tin: {pkt_str.strip()}")
            elif self.uart.is_enabled and not self.uart.is_connected:
                self.lbl_uart_packet.setText("Gói tin: (chưa kết nối cổng serial)")

        total_ms = (time.perf_counter() - t_start) * 1000.0
        fps = (1000.0 / total_ms) if total_ms > 0 else 0.0

        if self.active_mode == self.MODE_AI:
            self.lbl_latency_breakdown.setText(
                f"YOLO: {yolo_ms:.1f}ms | Tìm biên: {detect_ms:.1f}ms | Render UI: {render_ms:.1f}ms"
            )
        else:
            self.lbl_latency_breakdown.setText(
                f"CIELAB: {detect_ms:.1f}ms | Render UI: {render_ms:.1f}ms"
            )
        self.lbl_fps.setText(f"Tạm dừng ({total_ms:.1f} ms / ~{fps:.0f} FPS)")

    def _reprocess_current_frame(self):
        """Immediately re-processes the currently paused frame when sliders change."""
        if self.is_paused and self.current_frame_bgr is not None:
            self._process_single_frame(
                self.current_frame_bgr,
                self.reader.current_frame_idx,
                self.reader.get_timestamp_sec()
            )

    # --- View Rendering ---

    def _render_views(
        self,
        frame_bgr: np.ndarray,
        yolo_res: Optional[YoloResult],
        det_res: DetectionResult,
        score_map: Optional[np.ndarray]
    ):
        orig_h, orig_w = frame_bgr.shape[:2]
        rx, ry, rw, rh = det_res.roi_rect
        cam_cx = int(orig_w / 2)
        is_ai = (self.active_mode == self.MODE_AI)

        # 1. Downscale frame once for fast UI rendering (Max 640px width)
        disp_w = min(640, orig_w)
        disp_h = int(orig_h * (disp_w / float(orig_w)))
        sx = disp_w / float(orig_w)
        sy = disp_h / float(orig_h)

        disp_frame = cv2.resize(frame_bgr, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

        # Scale ROI coordinates to display resolution
        drx = int(round(rx * sx))
        dry = int(round(ry * sy))
        drw = int(round(rw * sx))
        drh = int(round(rh * sy))
        dcam_cx = int(round(cam_cx * sx))

        # -------------------------------------------------------------
        # 1. View 1: Original + ROI rectangle + Patches (if Color mode)
        # -------------------------------------------------------------
        view1_bgr = disp_frame.copy()
        cv2.rectangle(view1_bgr, (drx, dry), (drx + drw, dry + drh), (0, 255, 255), 2)
        cv2.putText(view1_bgr, f"ROI [{rw}x{rh} px]", (drx + 6, max(18, dry + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.line(view1_bgr, (dcam_cx, dry), (dcam_cx, dry + drh), (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(view1_bgr, "Moc camera", (max(10, dcam_cx - 35), max(15, dry - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

        # In color mode, draw existing patches on View 1
        if not is_ai:
            for p in self.color_model.patches:
                px, py, pw, ph = p.rect
                col = (0, 255, 0) if p.category == "path" else (0, 0, 255)
                dpx = int(round(px * sx))
                dpy = int(round(py * sy))
                dpw = int(round(pw * sx))
                dph = int(round(ph * sy))
                cv2.rectangle(view1_bgr, (dpx, dpy), (dpx + dpw, dpy + dph), col, 1)

        self._set_label_image(self.view_original, view1_bgr, (orig_w, orig_h))

        # -------------------------------------------------------------
        # 2. View 2: Detection Overlay + Bounding Box + Boundaries + Centerline
        # -------------------------------------------------------------
        view2_bgr = disp_frame.copy()

        # ROI-Clipped Path Mask Overlay (GUARANTEED NO OVERFLOW OUTSIDE ROI)
        roi_mask = det_res.binary_mask
        if roi_mask is not None and np.any(roi_mask > 0) and drw > 0 and drh > 0:
            disp_roi_mask = cv2.resize(roi_mask, (drw, drh), interpolation=cv2.INTER_NEAREST)
            roi_crop = view2_bgr[dry:dry + drh, drx:drx + drw]
            tint_color = (210, 220, 20) if is_ai else (0, 220, 220)
            tint = np.full_like(roi_crop, tint_color, dtype=np.uint8)
            blended = cv2.addWeighted(roi_crop, 0.60, tint, 0.40, 0)
            np.copyto(roi_crop, blended, where=(disp_roi_mask > 0)[..., None])

        # ROI outline & camera center axis
        cv2.rectangle(view2_bgr, (drx, dry), (drx + drw, dry + drh), (100, 180, 255), 1)
        cv2.line(view2_bgr, (dcam_cx, dry), (dcam_cx, dry + drh), (240, 240, 240), 1, cv2.LINE_AA)

        # Mode Badge & Realtime GPU / CPU Indicator
        if is_ai and yolo_res is not None and yolo_res.chosen_instance is not None:
            c_name = yolo_res.chosen_instance.class_name
            c_conf = yolo_res.chosen_instance.confidence
            t_ms = yolo_res.inference_time_ms or 0.0
            dev_str = "GPU" if ("cuda" in self.segmenter.effective_device.lower() and not self.segmenter.fallback_reason) else "CPU"
            badge_text = f"{c_name} ({c_conf:.2f}) | {dev_str}: {t_ms:.1f}ms"
            cv2.rectangle(view2_bgr, (10, 8), (320, 36), (20, 20, 20), -1)
            cv2.rectangle(view2_bgr, (10, 8), (320, 36), (0, 200, 100), 1)
            cv2.putText(view2_bgr, badge_text, (16, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120), 1)
        elif not is_ai:
            n_patches = len(self.color_model.patches)
            cv2.rectangle(view2_bgr, (10, 8), (280, 36), (20, 20, 20), -1)
            cv2.rectangle(view2_bgr, (10, 8), (280, 36), (255, 180, 0), 1)
            cv2.putText(view2_bgr, f"CIELAB ({n_patches} mẫu)", (16, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1)

        # Centering and Navigation Overlay (Path Centerline vs Bounding Box Midpoint)
        if det_res.center_method == "path":
            # --- Option 2: Tâm đường dẫn (Sinh path centerline) ---
            if det_res.centerline_points:
                pts_scaled = np.array(
                    [(int(round(x * sx)), int(round(y * sy))) for x, y in det_res.centerline_points],
                    dtype=np.int32
                )
                cv2.polylines(view2_bgr, [pts_scaled], False, (0, 0, 255), 2, cv2.LINE_AA)

            if det_res.extrapolated_points:
                extrap_scaled = [(int(round(x * sx)), int(round(y * sy))) for x, y in det_res.extrapolated_points]
                draw_dashed_polyline(view2_bgr, extrap_scaled, (255, 0, 255), 2, dash_len=6, gap_len=5)

            if det_res.ref_y_global is not None:
                ref_s = next((s for s in det_res.scanlines if s.y_global == det_res.ref_y_global and s.is_valid), None)
                if ref_s and ref_s.x_left is not None and ref_s.x_right is not None:
                    sy_ref = int(round(det_res.ref_y_global * sy))
                    sx_l = int(round(ref_s.x_left * sx))
                    sx_r = int(round(ref_s.x_right * sx))
                    cv2.line(view2_bgr, (sx_l, sy_ref), (sx_r, sy_ref), (0, 220, 255), 2)
                    cv2.circle(view2_bgr, (sx_l, sy_ref), 4, (0, 140, 255), -1)
                    cv2.circle(view2_bgr, (sx_r, sy_ref), 4, (0, 140, 255), -1)
                    cv2.putText(view2_bgr, f"Lát cắt Y={det_res.ref_y_global}",
                                (max(10, sx_l - 10), sy_ref - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1)

            if det_res.path_center is not None:
                spx = int(round(det_res.path_center[0] * sx))
                spy = int(round(det_res.path_center[1] * sy))
                cv2.circle(view2_bgr, (spx, spy), 6, (0, 255, 255), -1)
                cv2.circle(view2_bgr, (spx, spy), 9, (0, 0, 255), 2)
                cv2.putText(view2_bgr, f"Tâm path ({det_res.path_center[0]}, {det_res.path_center[1]})",
                            (min(disp_w - 140, spx + 10), spy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

                if det_res.lateral_offset_px is not None:
                    cv2.circle(view2_bgr, (dcam_cx, spy), 4, (255, 255, 255), -1)
                    cv2.arrowedLine(view2_bgr, (dcam_cx, spy), (spx, spy), (0, 255, 255), 2, tipLength=0.20)
                    lbl_x = max(10, min(disp_w - 180, (dcam_cx + spx) // 2 - 40))
                    cv2.putText(view2_bgr, det_res.offset_direction, (lbl_x, spy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            if det_res.bounding_box is not None:
                bx, by, bw, bh = det_res.bounding_box
                cv2.rectangle(view2_bgr, (int(bx * sx), int(by * sy)),
                              (int((bx + bw) * sx), int((by + bh) * sy)), (0, 200, 0), 1)

        else:
            # --- Option 1: Tâm Bounding Box (Bắt vùng detect) ---
            if det_res.bounding_box is not None and det_res.box_center is not None:
                bx, by, bw, bh = det_res.bounding_box
                cx, cy = det_res.box_center
                sbx = int(round(bx * sx))
                sby = int(round(by * sy))
                sbw = int(round(bw * sx))
                sbh = int(round(bh * sy))
                scx = int(round(cx * sx))
                scy = int(round(cy * sy))

                # 1. Bounding box (Green)
                cv2.rectangle(view2_bgr, (sbx, sby), (sbx + sbw, sby + sbh), (0, 255, 0), 2)
                cv2.putText(view2_bgr, f"Lối đi [{bw}x{bh} px]", (sbx + 6, max(20, sby + 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                # 2. Straight vertical center axis along box
                cv2.line(view2_bgr, (scx, sby), (scx, sby + sbh), (0, 0, 255), 2, cv2.LINE_AA)

                # 3. Center point
                cv2.circle(view2_bgr, (scx, scy), 6, (0, 255, 255), -1)
                cv2.circle(view2_bgr, (scx, scy), 9, (0, 0, 255), 2)
                cv2.putText(view2_bgr, f"Tâm box ({cx}, {cy})", (min(disp_w - 130, scx + 10), scy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

                # 4. Camera reference point & Deviation arrow
                if det_res.lateral_offset_px is not None:
                    cv2.circle(view2_bgr, (dcam_cx, scy), 4, (255, 255, 255), -1)
                    cv2.arrowedLine(view2_bgr, (dcam_cx, scy), (scx, scy), (0, 255, 255), 2, tipLength=0.20)
                    lbl_x = max(10, min(disp_w - 160, (dcam_cx + scx) // 2 - 40))
                    cv2.putText(view2_bgr, det_res.offset_direction, (lbl_x, scy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # State banner if lost or no model
        if det_res.state in (TrackingState.LOST, TrackingState.NO_MODEL):
            cv2.rectangle(view2_bgr, (disp_w // 2 - 120, 40), (disp_w // 2 + 120, 75), (0, 0, 180), -1)
            cv2.putText(view2_bgr, f"[ {det_res.state.value} ]", (disp_w // 2 - 105, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

        self._set_label_image(self.view_overlay, view2_bgr, (orig_w, orig_h))

        # -------------------------------------------------------------
        # 3. View 3: Binary Mask (Clipped strictly inside ROI)
        # -------------------------------------------------------------
        mask_bgr = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
        if roi_mask is not None and np.any(roi_mask > 0) and drw > 0 and drh > 0:
            disp_roi_mask = cv2.resize(roi_mask, (drw, drh), interpolation=cv2.INTER_NEAREST)
            mask_bgr[dry:dry + drh, drx:drx + drw] = cv2.cvtColor(disp_roi_mask, cv2.COLOR_GRAY2BGR)

        cv2.rectangle(mask_bgr, (drx, dry), (drx + drw, dry + drh), (100, 100, 255), 1)
        self._set_label_image(self.view_mask, mask_bgr, (orig_w, orig_h))

        # -------------------------------------------------------------
        # 4. View 4: Mode-dependent (AI Debug vs Color Heatmap) [15 FPS throttled]
        # -------------------------------------------------------------
        render_v4 = getattr(self, "chk_view4_enable", None) is None or self.chk_view4_enable.isChecked()
        now = time.perf_counter()
        time_to_render_v4 = self.is_paused or (now - self._last_v4_render_time >= 0.066)

        if render_v4 and time_to_render_v4:
            self._last_v4_render_time = now
            if is_ai and yolo_res is not None:
                v4_bgr = cv2.convertScaleAbs(disp_frame, alpha=0.5)
                inst_colors = [(255, 130, 0), (255, 0, 255), (0, 255, 255), (0, 180, 255), (180, 100, 255)]

                for inst in yolo_res.all_instances:
                    col = inst_colors[inst.instance_id % len(inst_colors)]
                    # Scale instance mask to thumbnail
                    disp_inst_mask = cv2.resize(inst.mask, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
                    mask_i = disp_inst_mask > 0
                    if np.any(mask_i):
                        color_tint = np.full_like(v4_bgr, col, dtype=np.uint8)
                        blended = cv2.addWeighted(v4_bgr, 0.5, color_tint, 0.5, 0)
                        np.copyto(v4_bgr, blended, where=mask_i[..., None])

                    bx, by, bw, bh = inst.bbox
                    is_ch = (inst.instance_id == yolo_res.chosen_index)
                    border_col = (0, 255, 0) if is_ch else col
                    sbx = int(round(bx * sx))
                    sby = int(round(by * sy))
                    sbw = int(round(bw * sx))
                    sbh = int(round(bh * sy))
                    cv2.rectangle(v4_bgr, (sbx, sby), (sbx + sbw, sby + sbh), border_col, 2 if is_ch else 1)
                    tag = f"#{inst.instance_id} {inst.class_name} ({inst.confidence:.2f})" + (" [CHOSEN]" if is_ch else "")
                    cv2.putText(v4_bgr, tag, (sbx + 4, max(15, sby + 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, border_col, 1)

                # Target center
                target_pt = det_res.path_center if (det_res.center_method == "path" and det_res.path_center is not None) else det_res.box_center
                if target_pt is not None:
                    scx = int(round(target_pt[0] * sx))
                    scy = int(round(target_pt[1] * sy))
                    cv2.circle(v4_bgr, (scx, scy), 5, (0, 255, 255), -1)

                method_lbl = "Tâm Path" if det_res.center_method == "path" else "Tâm Box"
                cv2.putText(v4_bgr, f"Debug AI [{method_lbl}]: {len(yolo_res.all_instances)} vùng | Chọn: #{yolo_res.chosen_index}",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

            else:
                full_score = np.zeros((disp_h, disp_w), dtype=np.uint8)
                if score_map is not None and score_map.size > 0 and drw > 0 and drh > 0:
                    disp_score = cv2.resize(score_map, (drw, drh), interpolation=cv2.INTER_LINEAR)
                    full_score[dry:dry + drh, drx:drx + drw] = disp_score

                v4_bgr = cv2.applyColorMap(full_score, cv2.COLORMAP_JET)
                cv2.rectangle(v4_bgr, (drx, dry), (drx + drw, dry + drh), (255, 255, 255), 1)
                cv2.putText(v4_bgr, f"Score Map CIELAB", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

            self._set_label_image(self.view_debug, v4_bgr, (orig_w, orig_h))
        elif not render_v4:
            v4_blank = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
            cv2.putText(v4_blank, "View 4 đã tắt (Tiết kiệm CPU)", (disp_w // 2 - 130, disp_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
            self._set_label_image(self.view_debug, v4_blank, (orig_w, orig_h))

    def _set_label_image(self, label: InteractiveImageLabel, img_bgr: np.ndarray, orig_size: Optional[Tuple[int, int]] = None):
        h, w, ch = img_bgr.shape
        bytes_per_line = ch * w
        qimg = QImage(img_bgr.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qimg)
        label.set_display_pixmap(pix, orig_size or (w, h))

    def _update_status_ui(self, yolo_res: Optional[YoloResult], det_res: DetectionResult):
        is_ai = (self.active_mode == self.MODE_AI)

        # Update badge
        state_colors = {
            TrackingState.NO_MODEL: ("#616161", "Chưa nạp model" if is_ai else "Chưa học màu"),
            TrackingState.TRACKING: ("#2e7d32", "Đang theo dõi"),
            TrackingState.LOW_CONFIDENCE: ("#f57f17", "Độ tin cậy thấp"),
            TrackingState.LOST: ("#c62828", "Mất đường")
        }
        bg_col, text = state_colors.get(det_res.state, ("#616161", str(det_res.state.value)))
        mode_suffix = " (AI)" if is_ai else " (Xử lý ảnh)"
        self.lbl_status_badge.setText(f"{text}{mode_suffix}")
        self.lbl_status_badge.setStyleSheet(
            f"background-color: {bg_col}; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;"
        )

        # Geometric confidence
        self.lbl_confidence.setText(f"{det_res.confidence:.1f}%")
        if det_res.confidence >= 65.0:
            self.lbl_confidence.setStyleSheet("font-size: 13px; font-weight: bold; color: #81c784;")
        elif det_res.confidence >= 35.0:
            self.lbl_confidence.setStyleSheet("font-size: 13px; font-weight: bold; color: #ffb74d;")
        else:
            self.lbl_confidence.setStyleSheet("font-size: 13px; font-weight: bold; color: #e57373;")

        # Sub metric
        if is_ai:
            self.lbl_sub_metric_title.setText("Độ tự tin YOLO instance:")
            if yolo_res and yolo_res.chosen_confidence is not None:
                self.lbl_sub_metric_val.setText(f"{yolo_res.chosen_confidence * 100:.1f}%")
            else:
                self.lbl_sub_metric_val.setText("N/A")
        else:
            self.lbl_sub_metric_title.setText("Số mẫu màu CIELAB đã nạp:")
            self.lbl_sub_metric_val.setText(f"{len(self.color_model.patches)} mẫu")

        # Offset
        self.lbl_offset.setText(det_res.offset_direction)

        # Ref slice
        if det_res.ref_y_global is not None:
            lbl_type = "lát cắt tham chiếu path" if det_res.center_method == "path" else "tâm bounding box"
            self.lbl_ref_slice.setText(f"Y = {det_res.ref_y_global} px ({lbl_type})")
        else:
            self.lbl_ref_slice.setText("N/A (không có lát cắt đủ 2 biên)")

    # --- Parameter Adjustments ---

    def _on_center_method_changed(self, idx: int):
        method = self.combo_center_method.currentData()
        self.detector.center_method = method
        self.detector.reset_tracking()
        self._reprocess_current_frame()

    def _on_roi_changed(self):
        self.detector.roi_top_pct = self.slider_roi_top.value() / 100.0
        self.detector.roi_height_pct = self.slider_roi_height.value() / 100.0
        self.detector.roi_width_pct = self.slider_roi_width.value() / 100.0
        self.detector.roi_center_x_pct = self.slider_roi_center_x.value() / 100.0
        self.detector.reset_tracking()
        self._reprocess_current_frame()

    def _on_detector_param_changed(self):
        self.detector.threshold = self.slider_threshold.value()
        self.detector.k_open = self.slider_open.value()
        self.detector.k_close = self.slider_close.value()
        self.detector.smoothing_factor = self.slider_smooth.value() / 100.0
        self.detector.num_scanlines = self.slider_scanlines.value()
        self._reprocess_current_frame()

    # --- Color Sampling Interactions (For Image Processing Mode) ---

    def _sample_from_roi(self):
        """Extracts a representative path patch from current ROI center."""
        if self.current_frame_bgr is None:
            self.status_bar.showMessage("Chưa có khung hình video để lấy mẫu!")
            return

        h_img, w_img = self.current_frame_bgr.shape[:2]
        rx, ry, rw, rh = self.detector.compute_roi((h_img, w_img))
        px = rx + int(rw * 0.25)
        py = ry + int(rh * 0.20)
        pw = max(10, int(rw * 0.50))
        ph = max(10, int(rh * 0.55))
        rect = (px, py, pw, ph)

        p = self.color_model.add_patch(self.current_frame_bgr, rect, "path", label=f"Bạt ROI #{self.color_model._next_patch_id}")
        if p:
            self._update_patches_table()
            self.status_bar.showMessage(f"Đã tạo mẫu Bạt #{p.patch_id} từ tâm ROI: {rect} và học màu.")
            self._reprocess_current_frame()

    def _on_sample_rect_selected(self, qrect: QRect):
        if self.current_frame_bgr is None:
            return

        h_img, w_img = self.current_frame_bgr.shape[:2]
        rx, ry, rw, rh = self.detector.compute_roi((h_img, w_img))
        qx, qy, qw, qh = qrect.x(), qrect.y(), qrect.width(), qrect.height()
        x1 = max(rx, qx)
        y1 = max(ry, qy)
        x2 = min(rx + rw, qx + qw)
        y2 = min(ry + rh, qy + qh)

        if (x2 - x1) < 8 or (y2 - y1) < 8:
            self.status_bar.showMessage("Vùng mẫu quá nhỏ hoặc nằm ngoài khung ROI! Vui lòng chọn bên trong khung ROI (viền vàng).")
            return

        rect = (x1, y1, x2 - x1, y2 - y1)
        p = self.color_model.add_patch(self.current_frame_bgr, rect, "path", label=f"Bạt #{self.color_model._next_patch_id}")
        if p:
            # If in AI mode, automatically switch to Color mode to display sampled patch results
            if self.active_mode != self.MODE_COLOR:
                self.combo_mode.setCurrentIndex(1)
            else:
                self._update_patches_table()
                self._reprocess_current_frame()
            self.status_bar.showMessage(f"Đã thêm mẫu Bạt #{p.patch_id} tại {rect} (trong ROI).")

    def _update_patches_table(self):
        self.table_patches.setRowCount(0)
        for p in self.color_model.patches:
            row = self.table_patches.rowCount()
            self.table_patches.insertRow(row)

            item_id = QTableWidgetItem(str(p.patch_id))
            item_cat = QTableWidgetItem("Bạt (PATH)")
            item_cat.setForeground(QtGui.QBrush(QColor("#81c784")))

            x, y, w, h = p.rect
            item_rect = QTableWidgetItem(f"{x},{y} ({w}x{h})")

            btn_del = QPushButton("Xóa")
            btn_del.setStyleSheet("padding: 2px 6px; font-size: 11px;")
            btn_del.clicked.connect(lambda _, pid=p.patch_id: self._delete_patch(pid))

            self.table_patches.setItem(row, 0, item_id)
            self.table_patches.setItem(row, 1, item_cat)
            self.table_patches.setItem(row, 2, item_rect)
            self.table_patches.setCellWidget(row, 3, btn_del)

    def _delete_patch(self, patch_id: int):
        if self.color_model.remove_patch(patch_id, self.current_frame_bgr):
            self._update_patches_table()
            self._reprocess_current_frame()
            self.status_bar.showMessage(f"Đã xóa mẫu #{patch_id}.")

    def _clear_all_samples(self):
        self.color_model.clear_patches()
        self._update_patches_table()
        self._reprocess_current_frame()
        self.status_bar.showMessage("Đã xóa tất cả mẫu màu.")

    def _rebuild_color_model(self):
        if self.current_frame_bgr is None:
            self.status_bar.showMessage("Chưa có khung hình video để học màu.")
            return
        if len(self.color_model.patches) == 0:
            self._sample_from_roi()
            return
        self.color_model.train(self.current_frame_bgr)
        self._reprocess_current_frame()
        self.status_bar.showMessage(f"Đã học lại phân bố màu CIELAB cho {len(self.color_model.patches)} mẫu bạt.")

    def _on_online_update_toggled(self, checked: bool):
        self.color_model.online_update_enabled = checked
        status = "BẬT" if checked else "TẮT"
        self.status_bar.showMessage(f"Cập nhật màu chậm trong lúc chạy: {status}")

    # --- Profile & Export ---

    def _on_save_profile_clicked(self):
        farm_name = self.edit_farm_name.text().strip() or "DefaultFarm"
        default_file = os.path.join(ProfileManager.DEFAULT_PROFILE_DIR, f"{farm_name}.json")
        path, _ = QFileDialog.getSaveFileName(
            self, "Lưu Profile Farm", default_file, "JSON files (*.json)"
        )
        if path:
            yolo_dict = {
                "model_path": self.segmenter.model_path or "",
                "conf_threshold": self.yolo_conf,
                "imgsz": self.yolo_imgsz,
                "device": self.yolo_device
            }
            ok = ProfileManager.save_profile(
                filepath=path,
                farm_name=farm_name,
                detector_dict=self.detector.to_dict(),
                yolo_dict=yolo_dict,
                color_model_dict=self.color_model.to_dict(),
                mode=self.active_mode,
                description=f"Profile nhận diện lối đi nhà màng ({self.active_mode})"
            )
            if ok:
                QMessageBox.information(self, "Thành công", f"Đã lưu profile thành công:\n{path}")
            else:
                QMessageBox.critical(self, "Lỗi", "Không thể lưu profile.")

    def _on_load_profile_clicked(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Nạp Profile Farm", ProfileManager.DEFAULT_PROFILE_DIR, "JSON files (*.json)"
        )
        if path:
            self._explicit_initial_model = None
            self._load_profile_path(path)

    def _load_profile_path(self, path: str):
        data = ProfileManager.load_profile(path)
        if not data:
            return

        self.edit_farm_name.setText(data.get("farm_name", "Farm"))

        # Restore mode if saved
        if "mode" in data:
            saved_mode = data["mode"]
            if saved_mode in (self.MODE_AI, self.MODE_COLOR):
                self.active_mode = saved_mode
                self.combo_mode.setCurrentIndex(0 if saved_mode == self.MODE_AI else 1)

        # Load YOLO settings
        if "yolo" in data:
            y_data = data["yolo"]
            saved_dev = y_data.get("device", self.yolo_device)
            if saved_dev == "cuda" and not is_cuda_available_and_working():
                saved_dev = "cpu"
            self.yolo_device = saved_dev
            device_index = self.combo_device.findData(saved_dev)
            if device_index >= 0:
                was_blocked = self.combo_device.blockSignals(True)
                try:
                    self.combo_device.setCurrentIndex(device_index)
                finally:
                    self.combo_device.blockSignals(was_blocked)
            self.yolo_conf = float(y_data.get("conf_threshold", 0.25))
            self.slider_yolo_conf.setValue(int(self.yolo_conf * 100))
            self.yolo_imgsz = int(y_data.get("imgsz", 320))
            m_path = self._explicit_initial_model or y_data.get("model_path")
            if m_path and os.path.exists(m_path):
                cur_loaded = os.path.abspath(self.segmenter.model_path) if (self.segmenter.is_loaded and self.segmenter.model_path) else ""
                if os.path.abspath(m_path) != cur_loaded:
                    self._load_yolo_model(m_path)
            if self.segmenter.fixed_imgsz is not None:
                self.yolo_imgsz = self.segmenter.fixed_imgsz
            self.segmenter.imgsz = self.yolo_imgsz
            if hasattr(self, "combo_imgsz") and self.combo_imgsz.findText(str(self.yolo_imgsz)) >= 0:
                self.combo_imgsz.blockSignals(True)
                self.combo_imgsz.setCurrentText(str(self.yolo_imgsz))
                self.combo_imgsz.blockSignals(False)
                self.combo_imgsz.setEnabled(self.segmenter.fixed_imgsz is None)

        # Load Color model settings
        if "color_model" in data:
            self.color_model.from_dict(data["color_model"], self.current_frame_bgr)
            self.chk_online_update.setChecked(self.color_model.online_update_enabled)
            self._update_patches_table()

        # Load detector params
        if "detector" in data:
            self.detector.from_dict(data["detector"])
            if hasattr(self, "combo_center_method"):
                self.combo_center_method.blockSignals(True)
                self.combo_center_method.setCurrentIndex(0 if self.detector.center_method == "box" else 1)
                self.combo_center_method.blockSignals(False)
            # Snapshot values and block all sliders: each change handler reads
            # several widgets and would overwrite the remaining loaded values.
            slider_values = [
                (self.slider_roi_top, self.lbl_roi_top, int(self.detector.roi_top_pct * 100), "%"),
                (self.slider_roi_height, self.lbl_roi_height, int(self.detector.roi_height_pct * 100), "%"),
                (self.slider_roi_width, self.lbl_roi_width, int(self.detector.roi_width_pct * 100), "%"),
                (self.slider_roi_center_x, self.lbl_roi_center_x, int(self.detector.roi_center_x_pct * 100), "%"),
                (self.slider_threshold, self.lbl_threshold, self.detector.threshold, ""),
                (self.slider_open, self.lbl_open, self.detector.k_open, "px"),
                (self.slider_close, self.lbl_close, self.detector.k_close, "px"),
                (self.slider_smooth, self.lbl_smooth, int(self.detector.smoothing_factor * 100), "%"),
                (self.slider_scanlines, self.lbl_scanlines, self.detector.num_scanlines, ""),
            ]
            signal_states = [(slider, slider.blockSignals(True)) for slider, _, _, _ in slider_values]
            try:
                for slider, label, value, unit in slider_values:
                    slider.setValue(value)
                    label.setText(f"{slider.value()} {unit}".strip())
            finally:
                for slider, was_blocked in signal_states:
                    slider.blockSignals(was_blocked)

        self._apply_mode_ui_state()
        self.detector.reset_tracking()
        self.segmenter.reset_tracking()
        self._reprocess_current_frame()
        self.status_bar.showMessage(f"Đã nạp profile: {os.path.basename(path)}")

    def _render_full_res_views(
        self,
        frame_bgr: np.ndarray,
        yolo_res: Optional[YoloResult],
        det_res: Optional[DetectionResult],
        score_map: Optional[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Renders full-resolution native images (e.g. 1280x720) for high-precision snapshot export."""
        orig_h, orig_w = frame_bgr.shape[:2]
        cam_cx = int(orig_w / 2)
        is_ai = (self.active_mode == self.MODE_AI)

        if det_res is not None:
            rx, ry, rw, rh = det_res.roi_rect
            roi_mask = det_res.binary_mask
        else:
            rx, ry, rw, rh = self.detector.compute_roi((orig_h, orig_w))
            roi_mask = None

        # 1. View 1: Original + ROI rectangle + Patches (if Color mode)
        v1 = frame_bgr.copy()
        cv2.rectangle(v1, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
        cv2.putText(v1, f"ROI [{rw}x{rh} px]", (rx + 8, max(25, ry + 26)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.line(v1, (cam_cx, ry), (cam_cx, ry + rh), (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(v1, "Moc camera", (max(10, cam_cx - 45), max(20, ry - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        if not is_ai:
            for p in self.color_model.patches:
                px, py, pw, ph = p.rect
                col = (0, 255, 0) if p.category == "path" else (0, 0, 255)
                cv2.rectangle(v1, (px, py), (px + pw, py + ph), col, 2)

        # 2. View 2: Detection Overlay + Bounding Box + Boundaries + Centerline
        v2 = frame_bgr.copy()
        if roi_mask is not None and np.any(roi_mask > 0) and rw > 0 and rh > 0:
            roi_crop = v2[ry:ry + rh, rx:rx + rw]
            tint_color = (210, 220, 20) if is_ai else (0, 220, 220)
            tint = np.full_like(roi_crop, tint_color, dtype=np.uint8)
            blended = cv2.addWeighted(roi_crop, 0.60, tint, 0.40, 0)
            np.copyto(roi_crop, blended, where=(roi_mask > 0)[..., None])

        cv2.rectangle(v2, (rx, ry), (rx + rw, ry + rh), (100, 180, 255), 1)
        cv2.line(v2, (cam_cx, ry), (cam_cx, ry + rh), (240, 240, 240), 1, cv2.LINE_AA)

        # Mode Badge & Device Indicator
        if is_ai and yolo_res is not None and yolo_res.chosen_instance is not None:
            c_name = yolo_res.chosen_instance.class_name
            c_conf = yolo_res.chosen_instance.confidence
            t_ms = yolo_res.inference_time_ms or 0.0
            dev_str = "GPU" if ("cuda" in self.segmenter.effective_device.lower() and not self.segmenter.fallback_reason) else "CPU"
            badge_text = f"{c_name} ({c_conf:.2f}) | {dev_str}: {t_ms:.1f}ms"
            cv2.rectangle(v2, (15, 12), (480, 52), (20, 20, 20), -1)
            cv2.rectangle(v2, (15, 12), (480, 52), (0, 200, 100), 1)
            cv2.putText(v2, badge_text, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2)
        elif not is_ai:
            n_patches = len(self.color_model.patches)
            cv2.rectangle(v2, (15, 12), (420, 52), (20, 20, 20), -1)
            cv2.rectangle(v2, (15, 12), (420, 52), (255, 180, 0), 1)
            cv2.putText(v2, f"CIELAB ({n_patches} mẫu)", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)

        if det_res is not None:
            if det_res.center_method == "path":
                if det_res.centerline_points:
                    pts_arr = np.array(det_res.centerline_points, dtype=np.int32)
                    cv2.polylines(v2, [pts_arr], False, (0, 0, 255), 3, cv2.LINE_AA)

                if det_res.extrapolated_points:
                    draw_dashed_polyline(v2, det_res.extrapolated_points, (255, 0, 255), 2, dash_len=8, gap_len=6)

                if det_res.ref_y_global is not None:
                    ref_s = next((s for s in det_res.scanlines if s.y_global == det_res.ref_y_global and s.is_valid), None)
                    if ref_s and ref_s.x_left is not None and ref_s.x_right is not None:
                        cv2.line(v2, (ref_s.x_left, det_res.ref_y_global), (ref_s.x_right, det_res.ref_y_global), (0, 220, 255), 2)
                        cv2.circle(v2, (ref_s.x_left, det_res.ref_y_global), 5, (0, 140, 255), -1)
                        cv2.circle(v2, (ref_s.x_right, det_res.ref_y_global), 5, (0, 140, 255), -1)
                        cv2.putText(v2, f"Lát cắt Y={det_res.ref_y_global}",
                                    (max(10, ref_s.x_left - 10), det_res.ref_y_global - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1)

                if det_res.path_center is not None:
                    px, py = det_res.path_center
                    cv2.circle(v2, (px, py), 8, (0, 255, 255), -1)
                    cv2.circle(v2, (px, py), 12, (0, 0, 255), 2)
                    cv2.putText(v2, f"Tâm path ({px}, {py})", (min(orig_w - 180, px + 15), py + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                    if det_res.lateral_offset_px is not None:
                        cv2.circle(v2, (cam_cx, py), 5, (255, 255, 255), -1)
                        cv2.arrowedLine(v2, (cam_cx, py), (px, py), (0, 255, 255), 2, tipLength=0.20)
                        lbl_x = max(20, min(orig_w - 240, (cam_cx + px) // 2 - 50))
                        cv2.putText(v2, det_res.offset_direction, (lbl_x, py - 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)

                if det_res.bounding_box is not None:
                    bx, by, bw, bh = det_res.bounding_box
                    cv2.rectangle(v2, (bx, by), (bx + bw, by + bh), (0, 200, 0), 1)

            else:
                if det_res.bounding_box is not None and det_res.box_center is not None:
                    bx, by, bw, bh = det_res.bounding_box
                    cx, cy = det_res.box_center
                    cv2.rectangle(v2, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                    cv2.putText(v2, f"Lối đi [{bw}x{bh} px]", (bx + 8, max(25, by + 22)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                    cv2.line(v2, (cx, by), (cx, by + bh), (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.circle(v2, (cx, cy), 8, (0, 255, 255), -1)
                    cv2.circle(v2, (cx, cy), 12, (0, 0, 255), 2)
                    cv2.putText(v2, f"Tâm box ({cx}, {cy})", (min(orig_w - 180, cx + 15), cy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                    if det_res.lateral_offset_px is not None:
                        cv2.circle(v2, (cam_cx, cy), 5, (255, 255, 255), -1)
                        cv2.arrowedLine(v2, (cam_cx, cy), (cx, cy), (0, 255, 255), 2, tipLength=0.20)
                        lbl_x = max(20, min(orig_w - 240, (cam_cx + cx) // 2 - 50))
                        cv2.putText(v2, det_res.offset_direction, (lbl_x, cy - 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)

            if det_res.state in (TrackingState.LOST, TrackingState.NO_MODEL):
                cv2.rectangle(v2, (orig_w // 2 - 160, 50), (orig_w // 2 + 160, 95), (0, 0, 180), -1)
                cv2.putText(v2, f"[ {det_res.state.value} ]", (orig_w // 2 - 140, 82),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        # 3. View 3: Binary Mask (Clipped strictly inside ROI)
        v3 = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        if roi_mask is not None and np.any(roi_mask > 0) and rw > 0 and rh > 0:
            v3[ry:ry + rh, rx:rx + rw] = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(v3, (rx, ry), (rx + rw, ry + rh), (100, 100, 255), 1)

        # 4. View 4: Debug AI / Heatmap
        if is_ai and yolo_res is not None:
            v4 = cv2.convertScaleAbs(frame_bgr, alpha=0.5)
            inst_colors = [(255, 130, 0), (255, 0, 255), (0, 255, 255), (0, 180, 255), (180, 100, 255)]
            for inst in yolo_res.all_instances:
                col = inst_colors[inst.instance_id % len(inst_colors)]
                mask_i = inst.mask > 0
                if np.any(mask_i):
                    color_tint = np.full_like(v4, col, dtype=np.uint8)
                    blended = cv2.addWeighted(v4, 0.5, color_tint, 0.5, 0)
                    np.copyto(v4, blended, where=mask_i[..., None])

                bx, by, bw, bh = inst.bbox
                is_ch = (inst.instance_id == yolo_res.chosen_index)
                border_col = (0, 255, 0) if is_ch else col
                cv2.rectangle(v4, (bx, by), (bx + bw, by + bh), border_col, 2 if is_ch else 1)
                tag = f"#{inst.instance_id} {inst.class_name} ({inst.confidence:.2f})" + (" [CHOSEN]" if is_ch else "")
                cv2.putText(v4, tag, (bx + 6, max(20, by + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, border_col, 1)

            if det_res is not None:
                target_pt = det_res.path_center if (det_res.center_method == "path" and det_res.path_center is not None) else det_res.box_center
                if target_pt is not None:
                    cv2.circle(v4, (target_pt[0], target_pt[1]), 6, (0, 255, 255), -1)

                method_lbl = "Tâm Path" if det_res.center_method == "path" else "Tâm Box"
                cv2.putText(v4, f"Debug AI [{method_lbl}]: {len(yolo_res.all_instances)} vùng | Chọn: #{yolo_res.chosen_index}",
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        else:
            full_score = np.zeros((orig_h, orig_w), dtype=np.uint8)
            if score_map is not None and score_map.size > 0 and rw > 0 and rh > 0:
                full_score[ry:ry + rh, rx:rx + rw] = score_map
            v4 = cv2.applyColorMap(full_score, cv2.COLORMAP_JET)
            cv2.rectangle(v4, (rx, ry), (rx + rw, ry + rh), (255, 255, 255), 1)
            cv2.putText(v4, "Score Map CIELAB", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

        return {"view1": v1, "view2": v2, "view3": v3, "view4": v4}

    def _on_export_snapshot_clicked(self):
        if self.current_frame_bgr is None:
            QMessageBox.information(self, "Thông báo", "Chưa có frame nào để xuất.")
            return

        out_dir = os.path.join(PROJECT_ROOT, "output")
        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time())
        f_idx = self.reader.current_frame_idx

        # Render full-resolution native 1280x720 views
        full_views = self._render_full_res_views(
            self.current_frame_bgr,
            self.last_yolo_result,
            self.last_detection_result,
            self.last_score_map
        )

        v1_path = os.path.join(out_dir, f"frame_{f_idx:04d}_{ts}_orig.jpg")
        v2_path = os.path.join(out_dir, f"frame_{f_idx:04d}_{ts}_overlay.jpg")
        v3_path = os.path.join(out_dir, f"frame_{f_idx:04d}_{ts}_mask.jpg")
        v4_name = "debug" if self.active_mode == self.MODE_AI else "heatmap"
        v4_path = os.path.join(out_dir, f"frame_{f_idx:04d}_{ts}_{v4_name}.jpg")

        cv2.imwrite(v1_path, full_views["view1"])
        cv2.imwrite(v2_path, full_views["view2"])
        cv2.imwrite(v3_path, full_views["view3"])
        cv2.imwrite(v4_path, full_views["view4"])

        # Export Telemetry JSON
        telemetry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frame_index": f_idx,
            "video_timestamp_sec": self.reader.get_timestamp_sec(),
            "mode": self.active_mode,
            "state": self.last_detection_result.state.value if self.last_detection_result else "N/A",
            "path_confidence_pct": self.last_detection_result.confidence if self.last_detection_result else 0.0,
            "lateral_offset_px": self.last_detection_result.lateral_offset_px if self.last_detection_result else None,
            "offset_direction": self.last_detection_result.offset_direction if self.last_detection_result else "N/A",
            "ref_y_global": self.last_detection_result.ref_y_global if self.last_detection_result else None,
            "bounding_box": list(self.last_detection_result.bounding_box) if (self.last_detection_result and self.last_detection_result.bounding_box) else None,
            "box_center": list(self.last_detection_result.box_center) if (self.last_detection_result and self.last_detection_result.box_center) else None,
            "valid_scanlines": self.last_detection_result.valid_scanlines_count if self.last_detection_result else 0,
            "total_scanlines": self.last_detection_result.total_scanlines_count if self.last_detection_result else 0,
            "image_resolution": [self.current_frame_bgr.shape[1], self.current_frame_bgr.shape[0]],
            "device_used": self.segmenter.effective_device if self.active_mode == self.MODE_AI else "CPU",
            "device_name": self.segmenter.device_display_name if self.active_mode == self.MODE_AI else "CPU (CIELAB)",
            "fallback_reason": self.segmenter.fallback_reason if self.active_mode == self.MODE_AI else None
        }
        if self.active_mode == self.MODE_AI:
            telemetry["yolo_conf"] = self.yolo_conf
            telemetry["yolo_imgsz"] = self.yolo_imgsz
            telemetry["yolo_instance_confidence"] = self.last_yolo_result.chosen_confidence if self.last_yolo_result else None
            telemetry["yolo_inference_ms"] = self.last_yolo_result.inference_time_ms if self.last_yolo_result else None
        else:
            telemetry["color_patches_count"] = len(self.color_model.patches)
            telemetry["score_threshold"] = self.detector.threshold

        json_path = os.path.join(out_dir, f"frame_{f_idx:04d}_{ts}_telemetry.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(telemetry, f, indent=2, ensure_ascii=False)

        QMessageBox.information(
            self,
            "Xuất thành công",
            f"Đã lưu 4 view ảnh độ phân giải gốc ({self.current_frame_bgr.shape[1]}x{self.current_frame_bgr.shape[0]}) và telemetry JSON ({self.active_mode}) vào:\n{out_dir}\n- {os.path.basename(json_path)}"
        )
        self.status_bar.showMessage(f"Đã xuất snapshot frame {f_idx} độ phân giải gốc vào output/.")

    def closeEvent(self, event: QtGui.QCloseEvent):
        self._stop_worker()
        self.reader.close()
        if hasattr(self, "uart") and self.uart:
            self.uart.disconnect()
        super().closeEvent(event)
