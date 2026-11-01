"""
Module for YOLO segmentation inference on agricultural greenhouse path images.
Loads Ultralytics YOLO segmentation models, manages inference configuration (imgsz, conf, device),
and selects the primary path instance among multiple detections based on spatial heuristics.
"""

from dataclasses import dataclass, field
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class InstanceDetection:
    """Represents a single detected instance from YOLO."""
    instance_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in original frame pixels
    mask: np.ndarray                 # 2D uint8 binary mask (0 or 255) in original frame dimensions
    area_px: int
    centroid: Tuple[int, int]        # (cx, cy)
    bottom_y: int


@dataclass
class YoloResult:
    """Structured output from YOLO segmentation inference."""
    is_valid: bool
    error_message: str = ""
    chosen_mask: Optional[np.ndarray] = None  # Full-frame 2D uint8 (0 or 255) of chosen path instance
    chosen_instance: Optional[InstanceDetection] = None
    all_instances: List[InstanceDetection] = field(default_factory=list)
    chosen_index: int = -1
    chosen_confidence: Optional[float] = None
    inference_time_ms: float = 0.0
    model_name: str = ""
    class_names: Dict[int, str] = field(default_factory=dict)
    frame_shape: Tuple[int, int] = (0, 0)  # (H, W)
    device_used: str = ""
    fallback_reason: str = ""


def get_cpu_info() -> str:
    """Returns a readable CPU model identifier string."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    import platform
    return platform.processor() or "CPU"


@dataclass
class CudaDeviceInfo:
    """Diagnostic details regarding PyTorch CUDA availability and health."""
    is_available: bool
    device_name: str = ""
    error_reason: str = ""
    vram_free_mb: float = 0.0
    vram_total_mb: float = 0.0


def get_cuda_status() -> CudaDeviceInfo:
    """Verifies that PyTorch CUDA is not only detected, but actually working without driver/uvm errors."""
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return CudaDeviceInfo(
                is_available=False,
                error_reason="torch.cuda.is_available() trả về False (Chưa nạp driver CUDA hoặc không có GPU)."
            )
        # Test real tensor allocation on device to avoid runtime EIO errors on /dev/nvidia-uvm
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        dev_name = torch.cuda.get_device_name(0)
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            free_mb = free_b / (1024 * 1024)
            total_mb = total_b / (1024 * 1024)
        except Exception:
            free_mb, total_mb = 0.0, 0.0
        return CudaDeviceInfo(
            is_available=True,
            device_name=dev_name,
            vram_free_mb=free_mb,
            vram_total_mb=total_mb
        )
    except Exception as e:
        err_msg = str(e)
        if "unknown error" in err_msg.lower() or "uvm" in err_msg.lower() or "[errno 5]" in err_msg.lower():
            err_msg += " (NVIDIA driver UVM MMU fault /dev/nvidia-uvm EIO; cần reboot OS để driver GPU phục hồi)"
        return CudaDeviceInfo(
            is_available=False,
            error_reason=f"Lỗi khởi tạo CUDA: {err_msg}"
        )


def is_cuda_available_and_working() -> bool:
    """Verifies that PyTorch CUDA is operational without runtime errors."""
    return get_cuda_status().is_available


class YoloSegmenter:
    """Manages YOLO segmentation model lifecycle, inference, and path instance filtering."""

    DEFAULT_MODEL_REL_PATH = "models/farm_path_bootstrap.pt"
    EXPECTED_CLASS_NAME = "bat_loi_di"

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        imgsz: int = 416,
        device: Optional[str] = None
    ):
        self.requested_device: str = device or "auto"
        self.effective_device: str = "cpu"
        self.device_display_name: str = "CPU"
        self.fallback_reason: str = ""
        self.device: str = "cpu"
        self._resolve_device(self.requested_device)
        try:
            import torch
            # Giới hạn số luồng PyTorch trên CPU (mặc định 2 luồng để chống sụt áp nguồn Pi 5)
            threads = int(os.environ.get("TORCH_NUM_THREADS", "2"))
            torch.set_num_threads(threads)
        except Exception:
            pass

        self.conf_threshold: float = conf_threshold
        self.imgsz: int = imgsz
        self.model_path: Optional[str] = None
        self.model: Any = None
        self.class_names: Dict[int, str] = {}
        self.is_loaded: bool = False
        self.load_error: str = ""

        # Tracking memory across frames for stable instance association
        self.prev_chosen_centroid: Optional[Tuple[int, int]] = None

        if model_path:
            self.load_model(model_path, device=self.requested_device)

    def _resolve_device(self, device_req: str):
        """Resolves requested device into an effective device with explicit fallbacks and diagnostics."""
        self.requested_device = device_req
        req_lower = (device_req or "").lower().strip()
        if req_lower == "cpu":
            self.effective_device = "cpu"
            self.device_display_name = f"CPU ({get_cpu_info()})"
            self.fallback_reason = ""
        elif req_lower in ("cuda", "auto") or "cuda" in req_lower:
            cuda_stat = get_cuda_status()
            if cuda_stat.is_available:
                self.effective_device = "cuda"
                self.device_display_name = f"CUDA:0 ({cuda_stat.device_name})"
                self.fallback_reason = ""
            else:
                self.effective_device = "cpu"
                self.device_display_name = f"CPU ({get_cpu_info()})"
                self.fallback_reason = cuda_stat.error_reason
                print(f"[YoloSegmenter] {device_req} không khả dụng: {cuda_stat.error_reason}. Chuyển sang CPU.")
        else:
            self.effective_device = "cpu"
            self.device_display_name = f"CPU ({get_cpu_info()})"
            self.fallback_reason = ""
        self.device = self.effective_device

    def load_model(self, model_path: str, device: Optional[str] = None) -> bool:
        """Loads and validates a YOLO segmentation model."""
        if device is not None:
            self._resolve_device(device)

        self.is_loaded = False
        self.load_error = ""
        self.model = None
        self.class_names = {}
        self.model_path = model_path

        if not os.path.exists(model_path):
            self.load_error = f"File model không tồn tại: {model_path}"
            return False

        try:
            from ultralytics import YOLO
        except ImportError as e:
            self.load_error = f"Không tìm thấy thư viện ultralytics: {e}"
            return False

        try:
            loaded_model = YOLO(model_path)
            # Validate model task
            task = getattr(loaded_model, "task", None)
            if not task and hasattr(loaded_model, "overrides"):
                task = loaded_model.overrides.get("task")

            if task != "segment":
                self.load_error = (
                    f"Model không phải là mô hình segmentation (task='{task}'). "
                    f"Vui lòng cung cấp model YOLO segmentation (.pt)!"
                )
                return False

            # Validate classes
            names = getattr(loaded_model, "names", {})
            if isinstance(names, list):
                names = {i: name for i, name in enumerate(names)}
            self.class_names = names

            self.model = loaded_model
            self.is_loaded = True
            self.prev_chosen_centroid = None
            return True
        except Exception as e:
            self.load_error = f"Lỗi nạp model '{os.path.basename(model_path)}': {e}"
            self.model = None
            self.is_loaded = False
            return False

    def reset_tracking(self):
        """Resets temporal instance association memory (e.g. on seek)."""
        self.prev_chosen_centroid = None

    def predict(
        self,
        frame_bgr: np.ndarray,
        conf: Optional[float] = None,
        imgsz: Optional[int] = None,
        roi_rect: Optional[Tuple[int, int, int, int]] = None
    ) -> YoloResult:
        """Executes YOLO segmentation inference on the input frame.
        
        Args:
            frame_bgr: Full BGR frame (H, W, 3).
            conf: Override confidence threshold if provided.
            imgsz: Override input image size if provided.
            roi_rect: Optional (rx, ry, rw, rh) to guide path instance selection.
            
        Returns:
            YoloResult containing the chosen mask, instance metrics, and debug data.
        """
        if not self.is_loaded or self.model is None:
            return YoloResult(
                is_valid=False,
                error_message=self.load_error or "Model chưa được nạp.",
                frame_shape=frame_bgr.shape[:2] if frame_bgr is not None else (0, 0)
            )

        if frame_bgr is None or frame_bgr.size == 0:
            return YoloResult(
                is_valid=False,
                error_message="Khung hình rỗng.",
                frame_shape=(0, 0)
            )

        h_img, w_img = frame_bgr.shape[:2]
        effective_conf = conf if conf is not None else self.conf_threshold
        effective_imgsz = imgsz if imgsz is not None else self.imgsz

        t_start = time.perf_counter()
        device_used = self.device_display_name
        fallback_reason = self.fallback_reason

        try:
            # Predict on full frame with retina_masks=True to ensure exact full-frame mask alignment
            results = self.model.predict(
                source=frame_bgr,
                imgsz=effective_imgsz,
                conf=effective_conf,
                device=self.effective_device,
                retina_masks=True,
                verbose=False
            )
            inference_time_ms = (time.perf_counter() - t_start) * 1000.0
        except Exception as e:
            err_str = str(e).lower()
            if self.effective_device != "cpu" and ("cuda" in err_str or "device" in err_str or "driver" in err_str or "illegal" in err_str or "uvm" in err_str):
                fallback_reason = f"Lỗi runtime {self.effective_device}: {e}. Đã tự động fallback về CPU."
                print(f"[YoloSegmenter] {fallback_reason}")
                self.effective_device = "cpu"
                self.device = "cpu"
                self.device_display_name = f"CPU ({get_cpu_info()})"
                self.fallback_reason = fallback_reason
                device_used = self.device_display_name
                try:
                    results = self.model.predict(
                        source=frame_bgr,
                        imgsz=effective_imgsz,
                        conf=effective_conf,
                        device="cpu",
                        retina_masks=True,
                        verbose=False
                    )
                    inference_time_ms = (time.perf_counter() - t_start) * 1000.0
                except Exception as e_cpu:
                    return YoloResult(
                        is_valid=False,
                        error_message=f"Lỗi suy luận YOLO (CPU fallback): {e_cpu}",
                        inference_time_ms=(time.perf_counter() - t_start) * 1000.0,
                        frame_shape=(h_img, w_img),
                        device_used=device_used,
                        fallback_reason=fallback_reason
                    )
            else:
                return YoloResult(
                    is_valid=False,
                    error_message=f"Lỗi suy luận YOLO: {e}",
                    inference_time_ms=(time.perf_counter() - t_start) * 1000.0,
                    frame_shape=(h_img, w_img),
                    device_used=device_used,
                    fallback_reason=fallback_reason
                )

        if not results or len(results) == 0:
            return YoloResult(
                is_valid=True,
                chosen_mask=np.zeros((h_img, w_img), dtype=np.uint8),
                inference_time_ms=inference_time_ms,
                model_name=os.path.basename(self.model_path or ""),
                class_names=self.class_names,
                frame_shape=(h_img, w_img),
                device_used=device_used,
                fallback_reason=fallback_reason
            )

        r = results[0]
        boxes = r.boxes
        masks = r.masks

        if boxes is None or masks is None or len(boxes) == 0 or len(masks) == 0:
            return YoloResult(
                is_valid=True,
                chosen_mask=np.zeros((h_img, w_img), dtype=np.uint8),
                inference_time_ms=inference_time_ms,
                model_name=os.path.basename(self.model_path or ""),
                class_names=self.class_names,
                frame_shape=(h_img, w_img),
                device_used=device_used,
                fallback_reason=fallback_reason
            )

        # Extract all detected instances
        instances: List[InstanceDetection] = []
        mask_data = masks.data.cpu().numpy()  # (N, H, W) or (N, H_mask, W_mask)
        num_instances = len(boxes)

        for i in range(num_instances):
            cls_id = int(boxes.cls[i].item())
            cls_name = self.class_names.get(cls_id, str(cls_id))
            c_score = float(boxes.conf[i].item())
            b_coords = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = b_coords
            x1 = int(np.clip(x1, 0, w_img - 1))
            y1 = int(np.clip(y1, 0, h_img - 1))
            x2 = int(np.clip(x2, x1 + 1, w_img))
            y2 = int(np.clip(y2, y1 + 1, h_img))

            # Retrieve mask and ensure full-frame dimensions (h_img, w_img)
            inst_mask = (mask_data[i] > 0.5).astype(np.uint8) * 255
            if inst_mask.shape != (h_img, w_img):
                # Letterbox-safe resize if retina_masks wasn't exact
                inst_mask = cv2.resize(inst_mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

            area_px = int(np.count_nonzero(inst_mask))
            if area_px == 0:
                continue

            # Centroid computation via moments
            M = cv2.moments(inst_mask)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

            instances.append(InstanceDetection(
                instance_id=i,
                class_id=cls_id,
                class_name=cls_name,
                confidence=c_score,
                bbox=(x1, y1, x2 - x1, y2 - y1),
                mask=inst_mask,
                area_px=area_px,
                centroid=(cx, cy),
                bottom_y=y2
            ))

        if not instances:
            return YoloResult(
                is_valid=True,
                chosen_mask=np.zeros((h_img, w_img), dtype=np.uint8),
                inference_time_ms=inference_time_ms,
                model_name=os.path.basename(self.model_path or ""),
                class_names=self.class_names,
                frame_shape=(h_img, w_img),
                device_used=device_used,
                fallback_reason=fallback_reason
            )

        # Select the single instance that best represents the corridor path in front of the robot
        chosen_idx, chosen_inst = self._select_best_path_instance(
            instances, (h_img, w_img), roi_rect
        )

        if chosen_inst is not None:
            self.prev_chosen_centroid = chosen_inst.centroid
            chosen_mask = chosen_inst.mask.copy()
            chosen_conf = chosen_inst.confidence
        else:
            chosen_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            chosen_conf = None

        return YoloResult(
            is_valid=True,
            chosen_mask=chosen_mask,
            chosen_instance=chosen_inst,
            all_instances=instances,
            chosen_index=chosen_idx,
            chosen_confidence=chosen_conf,
            inference_time_ms=inference_time_ms,
            model_name=os.path.basename(self.model_path or ""),
            class_names=self.class_names,
            frame_shape=(h_img, w_img),
            device_used=device_used,
            fallback_reason=fallback_reason
        )

    def _select_best_path_instance(
        self,
        instances: List[InstanceDetection],
        frame_shape: Tuple[int, int],
        roi_rect: Optional[Tuple[int, int, int, int]]
    ) -> Tuple[int, Optional[InstanceDetection]]:
        """Selects the path corridor instance based on spatial proximity, area, and tracking continuity."""
        if len(instances) == 1:
            return 0, instances[0]

        h_img, w_img = frame_shape
        if roi_rect is not None:
            rx, ry, rw, rh = roi_rect
            roi_cx = rx + rw / 2.0
            roi_bottom = ry + rh
        else:
            roi_cx = w_img / 2.0
            roi_bottom = h_img

        best_score = -1e9
        best_idx = 0
        best_instance = instances[0]

        for idx, inst in enumerate(instances):
            cx, cy = inst.centroid
            # 1. Proximity to horizontal center (corridor path lies directly in front)
            dist_center = abs(cx - roi_cx) / (w_img / 2.0)
            center_score = max(0.0, 1.0 - dist_center)

            # 2. Bottom extension (path in front extends to vehicle's near foreground)
            bottom_reach = min(1.0, inst.bottom_y / float(roi_bottom))

            # 3. Size / Coverage in lower half of image
            area_norm = min(1.0, inst.area_px / float(h_img * w_img * 0.4))

            # 4. Temporal consistency with previous frame's centroid
            continuity_score = 0.5
            if self.prev_chosen_centroid is not None:
                prev_cx, prev_cy = self.prev_chosen_centroid
                dist_prev = np.hypot(cx - prev_cx, cy - prev_cy) / float(w_img)
                continuity_score = max(0.0, 1.0 - dist_prev * 2.0)

            # Composite heuristic score:
            # Spatial positioning in foreground dominates, with confidence and continuity reinforcing
            score = (
                4.0 * bottom_reach +
                3.0 * center_score +
                2.0 * area_norm +
                1.5 * continuity_score +
                1.0 * inst.confidence
            )

            if score > best_score:
                best_score = score
                best_idx = idx
                best_instance = inst

        return best_idx, best_instance

