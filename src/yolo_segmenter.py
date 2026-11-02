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
        self.is_unet_int8: bool = False
        self._resolve_device(self.requested_device)
        try:
            import torch
            # Giới hạn số luồng PyTorch trên CPU (mặc định 3 luồng để tối ưu tốc độ và chừa 1 core)
            threads = int(os.environ.get("TORCH_NUM_THREADS", "3"))
            torch.set_num_threads(threads)
        except Exception:
            pass

        self.conf_threshold: float = conf_threshold
        self.imgsz: int = imgsz
        self.fixed_imgsz: Optional[int] = None
        self.is_ncnn: bool = False
        self.ncnn_net: Any = None
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
        """Loads and validates a YOLO segmentation model (.pt or NCNN quantized directory)."""
        if device is not None:
            self._resolve_device(device)

        self.is_loaded = False
        self.load_error = ""
        self.model = None
        self.class_names = {}
        self.fixed_imgsz = None
        self.is_ncnn = False
        self.ncnn_net = None

        # Normalize if user picked a file inside an NCNN model directory
        if model_path and (model_path.endswith(".param") or model_path.endswith(".bin") or os.path.basename(model_path) == "metadata.yaml"):
            model_path = os.path.dirname(model_path)
        self.model_path = model_path
        self.is_unet_int8 = False

        if not os.path.exists(model_path):
            self.load_error = f"File/thư mục model không tồn tại: {model_path}"
            return False

        # Support direct MobileNetV3-UNet INT8 NCNN models (.param / .bin)
        if "model_int8" in model_path.lower() or model_path.endswith(".param") or (os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "model_int8.param"))):
            try:
                import ncnn
                self.is_unet_int8 = True
                self.is_ncnn = True
                param_path = model_path if model_path.endswith(".param") else (
                    os.path.join(model_path, "model_int8.param") if os.path.isdir(model_path) else (
                        model_path if os.path.isfile(model_path) else model_path + ".param"
                    )
                )
                bin_path = param_path.rsplit(".", 1)[0] + ".bin"
                net = ncnn.Net()
                net.opt.use_vulkan_compute = False
                n_threads = int(os.environ.get("OMP_NUM_THREADS", "3"))
                net.opt.num_threads = n_threads
                net.opt.use_int8_inference = True
                if net.load_param(param_path) == 0 and net.load_model(bin_path) == 0:
                    self.ncnn_net = net
                    self.model = net
                    self.is_loaded = True
                    self.fixed_imgsz = 320
                    self.imgsz = 320
                    self.device = "cpu"
                    self.effective_device = "cpu"
                    self.device_display_name = f"UNet INT8 NCNN ({get_cpu_info()})"
                    self.prev_chosen_centroid = None
                    return True
                else:
                    self.load_error = f"Không thể nạp file NCNN INT8: {param_path}"
                    return False
            except Exception as e_unet:
                self.load_error = f"Lỗi khởi tạo NCNN INT8: {e_unet}"
                return False

        try:
            from ultralytics import YOLO
        except ImportError as e:
            self.load_error = f"Không tìm thấy thư viện ultralytics: {e}"
            return False

        try:
            if os.path.isdir(model_path) or "ncnn" in model_path.lower():
                self.is_ncnn = True
                meta_file = os.path.join(model_path, "metadata.yaml")
                if os.path.exists(meta_file):
                    try:
                        import yaml
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = yaml.safe_load(f) or {}
                        imgsz_meta = meta.get("imgsz")
                        if isinstance(imgsz_meta, (list, tuple)) and len(imgsz_meta) > 0:
                            self.fixed_imgsz = int(imgsz_meta[0])
                            self.imgsz = self.fixed_imgsz
                        elif isinstance(imgsz_meta, int):
                            self.fixed_imgsz = imgsz_meta
                            self.imgsz = self.fixed_imgsz
                    except Exception:
                        pass
                if self.fixed_imgsz is None:
                    self.fixed_imgsz = 320
                    self.imgsz = 320
                self.effective_device = "cpu"
                self.device = "cpu"
                self.device_display_name = f"NCNN Quantized ({get_cpu_info()})"
                self.fallback_reason = ""

                # Try loading pure NCNN C++ net for zero-overhead ARM NEON inference
                try:
                    import ncnn
                    param_file = os.path.join(model_path, "model.ncnn.param")
                    bin_file = os.path.join(model_path, "model.ncnn.bin")
                    if os.path.exists(param_file) and os.path.exists(bin_file):
                        net = ncnn.Net()
                        net.opt.use_vulkan_compute = False
                        net.opt.num_threads = 1  # 1 thread optimal for Pi 5 L2 cache locality
                        net.opt.use_fp16_packed = True
                        net.opt.use_fp16_storage = True
                        net.opt.use_fp16_arithmetic = True
                        if net.load_param(param_file) == 0 and net.load_model(bin_file) == 0:
                            self.ncnn_net = net
                            self.device_display_name = f"NCNN Fast C++ 1-Thread ({get_cpu_info()})"
                except Exception as e_ncnn:
                    self.ncnn_net = None

            loaded_model = YOLO(model_path, task="segment")
            if self.is_ncnn:
                # Warm up NCNN AutoBackend once during load so the first video frame has zero lag
                dummy_sz = self.fixed_imgsz or 320
                dummy_img = np.zeros((dummy_sz, dummy_sz, 3), dtype=np.uint8)
                loaded_model.predict(
                    source=dummy_img,
                    imgsz=dummy_sz,
                    device="cpu",
                    verbose=False,
                    retina_masks=False
                )
            # Validate model task
            task = getattr(loaded_model, "task", None)
            if not task and hasattr(loaded_model, "overrides"):
                task = loaded_model.overrides.get("task")

            if task != "segment":
                self.load_error = (
                    f"Model không phải là mô hình segmentation (task='{task}'). "
                    f"Vui lòng cung cấp model YOLO segmentation (.pt hoặc NCNN)!"
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
        effective_imgsz = self.fixed_imgsz if self.fixed_imgsz is not None else (imgsz if imgsz is not None else self.imgsz)

        t_start = time.perf_counter()
        device_used = self.device_display_name
        fallback_reason = self.fallback_reason

        # === 0. UNET INT8 NCNN FAST SEMANTIC SEGMENTATION ===
        if getattr(self, "is_unet_int8", False) and self.ncnn_net is not None:
            try:
                import ncnn
                eff_sz = 320
                resized_bgr = cv2.resize(frame_bgr, (eff_sz, eff_sz), interpolation=cv2.INTER_LINEAR)
                rgb_contiguous = np.ascontiguousarray(cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB))
                in_mat = ncnn.Mat.from_pixels(rgb_contiguous, ncnn.Mat.PixelType.PIXEL_RGB, eff_sz, eff_sz)
                in_mat.substract_mean_normalize([0.0, 0.0, 0.0], [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0])

                ex = self.ncnn_net.create_extractor()
                ex.input("in0", in_mat)
                ret, out_mat = ex.extract("out0")
                prob_map = np.array(out_mat, dtype=np.float32)
                if prob_map.ndim == 3:
                    prob_map = prob_map[0]

                thresh = max(float(effective_conf or 0.62), 0.62)
                mask_320 = (prob_map >= thresh).astype(np.uint8) * 255
                mask_orig = cv2.resize(mask_320, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                mask_clean = cv2.morphologyEx(mask_orig, cv2.MORPH_OPEN, kernel)

                all_instances = []
                chosen_instance = None
                chosen_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    largest_cnt = max(contours, key=cv2.contourArea)
                    area = int(cv2.contourArea(largest_cnt))
                    if area > 300:
                        x, y, w, h = cv2.boundingRect(largest_cnt)
                        M = cv2.moments(largest_cnt)
                        cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else (x + w // 2)
                        cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else (y + h // 2)
                        inst_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                        cv2.drawContours(inst_mask, [largest_cnt], -1, 255, -1)
                        chosen_mask = inst_mask

                        chosen_instance = InstanceDetection(
                            instance_id=0,
                            class_id=0,
                            class_name="bat_loi_di",
                            confidence=float(np.mean(prob_map[mask_320 > 0])) if (mask_320 > 0).any() else 0.95,
                            bbox=(x, y, x + w, y + h),
                            mask=inst_mask,
                            area_px=area,
                            centroid=(cx, cy),
                            bottom_y=y + h
                        )
                        all_instances.append(chosen_instance)

                inference_time_ms = (time.perf_counter() - t_start) * 1000.0
                return YoloResult(
                    is_valid=True,
                    chosen_mask=chosen_mask,
                    chosen_instance=chosen_instance,
                    all_instances=all_instances,
                    chosen_index=0 if chosen_instance else -1,
                    chosen_confidence=chosen_instance.confidence if chosen_instance else None,
                    inference_time_ms=inference_time_ms,
                    model_name=os.path.basename(self.model_path or "model_int8"),
                    class_names={0: "bat_loi_di"},
                    frame_shape=(h_img, w_img),
                    device_used=self.device_display_name,
                    fallback_reason=""
                )
            except Exception as e_inf:
                return YoloResult(
                    is_valid=False,
                    error_message=f"Lỗi suy luận UNet INT8: {e_inf}",
                    frame_shape=(h_img, w_img)
                )

        # === 1. ULTRA-FAST DIRECT NCNN C++ EXECUTION (ARM NEON ZERO-OVERHEAD) ===
        if self.ncnn_net is not None:
            try:
                import ncnn
                eff_sz = int(effective_imgsz or 256)
                img_resized = cv2.resize(frame_bgr, (eff_sz, eff_sz))
                in_mat = ncnn.Mat.from_pixels(img_resized, ncnn.Mat.PixelType.PIXEL_BGR2RGB, eff_sz, eff_sz)
                in_mat.substract_mean_normalize([], [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0])

                ex = self.ncnn_net.create_extractor()
                ex.input("in0", in_mat)
                out0, out1 = ncnn.Mat(), ncnn.Mat()
                ex.extract("out0", out0)
                ex.extract("out1", out1)

                arr_out0 = np.array(out0)
                arr_out1 = np.array(out1)
                c0 = arr_out0.shape[0] if arr_out0.ndim >= 2 else 37
                c1 = arr_out1.shape[0] if arr_out1.ndim >= 2 else 32
                pred = arr_out0.reshape((c0, -1)) # (c0, num_anchors)
                protos = arr_out1.reshape((c1, -1)) # (num_protos, proto_pixels)

                scores = pred[4, :] # class 0 (bat_loi_di)
                best_idx = int(np.argmax(scores))
                best_score = float(scores[best_idx])
                inference_time_ms = (time.perf_counter() - t_start) * 1000.0

                if best_score >= effective_conf:
                    mask_coeffs = pred[5:5 + c1, best_idx]
                    proto_sz = int(np.sqrt(protos.shape[1]))
                    mask_raw = np.dot(mask_coeffs, protos).reshape((proto_sz, proto_sz))
                    small_mask = (mask_raw > 0).astype(np.uint8) * 255
                    inst_mask = cv2.resize(small_mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

                    if roi_rect is not None:
                        rx, ry, rw, rh = roi_rect
                        roi_m = np.zeros_like(inst_mask)
                        rx_e = min(w_img, rx + rw)
                        ry_e = min(h_img, ry + rh)
                        roi_m[ry:ry_e, rx:rx_e] = inst_mask[ry:ry_e, rx:rx_e]
                        inst_mask = roi_m

                    M = cv2.moments(inst_mask)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        area_px = int(M["m00"] / 255.0)
                    else:
                        cx, cy, area_px = w_img // 2, h_img // 2, 0

                    chosen_inst = InstanceDetection(
                        instance_id=0,
                        class_id=0,
                        class_name="bat_loi_di",
                        confidence=best_score,
                        bbox=(0, 0, w_img, h_img),
                        mask=inst_mask,
                        area_px=area_px,
                        centroid=(cx, cy),
                        bottom_y=h_img
                    )
                    return YoloResult(
                        is_valid=True,
                        chosen_mask=inst_mask,
                        chosen_instance=chosen_inst,
                        all_instances=[chosen_inst],
                        chosen_index=0,
                        chosen_confidence=best_score,
                        inference_time_ms=inference_time_ms,
                        model_name=os.path.basename(self.model_path or ""),
                        class_names={0: "bat_loi_di"},
                        frame_shape=(h_img, w_img),
                        device_used=device_used,
                        fallback_reason=fallback_reason
                    )
                else:
                    return YoloResult(
                        is_valid=True,
                        chosen_mask=np.zeros((h_img, w_img), dtype=np.uint8),
                        chosen_confidence=best_score,
                        inference_time_ms=inference_time_ms,
                        model_name=os.path.basename(self.model_path or ""),
                        class_names={0: "bat_loi_di"},
                        frame_shape=(h_img, w_img),
                        device_used=device_used,
                        fallback_reason=fallback_reason
                    )
            except Exception as e_direct:
                # If direct fast NCNN encounters unexpected shapes, fallback cleanly to standard predict
                pass

        try:
            # Predict with retina_masks=False + fast letterbox unpadding to avoid 118MB full-res tensor ops on CPU
            results = self.model.predict(
                source=frame_bgr,
                imgsz=effective_imgsz,
                conf=effective_conf,
                device=self.effective_device,
                retina_masks=False,
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
                        retina_masks=False,
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
        mask_data = masks.data.cpu().numpy()  # (N, H_mask, W_mask)
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

            raw_m = mask_data[i]
            mh, mw = raw_m.shape[:2]
            if (mh, mw) != (h_img, w_img):
                gain = min(mw / float(w_img), mh / float(h_img))
                pad_x = int(round((mw - w_img * gain) / 2.0 - 0.1))
                pad_y = int(round((mh - h_img * gain) / 2.0 - 0.1))
                y0 = max(0, pad_y)
                y1_m = max(y0 + 1, mh - pad_y) if pad_y > 0 else mh
                x0 = max(0, pad_x)
                x1_m = max(x0 + 1, mw - pad_x) if pad_x > 0 else mw
                small_mask = (raw_m[y0:y1_m, x0:x1_m] > 0.5).astype(np.uint8) * 255
                sh, sw = small_mask.shape[:2]
                M = cv2.moments(small_mask)
                if M["m00"] <= 0:
                    continue
                cx = int((M["m10"] / M["m00"]) * (w_img / float(sw)))
                cy = int((M["m01"] / M["m00"]) * (h_img / float(sh)))
                area_px = max(1, int((M["m00"] / 255.0) * ((w_img * h_img) / float(sw * sh))))
                inst_mask = cv2.resize(small_mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
            else:
                inst_mask = (raw_m > 0.5).astype(np.uint8) * 255
                M = cv2.moments(inst_mask)
                if M["m00"] <= 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                area_px = int(M["m00"] / 255.0)

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

