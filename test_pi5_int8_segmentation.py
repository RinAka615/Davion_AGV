#!/usr/bin/env python3
"""
Chương trình Inference Semantic Segmentation (MobileNetV3-UNet INT8 NCNN) trên Raspberry Pi 5 CPU.
- Kiến trúc: MobileNetV3-Small U-Net (320x320, 1 class: "đường đi màu đen")
- Lượng tử hóa tĩnh: NCNN INT8 (KL-Divergence Calibration)
- Đầu ra: Mask phân vùng nhị phân + Tọa độ Trọng tâm (cx, cy) & Độ lệch lái (error_px)
"""

import os
import sys
import time
from typing import Optional, Tuple

import cv2
import ncnn
import numpy as np


class BlackPathSegmenterINT8:
    def __init__(
        self,
        param_path: str = "models/model_int8.param",
        bin_path: str = "models/model_int8.bin",
        input_size: Tuple[int, int] = (320, 320),
        num_threads: int = 2,
        prob_threshold: float = 0.5,
    ):
        assert os.path.isfile(param_path), f"Không tìm thấy file cấu trúc INT8: {param_path}"
        assert os.path.isfile(bin_path), f"Không tìm thấy file trọng số INT8: {bin_path}"

        self.input_w, self.input_h = input_size
        assert self.input_w <= 320 and self.input_h <= 320, "Kích thước đầu vào vượt quá giới hạn 320x320!"
        self.prob_threshold = prob_threshold
        self._debug_logged = False

        # Khởi tạo NCNN INT8 Engine cho CPU ARM Cortex-A76 (Raspberry Pi 5)
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = num_threads
        self.net.opt.use_int8_inference = True

        ret_param = self.net.load_param(param_path)
        ret_bin = self.net.load_model(bin_path)
        assert ret_param == 0 and ret_bin == 0, "Lỗi nạp file model_int8.param / model_int8.bin vào NCNN!"

        self.mean_vals = [0.0, 0.0, 0.0]
        self.norm_vals = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]

    def infer(
        self, frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, Optional[Tuple[int, int]], float, float]:
        """
        Thực thi phân vùng vạch đen và tính tọa độ trọng tâm.
        Trả về: (mask_uint8, centroid_xy, steering_error_px, latency_ms)
        """
        # 1. Kiểm tra dữ liệu đầu vào hợp lệ từ OpenCV
        assert frame_bgr is not None, "Ảnh đầu vào từ OpenCV là None!"
        assert isinstance(frame_bgr, np.ndarray) and frame_bgr.size > 0, "Ảnh đầu vào rỗng!"
        assert frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3, (
            f"Ảnh phải ở định dạng 3 kênh màu BGR (H, W, 3), nhận được: {frame_bgr.shape}"
        )
        assert frame_bgr.dtype == np.uint8, f"Kiểu dữ liệu ảnh phải là uint8, nhận được: {frame_bgr.dtype}"

        orig_h, orig_w = frame_bgr.shape[:2]
        t0 = time.perf_counter()

        # 2. Tiền xử lý: Resize về 320x320 và chuyển BGR -> RGB chuẩn (C, H, W)
        resized_bgr = cv2.resize(
            frame_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR
        )
        assert resized_bgr is not None and resized_bgr.shape == (self.input_h, self.input_w, 3)

        rgb_contiguous = np.ascontiguousarray(cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB))

        if not self._debug_logged:
            # In ra kích thước (shape) và kiểu dữ liệu (dtype) ngay trước khi đẩy vào Inference
            chw_debug = np.transpose(rgb_contiguous.astype(np.float32) / 255.0, (2, 0, 1))
            print(
                f"[DEBUG INFERENCE INPUT] Tensor CHW shape: {chw_debug.shape}, "
                f"dtype: {chw_debug.dtype}, color_space: RGB, range: [{chw_debug.min():.3f}, {chw_debug.max():.3f}]"
            )

        # Tạo NCNN Mat trực tiếp từ bộ đệm RGB (3, 320, 320) và chuẩn hóa [0, 1]
        in_mat = ncnn.Mat.from_pixels(
            rgb_contiguous,
            ncnn.Mat.PixelType.PIXEL_RGB,
            self.input_w,
            self.input_h,
        )
        in_mat.substract_mean_normalize(self.mean_vals, self.norm_vals)

        # 3. Chạy suy luận INT8 trên CPU
        ex = self.net.create_extractor()
        ex.input("in0", in_mat)
        ret, out_mat = ex.extract("out0")
        assert ret == 0, f"NCNN Extractor lỗi với mã trả về: {ret}"

        prob_map = np.array(out_mat, dtype=np.float32)
        if not self._debug_logged:
            print(
                f"[DEBUG INFERENCE OUTPUT] Tensor shape: {prob_map.shape}, "
                f"dtype: {prob_map.dtype}, range: [{prob_map.min():.4f}, {prob_map.max():.4f}]"
            )
            self._debug_logged = True

        # Chuẩn hóa kích thước (H, W) từ đầu ra (1, 320, 320) hoặc (320, 320)
        if prob_map.ndim == 3:
            prob_map = prob_map[0]

        # Nhị phân hóa mask và khôi phục về kích thước khung hình camera gốc
        mask_320 = (prob_map >= self.prob_threshold).astype(np.uint8) * 255
        mask_orig = cv2.resize(mask_320, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Khử nhiễu hình thái học nhẹ
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask_orig, cv2.MORPH_OPEN, kernel)

        # 4. Tìm vùng liên thông lớn nhất và tính Trọng tâm (Centroid) bằng Image Moments
        centroid = None
        steering_error = 0.0
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_cnt) > 300.0:
                M = cv2.moments(largest_cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroid = (cx, cy)
                    steering_error = float(cx - (orig_w // 2))

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return mask_clean, centroid, steering_error, latency_ms


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else 0
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    segmenter = BlackPathSegmenterINT8(
        param_path="models/model_int8.param",
        bin_path="models/model_int8.bin",
        input_size=(320, 320),
        num_threads=2,
        prob_threshold=0.5,
    )

    cap = cv2.VideoCapture(source)
    assert cap.isOpened(), f"Không thể mở nguồn camera/video: {source}"
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[INFO] Bắt đầu chạy suy luận INT8 trên Pi 5. Nhấn 'q' để thoát.")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Kết thúc luồng video hoặc không đọc được frame từ camera.")
            break

        assert frame is not None and frame.size > 0, "Frame đọc từ camera không hợp lệ!"

        mask, centroid, error_px, latency_ms = segmenter.infer(frame)
        fps = 1000.0 / max(latency_ms, 1e-3)

        # Hiển thị kết quả trực quan
        vis = frame.copy()
        green_overlay = np.zeros_like(vis)
        green_overlay[:, :, 1] = mask
        vis = cv2.addWeighted(vis, 1.0, green_overlay, 0.45, 0)

        h, w = vis.shape[:2]
        cv2.line(vis, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)

        if centroid is not None:
            cx, cy = centroid
            cv2.circle(vis, (cx, cy), 8, (0, 0, 255), -1)
            cv2.line(vis, (w // 2, cy), (cx, cy), (0, 165, 255), 2)
            status_txt = f"Centroid: ({cx}, {cy}) | Err: {error_px:+.1f}px | {fps:.1f} FPS ({latency_ms:.1f}ms)"
        else:
            status_txt = f"LOST LINE | {fps:.1f} FPS ({latency_ms:.1f}ms)"

        print(status_txt)

        if os.environ.get("DISPLAY"):
            cv2.putText(vis, status_txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.imshow("Pi 5 Edge AI - MobileNetV3-UNet INT8", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

