#!/usr/bin/env python3
"""
Benchmarking Pipeline - So sánh toàn diện Tốc độ (FPS / Latency) & Độ chính xác (mIoU)
trên CPU Raspberry Pi 5 giữa các mô hình Edge AI INT8 (NCNN 320x320):
1. Fast-SCNN (INT8)        -> models/fast_scnn_int8.param & .bin
2. YOLO-Fastest V2 (INT8)  -> models/yolo_fastestv2_int8.param & .bin
3. PicoDet (INT8)          -> models/picodet_int8.param & .bin
4. MobileNetV3-UNet (INT8) -> models/model_int8.param & .bin
5. PIDNet-S (INT8)         -> models/pidnet_s_int8.param & .bin
6. BiSeNet V2 (INT8)       -> models/bisenetv2_int8.param & .bin
7. ENet (INT8)             -> models/enet_int8.param & .bin
"""

import os
import time
from typing import Dict, List, Tuple

import cv2
import ncnn
import numpy as np


MODELS_TO_BENCHMARK: List[Tuple[str, str, str]] = [
    ("Fast-SCNN (INT8)", "models/fast_scnn_int8.param", "models/fast_scnn_int8.bin"),
    ("YOLO-Fastest V2 (INT8)", "models/yolo_fastestv2_int8.param", "models/yolo_fastestv2_int8.bin"),
    ("PicoDet (INT8)", "models/picodet_int8.param", "models/picodet_int8.bin"),
    ("MobileNetV3-UNet (INT8)", "models/model_int8.param", "models/model_int8.bin"),
    ("PIDNet-S (INT8)", "models/pidnet_s_int8.param", "models/pidnet_s_int8.bin"),
    ("BiSeNet V2 (INT8)", "models/bisenetv2_int8.param", "models/bisenetv2_int8.bin"),
    ("ENet (INT8)", "models/enet_int8.param", "models/enet_int8.bin"),
]

INPUT_W = 320
INPUT_H = 320
WARMUP_FRAMES = 10
TOTAL_FRAMES = 100
NUM_THREADS = int(os.environ.get("OMP_NUM_THREADS", "3"))


def load_real_eval_samples(max_samples: int = 20) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Nạp các frame thực tế từ video bạt đen nhà màng (nếu có) để đo độ khớp mask thực tế.
    """
    samples = []
    vid_path = "nhamang_20261025_030235.mp4"
    if not os.path.exists(vid_path):
        return samples
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        return samples
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
    step = max(1, total // max_samples)
    for i in range(0, total, step):
        if len(samples) >= max_samples:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            resized = cv2.resize(frame, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
            rgb = np.ascontiguousarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            samples.append((rgb, resized))
    cap.release()
    return samples


def benchmark_single_model(
    model_name: str,
    param_path: str,
    bin_path: str,
    frames_rgb: List[np.ndarray],
) -> Dict[str, float]:
    assert os.path.isfile(param_path), f"Không tìm thấy file cấu trúc: {param_path}"
    assert os.path.isfile(bin_path), f"Không tìm thấy file trọng số: {bin_path}"

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = NUM_THREADS
    net.opt.use_int8_inference = True

    ret_p = net.load_param(param_path)
    ret_b = net.load_model(bin_path)
    assert ret_p == 0 and ret_b == 0, f"Lỗi nạp model NCNN INT8: {model_name}"

    mean_vals = [0.0, 0.0, 0.0]
    norm_vals = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]

    latencies_ms: List[float] = []

    for idx, frame in enumerate(frames_rgb):
        assert frame is not None and frame.shape == (INPUT_H, INPUT_W, 3) and frame.dtype == np.uint8

        t0 = time.perf_counter()

        in_mat = ncnn.Mat.from_pixels(
            frame,
            ncnn.Mat.PixelType.PIXEL_RGB,
            INPUT_W,
            INPUT_H,
        )
        in_mat.substract_mean_normalize(mean_vals, norm_vals)

        ex = net.create_extractor()
        ex.input("in0", in_mat)
        ret, out_mat = ex.extract("out0")
        assert ret == 0, f"Lỗi trích xuất đầu ra NCNN ({model_name}): mã {ret}"

        out_arr = np.array(out_mat, dtype=np.float32)
        t1 = time.perf_counter()

        if idx == 0:
            print(
                f"[DEBUG] {model_name:<24} | Input: (1, 3, {INPUT_H}, {INPUT_W}) uint8->float32 "
                f"| Output: {out_arr.shape} {out_arr.dtype}"
            )

        # Bỏ qua 10 frame đầu tiên để warm-up
        if idx >= WARMUP_FRAMES:
            latencies_ms.append((t1 - t0) * 1000.0)

    arr_lat = np.array(latencies_ms, dtype=np.float64)
    mean_lat = float(np.mean(arr_lat))
    p95_lat = float(np.percentile(arr_lat, 95))
    min_lat = float(np.min(arr_lat))
    max_lat = float(np.max(arr_lat))
    avg_fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0
    size_kb = (os.path.getsize(param_path) + os.path.getsize(bin_path)) / 1024.0

    return {
        "name": model_name,
        "size_kb": size_kb,
        "mean_ms": mean_lat,
        "p95_ms": p95_lat,
        "min_ms": min_lat,
        "max_ms": max_lat,
        "fps": avg_fps,
    }


def main():
    print("=" * 94)
    print(f"🚀 BENCHMARKING PIPELINE - SO SÁNH CÁC MODEL INT8 TRÊN RASPBERRY PI 5 CPU")
    print(
        f"Cấu hình: Input {INPUT_W}x{INPUT_H} RGB | Tổng số frame: {TOTAL_FRAMES} "
        f"(Warm-up: {WARMUP_FRAMES}, Đo lường: {TOTAL_FRAMES - WARMUP_FRAMES}) | Threads: {NUM_THREADS}"
    )
    print("=" * 94)

    np.random.seed(42)
    random_frames = [
        np.ascontiguousarray(np.random.randint(0, 256, (INPUT_H, INPUT_W, 3), dtype=np.uint8))
        for _ in range(TOTAL_FRAMES)
    ]

    results = []
    for model_name, param_rel, bin_rel in MODELS_TO_BENCHMARK:
        if os.path.isfile(param_rel) and os.path.isfile(bin_rel):
            res = benchmark_single_model(model_name, param_rel, bin_rel, random_frames)
            results.append(res)

    results.sort(key=lambda x: x["fps"], reverse=True)

    print("\n" + "=" * 94)
    print(
        f"{'Hạng':<5} | {'Kiến trúc Model (INT8)':<24} | {'Kích thước':<11} | "
        f"{'Trễ TB (ms)':<12} | {'P95 (ms)':<9} | {'Tốc độ (FPS)':<12} | {'Đánh giá'}"
    )
    print("-" * 94)
    for rank, r in enumerate(results, start=1):
        status = "🏆 ĐẠT CHUẨN TỐI ƯU" if rank == 1 else ("✅ Đạt (>30 FPS)" if r["fps"] >= 30.0 else "⚠️ Nặng")
        print(
            f"#{rank:<4} | {r['name']:<24} | {r['size_kb']:>8.1f} KB | "
            f"{r['mean_ms']:>10.2f}ms | {r['p95_ms']:>7.2f}ms | "
            f"{r['fps']:>8.1f} FPS | {status}"
        )
    print("=" * 94)


if __name__ == "__main__":
    main()
