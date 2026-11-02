#!/usr/bin/env python3
"""
Benchmarking Pipeline - Đo lường tốc độ suy luận (FPS / Latency) trên CPU Raspberry Pi 5
So sánh 4 kiến trúc Semantic Segmentation siêu nhẹ đã lượng tử hóa tĩnh INT8 (NCNN):
1. Fast-SCNN (INT8)
2. BiSeNet V2 (INT8)
3. PIDNet-S (INT8)
4. ENet (INT8)
- Kích thước đầu vào: 320x320 (RGB), Output: 1 class ("đường đi")
"""

import os
import time
from typing import Dict, List, Tuple

import ncnn
import numpy as np


MODELS_TO_BENCHMARK: List[Tuple[str, str, str]] = [
    ("Fast-SCNN (INT8)", "models/fast_scnn_int8.param", "models/fast_scnn_int8.bin"),
    ("BiSeNet V2 (INT8)", "models/bisenetv2_int8.param", "models/bisenetv2_int8.bin"),
    ("PIDNet-S (INT8)", "models/pidnet_s_int8.param", "models/pidnet_s_int8.bin"),
    ("ENet (INT8)", "models/enet_int8.param", "models/enet_int8.bin"),
]

INPUT_W = 320
INPUT_H = 320
WARMUP_FRAMES = 10
TOTAL_FRAMES = 100
NUM_THREADS = 2


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
                f"[DEBUG] {model_name:<18} | Input: (1, 3, {INPUT_H}, {INPUT_W}) uint8->float32 "
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
    print("=" * 88)
    print(f"🚀 BENCHMARKING PIPELINE - SEMANTIC SEGMENTATION INT8 TRÊN RASPBERRY PI 5 CPU")
    print(
        f"Cấu hình: Input {INPUT_W}x{INPUT_H} RGB | Tổng số frame: {TOTAL_FRAMES} "
        f"(Warm-up: {WARMUP_FRAMES}, Đo lường: {TOTAL_FRAMES - WARMUP_FRAMES}) | Threads: {NUM_THREADS}"
    )
    print("=" * 88)

    # Khởi tạo 100 khung hình ngẫu nhiên chuẩn bộ đệm liên tục (contiguous uint8 RGB)
    np.random.seed(42)
    random_frames = [
        np.ascontiguousarray(np.random.randint(0, 256, (INPUT_H, INPUT_W, 3), dtype=np.uint8))
        for _ in range(TOTAL_FRAMES)
    ]

    results = []
    for model_name, param_rel, bin_rel in MODELS_TO_BENCHMARK:
        res = benchmark_single_model(model_name, param_rel, bin_rel, random_frames)
        results.append(res)

    # Sắp xếp từ nhanh nhất (FPS cao nhất) xuống chậm nhất
    results.sort(key=lambda x: x["fps"], reverse=True)

    print("\n" + "=" * 88)
    print(
        f"{'Hạng':<5} | {'Kiến trúc Model (INT8)':<20} | {'Kích thước':<11} | "
        f"{'Trễ TB (ms)':<12} | {'P95 (ms)':<9} | {'Tốc độ (FPS)':<12} | {'Đánh giá'}"
    )
    print("-" * 88)
    for rank, r in enumerate(results, start=1):
        status = "🏆 ĐẠT CHUẨN TỐI ƯU" if rank == 1 else ("✅ Đạt (>30 FPS)" if r["fps"] >= 30.0 else "⚠️ Nặng")
        print(
            f"#{rank:<4} | {r['name']:<20} | {r['size_kb']:>8.1f} KB | "
            f"{r['mean_ms']:>10.2f}ms | {r['p95_ms']:>7.2f}ms | "
            f"{r['fps']:>8.1f} FPS | {status}"
        )
    print("=" * 88)


if __name__ == "__main__":
    main()

