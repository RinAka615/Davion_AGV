"""
Automated Integration and Accuracy Test Suite for QR Scanner Pipeline.
"""

import os
import sys
import json
import cv2
import numpy as np

# Thêm root vào sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qr_scanner import QRDetector, QRJsonLogger, QRVisualizer
from qr_scanner.preprocessor import compute_quad_center


def test_standard_qr():
    img_path = "test_qr_dataset/01_standard_qr.png"
    img = cv2.imread(img_path)
    assert img is not None, f"Cannot read {img_path}"

    detector = QRDetector(backend="auto")
    results = detector.detect_and_decode(img)

    assert len(results) == 1, f"Expected 1 QR, got {len(results)}"
    res = results[0]
    assert res.is_decoded is True, "QR was not marked as decoded"
    expected = "https://github.com/warlord/RobotNongNghiep"
    assert res.text == expected, f"Expected '{expected}', got '{res.text}'"

    # Kiểm tra tâm mã QR
    assert res.center is not None, "Center coordinate must exist"
    cx, cy = res.center
    min_x, max_x = np.min(res.points[:, 0]), np.max(res.points[:, 0])
    min_y, max_y = np.min(res.points[:, 1]), np.max(res.points[:, 1])
    assert min_x <= cx <= max_x and min_y <= cy <= max_y, f"Center ({cx}, {cy}) outside bounds"
    print(f"[PASS] test_standard_qr: Decoded '{res.text}' ({res.backend}) | Center: ({cx:.1f}, {cy:.1f})")


def test_tilted_perspective_qr():
    img_path = "test_qr_dataset/02_tilted_perspective_qr.png"
    img = cv2.imread(img_path)
    assert img is not None, f"Cannot read {img_path}"

    detector = QRDetector(backend="auto", enable_dewarp=True)
    results = detector.detect_and_decode(img)

    assert len(results) >= 1, f"Expected at least 1 QR, got {len(results)}"
    res = results[0]
    assert res.is_decoded is True, f"Tilted QR decode failed. Error: {res.error}"
    expected = "QR-MELON-A3-TILTED"
    assert res.text == expected, f"Expected '{expected}', got '{res.text}'"
    assert res.dewarped_patch is not None, "Dewarped patch was not produced"
    print(f"[PASS] test_tilted_perspective_qr: Successfully dewarped & decoded '{res.text}' ({res.backend})")


def test_multi_qr():
    img_path = "test_qr_dataset/03_multi_qr.png"
    img = cv2.imread(img_path)
    assert img is not None, f"Cannot read {img_path}"

    detector = QRDetector(backend="auto")
    results = detector.detect_and_decode(img)

    assert len(results) == 3, f"Expected 3 QRs, found {len(results)}"
    decoded_texts = sorted([r.text for r in results if r.is_decoded])
    expected = sorted(["QR-BED-ROW-01", "QR-BED-ROW-02", "QR-BED-ROW-03"])
    assert decoded_texts == expected, f"Expected {expected}, got {decoded_texts}"
    print(f"[PASS] test_multi_qr: Detected & decoded all 3 QRs simultaneously: {decoded_texts}")


def test_negative_no_qr():
    img_path = "test_qr_dataset/04_negative_greenhouse.png"
    img = cv2.imread(img_path)
    assert img is not None, f"Cannot read {img_path}"

    detector = QRDetector(backend="auto")
    results = detector.detect_and_decode(img)

    # Phải có 0 phát hiện (không nhầm cây/chậu/bạc đen/khung nhà màng thành QR)
    assert len(results) == 0, f"False positive detected: {[r.text for r in results]}"
    print("[PASS] test_negative_no_qr: Clean negative result (0 false positives)")


def test_logger_deduplication():
    test_log = "output_qr/test_dedup.jsonl"
    if os.path.exists(test_log):
        os.remove(test_log)

    logger = QRJsonLogger(log_path=test_log, dedup_cooldown_sec=1.5)
    corners = np.array([[10, 10], [100, 10], [100, 100], [10, 100]])

    # Lần 1: Được ghi
    logged_1 = logger.log("TEST-CODE-A", corners, video_timestamp_sec=1.0)
    assert logged_1 is True, "First log should succeed"

    # Lần 2 (ngay sau đó 0.5s): Bị chặn bởi cooldown
    logged_2 = logger.log("TEST-CODE-A", corners, video_timestamp_sec=1.5)
    assert logged_2 is False, "Duplicate log within cooldown should be suppressed"

    # Lần 3 (mã khác): Được ghi
    logged_3 = logger.log("TEST-CODE-B", corners, video_timestamp_sec=1.6)
    assert logged_3 is True, "Different code should be logged immediately"

    # Lần 4 (sau khi hết cooldown 1.5s, t=3.0s): Được ghi lại
    logged_4 = logger.log("TEST-CODE-A", corners, video_timestamp_sec=3.0)
    assert logged_4 is True, "Log after cooldown should succeed"

    logger.close()

    with open(test_log, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) == 3, f"Expected 3 logged events in JSONL, got {len(lines)}"
    assert lines[0]["text"] == "TEST-CODE-A"
    assert lines[1]["text"] == "TEST-CODE-B"
    assert lines[2]["text"] == "TEST-CODE-A"
    print(f"[PASS] test_logger_deduplication: Cooldown logic correctly verified (3/4 logged, 1 suppressed)")


def test_qr_center_detection():
    # 1. Kiểm tra thuật toán giao điểm 2 đường chéo trên tứ giác lý tưởng
    pts = np.array([
        [100.0, 100.0],  # TL
        [300.0, 100.0],  # TR
        [300.0, 300.0],  # BR
        [100.0, 300.0]   # BL
    ])
    cx, cy = compute_quad_center(pts)
    assert abs(cx - 200.0) < 1e-3 and abs(cy - 200.0) < 1e-3, f"Expected (200, 200), got ({cx}, {cy})"

    # 2. Kiểm tra trên ảnh thực tế có mã nghiêng góc phối cảnh
    img_path = "test_qr_dataset/02_tilted_perspective_qr.png"
    img = cv2.imread(img_path)
    detector = QRDetector(backend="auto", enable_dewarp=True)
    results = detector.detect_and_decode(img)
    assert len(results) >= 1
    res = results[0]
    assert hasattr(res, "center")
    cx, cy = res.center
    assert cx > 0 and cy > 0, f"Invalid center ({cx}, {cy})"

    # 3. Kiểm tra Visualizer vẽ tâm mà không phát sinh ngoại lệ
    vis = QRVisualizer()
    vis_frame = vis.draw_results(img, results)
    assert vis_frame is not None and vis_frame.shape == img.shape

    # Lưu ảnh trực quan hóa tâm QR để người dùng kiểm tra kết quả
    os.makedirs("output_qr", exist_ok=True)
    out_vis_path = "output_qr/test_qr_center_visualized.png"
    cv2.imwrite(out_vis_path, vis_frame)
    assert os.path.exists(out_vis_path), "Visualized frame file was not created"
    print(f"[PASS] test_qr_center_detection: Verified center calculation ({cx:.1f}, {cy:.1f}) & saved {out_vis_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING AUTOMATED TEST SUITE FOR QR SCANNER PIPELINE")
    print("=" * 60)
    test_standard_qr()
    test_tilted_perspective_qr()
    test_multi_qr()
    test_negative_no_qr()
    test_logger_deduplication()
    test_qr_center_detection()
    print("=" * 60)
    print("ALL 6 UNIT & INTEGRATION TESTS PASSED 100%!")
    print("=" * 60)

