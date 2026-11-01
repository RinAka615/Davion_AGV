"""
Unit tests for YOLO segmentation and multi-instance path corridor selection.
"""

import os
import unittest
import numpy as np
import cv2

from src.yolo_segmenter import YoloSegmenter, YoloResult, InstanceDetection
from src.path_detector import PathDetector, TrackingState
from src.profile_manager import PROJECT_ROOT


class TestYoloSegmenter(unittest.TestCase):

    def setUp(self):
        self.model_path = os.path.join(PROJECT_ROOT, "models", "farm_path_bootstrap.pt")
        self.segmenter = YoloSegmenter(model_path=self.model_path, conf_threshold=0.25, imgsz=416, device="cpu")

    def test_model_loaded_successfully(self):
        self.assertTrue(self.segmenter.is_loaded)
        self.assertIn(0, self.segmenter.class_names)
        self.assertEqual(self.segmenter.class_names[0], "bat_loi_di")

    def test_empty_frame_handling(self):
        empty_frame = np.array([])
        res = self.segmenter.predict(empty_frame)
        self.assertFalse(res.is_valid)

    def test_synthetic_frame_no_mulch(self):
        # Frame with only green foliage (no mulch)
        pure_foliage = np.zeros((720, 1280, 3), dtype=np.uint8)
        pure_foliage[:] = [30, 160, 40]
        res = self.segmenter.predict(pure_foliage, conf=0.5)
        self.assertTrue(res.is_valid)
        self.assertIsNotNone(res.chosen_mask)
        # Should return all-zero mask if no instances detected
        self.assertEqual(res.chosen_mask.shape, (720, 1280))

    def test_multi_instance_selection_prefers_foreground_center(self):
        # Mock 2 instances:
        # Instance 0: far left / background (cx=100, bottom_y=300)
        # Instance 1: center foreground path (cx=640, bottom_y=700)
        h, w = 720, 1280
        mask0 = np.zeros((h, w), dtype=np.uint8)
        mask0[100:300, 50:150] = 255
        inst0 = InstanceDetection(
            instance_id=0, class_id=0, class_name="bat_loi_di",
            confidence=0.85, bbox=(50, 100, 100, 200),
            mask=mask0, area_px=20000, centroid=(100, 200), bottom_y=300
        )

        mask1 = np.zeros((h, w), dtype=np.uint8)
        mask1[350:700, 400:880] = 255
        inst1 = InstanceDetection(
            instance_id=1, class_id=0, class_name="bat_loi_di",
            confidence=0.75, bbox=(400, 350, 480, 350),
            mask=mask1, area_px=168000, centroid=(640, 525), bottom_y=700
        )

        chosen_idx, chosen_inst = self.segmenter._select_best_path_instance(
            [inst0, inst1], (h, w), roi_rect=(0, 360, 1280, 360)
        )
        self.assertEqual(chosen_idx, 1)
        self.assertEqual(chosen_inst.instance_id, 1)

    def test_end_to_end_synthetic_frame_with_detector(self):
        # Test full pipeline: segmenter + detector
        detector = PathDetector()
        h, w = 720, 1280
        dummy_frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Mock binary mask resembling real mulch
        mock_mask = np.zeros((h, w), dtype=np.uint8)
        # Center corridor trapezoid from Y=350 to 715
        for y in range(350, 715):
            width = int(200 + (y - 350) * 1.0)
            x_left = 640 - width // 2
            x_right = 640 + width // 2
            mock_mask[y, x_left:x_right] = 255

        det_res = detector.process(dummy_frame, mock_mask, is_source_valid=True)
        self.assertEqual(det_res.state, TrackingState.TRACKING)
        self.assertGreater(det_res.confidence, 70.0)
        self.assertIsNotNone(det_res.lateral_offset_px)
        self.assertAlmostEqual(det_res.lateral_offset_px, 0.0, delta=10.0)
        self.assertIsNotNone(det_res.bounding_box)
        self.assertIsNotNone(det_res.box_center)
    def test_cuda_status_structure(self):
        from src.yolo_segmenter import get_cuda_status, CudaDeviceInfo
        status = get_cuda_status()
        self.assertIsInstance(status, CudaDeviceInfo)
        self.assertIsInstance(status.is_available, bool)
        self.assertIsInstance(status.device_name, str)
        self.assertIsInstance(status.error_reason, str)

    def test_segmenter_respects_cpu_choice(self):
        seg = YoloSegmenter(model_path=self.model_path, device="cpu")
        self.assertEqual(seg.effective_device, "cpu")
        self.assertIn("CPU", seg.device_display_name)
        self.assertEqual(seg.fallback_reason, "")

    def test_segmenter_auto_device_fallback(self):
        seg = YoloSegmenter(model_path=self.model_path, device="auto")
        # On this system where CUDA UVM hits error, auto must either successfully run CUDA or cleanly fallback to CPU
        self.assertIn(seg.effective_device, ("cpu", "cuda", "cuda:0"))
        if seg.effective_device == "cpu":
            self.assertIn("CPU", seg.device_display_name)

    def test_nhamang_batden_model_compatibility(self):
        nhamang_path = os.path.join(PROJECT_ROOT, "models", "nhamang_batden_seg.pt")
        self.assertTrue(os.path.isfile(nhamang_path))
        seg = YoloSegmenter(model_path=nhamang_path, conf_threshold=0.25, imgsz=416, device="cpu")
        self.assertTrue(seg.is_loaded)
        self.assertEqual(seg.class_names, {0: "bat_loi_di"})
        dummy_tile = np.full((720, 1280, 3), 185, dtype=np.uint8)
        res = seg.predict(dummy_tile)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.chosen_mask.shape, (720, 1280))
        self.assertEqual(int(np.count_nonzero(res.chosen_mask)), 0)


if __name__ == "__main__":
    unittest.main()

