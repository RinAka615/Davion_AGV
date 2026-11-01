"""
Unit tests for PathDetector module covering edge cases and detection logic.
"""

import unittest
import numpy as np
from src.path_detector import PathDetector, TrackingState


class TestPathDetector(unittest.TestCase):

    def setUp(self):
        self.detector = PathDetector()
        self.detector.roi_top_pct = 0.5
        self.detector.roi_height_pct = 0.5
        self.h_img, self.w_img = 400, 600
        self.dummy_frame = np.zeros((self.h_img, self.w_img, 3), dtype=np.uint8)
        self.rx, self.ry, self.rw, self.rh = self.detector.compute_roi((self.h_img, self.w_img))

    def test_no_color_model(self):
        score_map = np.zeros((self.rh, self.rw), dtype=np.uint8)
        res = self.detector.process(self.dummy_frame, score_map, is_color_model_trained=False)
        self.assertEqual(res.state, TrackingState.NO_COLOR_MODEL)
        self.assertEqual(res.confidence, 0.0)
        self.assertIsNone(res.lateral_offset_px)

    def test_all_black_score_map(self):
        # When path is completely lost
        score_map = np.zeros((self.rh, self.rw), dtype=np.uint8)
        # Run multiple times to trigger LOST state after max_lost_frames
        for _ in range(self.detector.max_lost_frames + 2):
            res = self.detector.process(self.dummy_frame, score_map, is_color_model_trained=True)
        self.assertEqual(res.state, TrackingState.LOST)
        self.assertEqual(res.confidence, 0.0)

    def test_clear_corridor(self):
        # Create a trapezoid corridor in ROI: narrow at top, wide at bottom
        score_map = np.zeros((self.rh, self.rw), dtype=np.uint8)
        # At top of ROI (y=0): x from 240 to 360 (width 120, center 300)
        # At bottom of ROI (y=rh): x from 150 to 450 (width 300, center 300)
        for y in range(self.rh):
            t = y / float(self.rh)
            w = int(120 + t * 180)
            x1 = 300 - w // 2
            x2 = 300 + w // 2
            score_map[y, x1:x2] = 255

        res = self.detector.process(self.dummy_frame, score_map, is_color_model_trained=True)
        self.assertEqual(res.state, TrackingState.TRACKING)
        self.assertGreater(res.confidence, 65.0)
        self.assertIsNotNone(res.lateral_offset_px)
        # Since corridor center is at x=300 (which is w_img/2 = 600/2 = 300), offset should be near 0
        self.assertAlmostEqual(res.lateral_offset_px, 0.0, delta=5.0)
        self.assertTrue("Đúng tâm" in res.offset_direction or "Dung tam" in res.offset_direction)

    def test_lateral_offset_shifted_left(self):
        # Create corridor shifted to the left (center at 200 instead of 300)
        score_map = np.zeros((self.rh, self.rw), dtype=np.uint8)
        for y in range(self.rh):
            score_map[y, 100:300] = 255  # Center is at 200

        res = self.detector.process(self.dummy_frame, score_map, is_color_model_trained=True)
        self.assertIsNotNone(res.lateral_offset_px)
        # Offset should be negative (shifted left relative to camera center at 300)
        self.assertLess(res.lateral_offset_px, -50.0)
        self.assertTrue("Lệch trái" in res.offset_direction or "Lech trai" in res.offset_direction)

    def test_boundary_clipping(self):
        # Corridor that touches both left and right edges (all white ROI)
        score_map = np.ones((self.rh, self.rw), dtype=np.uint8) * 255
        res = self.detector.process(self.dummy_frame, score_map, is_color_model_trained=True)
        # Scanlines should flag boundary clipping
        clipped_scans = [s for s in res.scanlines if s.left_clipped and s.right_clipped]
        self.assertGreater(len(clipped_scans), 0)

    def test_horizontal_roi(self):
        # Set ROI width to 60% and centered
        self.detector.roi_width_pct = 0.60
        self.detector.roi_center_x_pct = 0.50
        rx, ry, rw, rh = self.detector.compute_roi((self.h_img, self.w_img))
        self.assertEqual(rw, int(self.w_img * 0.60))
        self.assertEqual(rx, int((self.w_img - rw) / 2))

        # Shift center to 40%
        self.detector.roi_center_x_pct = 0.40
        rx2, ry2, rw2, rh2 = self.detector.compute_roi((self.h_img, self.w_img))
        self.assertEqual(rw2, int(self.w_img * 0.60))
        self.assertLess(rx2, rx)

    def test_center_method_box(self):
        score_map = np.zeros((self.rh, self.rw), dtype=np.uint8)
        score_map[20:self.rh - 20, 200:400] = 255
        self.detector.center_method = "box"
        res = self.detector.process(self.dummy_frame, score_map, is_color_model_trained=True)
        self.assertEqual(res.center_method, "box")
        self.assertIsNotNone(res.box_center)
        self.assertIsNotNone(res.bounding_box)

    def test_center_method_path(self):
        score_map = np.zeros((self.rh, self.rw), dtype=np.uint8)
        # Create clear corridor so scanlines find unclipped edges
        for y in range(self.rh):
            t = y / float(self.rh)
            w = int(120 + t * 180)
            x1 = 300 - w // 2
            x2 = 300 + w // 2
            score_map[y, x1:x2] = 255

        self.detector.center_method = "path"
        res = self.detector.process(self.dummy_frame, score_map, is_color_model_trained=True)
        self.assertEqual(res.center_method, "path")
        self.assertIsNotNone(res.path_center)
        self.assertGreater(len(res.centerline_points), 0)
        self.assertIsNotNone(res.ref_y_global)
        self.assertIn("đường dẫn", res.offset_direction.lower())


if __name__ == "__main__":
    unittest.main()

