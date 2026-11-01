"""
Unit tests for MainWindow sampling within ROI and retrain behavior.
"""

import unittest
import numpy as np
from src.ui_compat import QApplication, QRect
from src.main_window import MainWindow


class TestMainWindowSampling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if not cls.app:
            cls.app = QApplication([])

    def setUp(self):
        self.win = MainWindow()
        # Synthetic 720p frame with mulch in center
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        dummy_frame[300:700, 300:900] = [130, 130, 130]  # gray mulch
        dummy_frame[:300, :] = [30, 160, 40]  # leaves top
        dummy_frame[300:700, :300] = [30, 160, 40]  # leaves left
        dummy_frame[300:700, 900:] = [30, 160, 40]  # leaves right
        self.win.current_frame_bgr = dummy_frame

    def test_clear_samples_and_rebuild_creates_roi_patch(self):
        # Clear all samples
        self.win._clear_all_samples()
        self.assertEqual(len(self.win.color_model.patches), 0)
        self.assertEqual(self.win.table_patches.rowCount(), 0)

        # When rebuild is called on empty patches, it automatically creates a patch from ROI
        self.win._rebuild_color_model()
        self.assertEqual(len(self.win.color_model.patches), 1)
        self.assertEqual(self.win.table_patches.rowCount(), 1)
        p = self.win.color_model.patches[0]
        self.assertEqual(p.category, "path")
        self.assertTrue(self.win.color_model.is_trained)

    def test_sample_from_roi_button(self):
        self.win._clear_all_samples()
        self.win._sample_from_roi()
        self.assertEqual(len(self.win.color_model.patches), 1)
        self.assertEqual(self.win.table_patches.rowCount(), 1)
        p = self.win.color_model.patches[0]
        self.assertEqual(p.category, "path")
        self.assertIn("ROI", p.label)

    def test_drag_selection_strictly_clamped_to_roi(self):
        self.win._clear_all_samples()
        rx, ry, rw, rh = self.win.detector.compute_roi(self.win.current_frame_bgr.shape[:2])

        # Drag rectangle overlapping outside ROI
        self.win._on_sample_rect_selected(QRect(0, 0, 1280, 720))
        self.assertEqual(len(self.win.color_model.patches), 1)
        p = self.win.color_model.patches[0]

        # Patch must be strictly inside ROI
        self.assertGreaterEqual(p.rect[0], rx)
        self.assertGreaterEqual(p.rect[1], ry)
        self.assertLessEqual(p.rect[0] + p.rect[2], rx + rw)
        self.assertLessEqual(p.rect[1] + p.rect[3], ry + rh)
        self.assertEqual(p.category, "path")


if __name__ == "__main__":
    unittest.main()

