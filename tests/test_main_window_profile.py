"""Regression checks for loading all profile controls without signal interference."""

import json
from pathlib import Path
import tempfile
import unittest

from src.main_window import MainWindow
from src.ui_compat import QApplication


class TestMainWindowProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_profile_restores_independent_roi_and_filter_values(self):
        window = MainWindow(initial_model_path="missing_profile_test_model.pt", initial_device="cpu")
        self.addCleanup(window.close)
        settings = {
            "roi_top_pct": .20, "roi_height_pct": .35,
            "roi_width_pct": .75, "roi_center_x_pct": .55,
            "threshold": 150, "k_open": 3, "k_close": 7,
            "smoothing_factor": .60, "num_scanlines": 16,
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profile.json"
            path.write_text(json.dumps({"mode": "ai", "detector": settings}))
            window._load_profile_path(str(path))
        for name, expected in settings.items():
            with self.subTest(setting=name):
                self.assertEqual(getattr(window.detector, name), expected)
        self.assertEqual(window.slider_roi_top.value(), 20)
        self.assertEqual(window.slider_roi_height.value(), 35)
        self.assertEqual(window.lbl_roi_height.text(), "35 %")
        self.assertEqual(window.slider_open.value(), 3)
        self.assertFalse(window.slider_roi_top.signalsBlocked())
        self.assertFalse(window.slider_open.signalsBlocked())
        # Controls must remain interactive after the profile is loaded.
        window.slider_roi_height.setValue(45)
        self.assertEqual(window.detector.roi_height_pct, .45)
        self.assertEqual(window.detector.roi_top_pct, .20)

    def test_explicit_auto_device_is_selected_at_startup(self):
        window = MainWindow(initial_model_path="missing_profile_test_model.pt", initial_device="auto")
        self.addCleanup(window.close)
        self.assertEqual(window.combo_device.currentData(), "auto")
        self.assertEqual(window.segmenter.requested_device, "auto")

    def test_nhamang_batden_profile_loads_cleanly(self):
        window = MainWindow(initial_model_path="missing_profile_test_model.pt", initial_device="cpu")
        self.addCleanup(window.close)
        prof = Path(__file__).resolve().parent.parent / "profiles" / "nhamang_batden_profile.json"
        window._load_profile_path(str(prof))
        self.assertAlmostEqual(window.detector.roi_top_pct, 0.28)
        self.assertAlmostEqual(window.detector.roi_height_pct, 0.67)
        self.assertTrue(window.segmenter.is_loaded)
        self.assertTrue(window.segmenter.model_path.endswith("nhamang_batden_seg.pt"))


if __name__ == "__main__":
    unittest.main()
