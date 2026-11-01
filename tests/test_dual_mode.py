"""
Unit tests for Dual-Mode switching: AI (YOLO) and Image Processing (Color Model).
"""

import unittest
import numpy as np
from src.ui_compat import QApplication
from src.main_window import MainWindow


class TestDualMode(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance()
        if not cls.app:
            cls.app = QApplication([])

    def setUp(self):
        self.win = MainWindow(initial_mode="ai")
        # Synthetic 720p frame
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        dummy[350:700, 350:930] = [120, 120, 120]  # mulch
        dummy[:350, :] = [30, 160, 40]
        self.win.current_frame_bgr = dummy

    def test_default_mode_is_ai(self):
        self.assertEqual(self.win.active_mode, MainWindow.MODE_AI)
        self.assertFalse(self.win.group_yolo.isHidden())
        self.assertTrue(self.win.group_color.isHidden())
        self.assertTrue(self.win.row_threshold_widget.isHidden())

    def test_switch_to_color_mode(self):
        self.win.combo_mode.setCurrentIndex(1)
        self.assertEqual(self.win.active_mode, MainWindow.MODE_COLOR)
        self.assertTrue(self.win.group_yolo.isHidden())
        self.assertFalse(self.win.group_color.isHidden())
        self.assertFalse(self.win.row_threshold_widget.isHidden())

    def test_switch_back_to_ai_mode(self):
        self.win.combo_mode.setCurrentIndex(1)
        self.assertEqual(self.win.active_mode, MainWindow.MODE_COLOR)
        self.win.combo_mode.setCurrentIndex(0)
        self.assertEqual(self.win.active_mode, MainWindow.MODE_AI)
        self.assertFalse(self.win.group_yolo.isHidden())
        self.assertTrue(self.win.group_color.isHidden())
        self.assertTrue(self.win.row_threshold_widget.isHidden())

    def test_color_mode_processing(self):
        self.win.combo_mode.setCurrentIndex(1)
        # Sample patch from ROI
        self.win._sample_from_roi()
        self.assertTrue(self.win.color_model.is_trained)
        # Process frame in color mode
        self.win._process_single_frame(self.win.current_frame_bgr, 0, 0.0)
        self.assertIsNotNone(self.win.last_detection_result)
        self.assertIsNotNone(self.win.last_score_map)


    def test_uart_quick_button_toggle(self):
        # Initial state: False
        self.assertFalse(self.win.uart.is_enabled)
        self.assertFalse(self.win.btn_quick_uart.isChecked())
        self.assertIn("BẬT", self.win.btn_quick_uart.text())

        # Click quick button -> should turn ON
        self.win.btn_quick_uart.click()
        self.assertTrue(self.win.uart.is_enabled)
        self.assertTrue(self.win.btn_quick_uart.isChecked())
        self.assertIn("ĐANG GỬI", self.win.btn_quick_uart.text())
        self.assertTrue(self.win.chk_uart_enable.isChecked())

        # Click again -> should turn OFF
        self.win.btn_quick_uart.click()
        self.assertFalse(self.win.uart.is_enabled)
        self.assertFalse(self.win.btn_quick_uart.isChecked())
        self.assertIn("BẬT", self.win.btn_quick_uart.text())
        self.assertFalse(self.win.chk_uart_enable.isChecked())


if __name__ == "__main__":
    unittest.main()

