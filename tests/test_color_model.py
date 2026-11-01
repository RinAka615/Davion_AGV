"""
Unit tests for ColorModel in CIELAB color space.
"""

import unittest
import numpy as np
from src.color_model import ColorModel, ColorPatch


class TestColorModel(unittest.TestCase):

    def setUp(self):
        self.model = ColorModel()
        # Synthetic test image (300x400 BGR)
        self.dummy_frame = np.zeros((300, 400, 3), dtype=np.uint8)
        # Center has gray mulch [128, 128, 128]
        self.dummy_frame[:, 100:300] = [128, 128, 128]
        # Sides have green leaves [30, 180, 40]
        self.dummy_frame[:, :100] = [30, 180, 40]
        self.dummy_frame[:, 300:] = [30, 180, 40]

    def test_untrained_scoring(self):
        self.assertFalse(self.model.is_trained)
        score = self.model.score_frame(self.dummy_frame)
        self.assertEqual(score.shape, (300, 400))
        self.assertEqual(np.count_nonzero(score), 0)

    def test_add_patches_and_training(self):
        # Add path patch in center
        p_path = self.model.add_patch(self.dummy_frame, (150, 50, 50, 50), "path", "Bạt xám")
        self.assertIsNotNone(p_path)
        self.assertTrue(self.model.is_trained)
        self.assertEqual(len(self.model.patches), 1)

        # Score frame with only path patch
        score_path_only = self.model.score_frame(self.dummy_frame)
        self.assertGreater(np.mean(score_path_only[:, 150:250]), np.mean(score_path_only[:, :50]))

        # Add background patch on left
        p_bg = self.model.add_patch(self.dummy_frame, (10, 50, 50, 50), "background", "Lá xanh")
        self.assertIsNotNone(p_bg)
        self.assertEqual(len(self.model.patches), 2)

        # Score with both path and background
        score_both = self.model.score_frame(self.dummy_frame)
        # Center (gray mulch) should have high score
        self.assertGreater(np.mean(score_both[:, 150:250]), 180)
        # Left/Right (green leaves) should have low score
        self.assertLess(np.mean(score_both[:, :50]), 50)
        self.assertLess(np.mean(score_both[:, 350:]), 50)

    def test_remove_and_clear(self):
        p1 = self.model.add_patch(self.dummy_frame, (150, 50, 50, 50), "path")
        p2 = self.model.add_patch(self.dummy_frame, (10, 50, 50, 50), "background")
        self.assertEqual(len(self.model.patches), 2)

        self.model.remove_patch(p1.patch_id, self.dummy_frame)
        self.assertEqual(len(self.model.patches), 1)

        self.model.clear_patches()
        self.assertEqual(len(self.model.patches), 0)
        self.assertFalse(self.model.is_trained)

    def test_serialization(self):
        self.model.add_patch(self.dummy_frame, (150, 50, 50, 50), "path", "Bạt")
        self.model.add_patch(self.dummy_frame, (10, 50, 50, 50), "background", "Lá")
        data = self.model.to_dict()

        new_model = ColorModel()
        new_model.from_dict(data, self.dummy_frame)
        self.assertEqual(len(new_model.patches), 2)
        self.assertTrue(new_model.is_trained)


if __name__ == "__main__":
    unittest.main()

