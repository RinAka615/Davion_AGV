"""
Unit tests for VideoReader module.
"""

import os
import unittest
import numpy as np
from src.video_reader import VideoReader


class TestVideoReader(unittest.TestCase):

    def setUp(self):
        # Look for real video in project directory
        video_dir = "/home/warlord/RobotNongNghiep/Quỳnh Robotics [13-10-2026 20_33]"
        self.sample_video = None
        if os.path.exists(video_dir):
            mp4s = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
            if mp4s:
                self.sample_video = mp4s[0]

    def test_open_nonexistent(self):
        reader = VideoReader("nonexistent_video_path.mp4")
        self.assertFalse(reader.is_opened)
        self.assertIsNone(reader.cap)

    def test_open_and_read(self):
        if not self.sample_video:
            self.skipTest("No test video found")

        reader = VideoReader(self.sample_video)
        self.assertTrue(reader.is_opened)
        self.assertGreater(reader.width, 0)
        self.assertGreater(reader.height, 0)
        self.assertGreater(reader.total_frames, 0)
        self.assertGreater(reader.fps, 0)

        # Read first frame
        ret, frame, f_idx, ts = reader.read_frame()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (reader.height, reader.width, 3))
        self.assertEqual(f_idx, 0)

        # Seek to frame 10
        ret, frame_seek = reader.seek(10)
        self.assertTrue(ret)
        self.assertEqual(reader.current_frame_idx, 10)
        self.assertTrue(reader.consume_seek_event())
        self.assertFalse(reader.consume_seek_event())  # Should be cleared

        # Step forward
        ret, frame_step = reader.step(2)
        self.assertTrue(ret)
        self.assertEqual(reader.current_frame_idx, 12)

        reader.close()
        self.assertFalse(reader.is_opened)


if __name__ == "__main__":
    unittest.main()

