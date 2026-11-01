"""
Module for thread-safe video playback and frame navigation.
"""

import threading
import time
from typing import Any, Optional, Tuple, Union
import cv2
import numpy as np


class VideoReader:
    """Manages OpenCV VideoCapture with playback control, seek, and frame stepping."""

    def __init__(self, video_path: Optional[str] = None):
        self._lock = threading.Lock()
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_path: Optional[str] = None
        self.width: int = 0
        self.height: int = 0
        self.fps: float = 30.0
        self.total_frames: int = 0
        self.current_frame_idx: int = 0
        self.is_opened: bool = False
        self.is_live: bool = False
        self.last_frame: Optional[np.ndarray] = None
        
        # Seek flag to notify downstream modules to reset temporal tracking state
        self.seek_occurred: bool = False

        if video_path is not None:
            self.open(video_path)

    def open(self, video_path: Any) -> bool:
        """Opens a video file or live camera stream (e.g. 0, '0', '/dev/video0')."""
        with self._lock:
            self.close()
            is_camera = False
            device = video_path

            if isinstance(video_path, int):
                is_camera = True
            elif isinstance(video_path, str):
                s_strip = video_path.strip()
                if s_strip.isdigit():
                    is_camera = True
                    device = int(s_strip)
                elif s_strip.startswith("/dev/video"):
                    is_camera = True
                    device = s_strip

            if is_camera:
                cap = cv2.VideoCapture(device)
                if not cap.isOpened():
                    return False

                # Configure preferred format for Orbbec: MJPG 1280x720 @ 30 FPS
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)

                self.cap = cap
                self.video_path = f"Camera {device}"
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
                fps = cap.get(cv2.CAP_PROP_FPS)
                self.fps = fps if (fps and fps > 1.0) else 30.0
                self.total_frames = -1
                self.current_frame_idx = 0
                self.is_live = True
                self.is_opened = True
                self.seek_occurred = True

                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.last_frame = frame.copy()
                    self.width = frame.shape[1]
                    self.height = frame.shape[0]
                return True

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False

            self.cap = cap
            self.video_path = str(video_path)
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps if (fps and fps > 1.0) else 30.0
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame_idx = 0
            self.is_live = False
            self.is_opened = True
            self.seek_occurred = True

            # Read the very first frame
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.last_frame = frame
                self.current_frame_idx = 0
                # Rewind to frame 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return True

    def close(self):
        """Closes the current video capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_opened = False
        self.is_live = False
        self.last_frame = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.seek_occurred = True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], int, float]:
        """Reads the next sequential frame.
        
        Returns:
            (success, frame_bgr, frame_idx, timestamp_sec)
        """
        with self._lock:
            if not self.is_opened or self.cap is None:
                return False, None, 0, 0.0

            ret, frame = self.cap.read()
            if not ret or frame is None:
                return False, None, self.current_frame_idx, self.get_timestamp_sec()

            if self.is_live:
                self.current_frame_idx += 1
            else:
                self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            self.last_frame = frame.copy()
            return True, frame, self.current_frame_idx, self.get_timestamp_sec()

    def seek(self, frame_idx: int) -> Tuple[bool, Optional[np.ndarray]]:
        """Seeks to a specific frame index and resets tracking flag."""
        with self._lock:
            if not self.is_opened or self.cap is None:
                return False, None

            if self.is_live:
                return True, self.last_frame

            frame_idx = max(0, min(frame_idx, max(0, self.total_frames - 1)))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.current_frame_idx = frame_idx
                self.last_frame = frame.copy()
                self.seek_occurred = True
                return True, self.last_frame
            return False, None

    def step(self, delta: int = 1) -> Tuple[bool, Optional[np.ndarray]]:
        """Steps forward (+delta) or backward (-delta) by frames."""
        target = self.current_frame_idx + delta
        return self.seek(target)

    def get_timestamp_sec(self) -> float:
        """Calculates current timestamp in seconds."""
        if self.fps > 0:
            return self.current_frame_idx / self.fps
        return 0.0

    def consume_seek_event(self) -> bool:
        """Returns True if a seek occurred since last checked, then clears the flag."""
        with self._lock:
            event = self.seek_occurred
            self.seek_occurred = False
            return event

    def skip_frames(self, count: int) -> int:
        """Fast-forwards by grabbing frames without full decoding (ultra-fast)."""
        with self._lock:
            if not self.is_opened or self.cap is None or count <= 0 or self.is_live:
                return 0
            skipped = 0
            for _ in range(count):
                if self.current_frame_idx >= self.total_frames - 1:
                    break
                if self.cap.grab():
                    self.current_frame_idx += 1
                    skipped += 1
                else:
                    break
            return skipped

