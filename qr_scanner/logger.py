"""
JSONL Event Logger with Configurable Deduplication Cooldown.
"""

import os
import json
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import numpy as np


class QRJsonLogger:
    """
    Ghi nhật ký sự kiện quét mã QR thành công ra định dạng JSON Lines (.jsonl).
    Tích hợp cơ chế chống ghi log lặp (deduplication) với thời gian cooldown tùy chỉnh.
    """

    def __init__(self, log_path: str = "output_qr/qr_events.jsonl", dedup_cooldown_sec: float = 2.0):
        self.log_path = log_path
        self.dedup_cooldown_sec = dedup_cooldown_sec
        self.last_logged_time: Dict[str, float] = {}
        self.total_events_logged = 0
        self.unique_codes_seen = set()

        # Đảm bảo thư mục đầu ra tồn tại
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        self._file = open(self.log_path, "a", encoding="utf-8")

    def should_log(self, text: str, current_time: Optional[float] = None) -> bool:
        """
        Kiểm tra xem mã text có được phép ghi log hay đang trong thời gian cooldown.
        """
        if not text or not text.strip():
            return False

        now = current_time if current_time is not None else time.time()
        last_time = self.last_logged_time.get(text)

        if last_time is None or (now - last_time) >= self.dedup_cooldown_sec:
            return True
        return False

    def log(
        self,
        text: str,
        corners: np.ndarray,
        backend: str = "zxing-cpp",
        video_timestamp_sec: Optional[float] = None,
        barcode_format: str = "QR_CODE",
        geometry: Optional[Dict[str, Any]] = None,
        extra_meta: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Ghi một sự kiện đọc QR thành công nếu vượt qua kiểm tra chống trùng lặp.
        
        :return: True nếu sự kiện được ghi, False nếu bị chặn bởi cooldown.
        """
        now = time.time()
        # Đối với video file, có thể dùng video_timestamp_sec cho logic cooldown hoặc wall-clock time
        reference_time = video_timestamp_sec if video_timestamp_sec is not None else now

        if not self.should_log(text, reference_time):
            return False

        # Cập nhật thời điểm ghi gần nhất
        self.last_logged_time[text] = reference_time
        self.unique_codes_seen.add(text)
        self.total_events_logged += 1

        # Chuyển đổi tọa độ góc sang danh sách số nguyên/thực có thể serialize JSON
        corners_list = []
        if corners is not None:
            corners_arr = np.asarray(corners).reshape(-1, 2)
            corners_list = [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in corners_arr]

        iso_timestamp = datetime.now().astimezone().isoformat()

        record = {
            "timestamp": iso_timestamp,
            "video_timestamp_sec": round(video_timestamp_sec, 3) if video_timestamp_sec is not None else None,
            "text": text,
            "corners": corners_list,
            "backend": backend,
            "format": barcode_format
        }

        if geometry:
            record["size_px"] = {
                "width": geometry.get("width_px"),
                "height": geometry.get("height_px"),
                "area": geometry.get("area_px"),
                "aspect_ratio": geometry.get("aspect_ratio"),
                "orientation_deg": geometry.get("orientation_deg")
            }
            if "center" in geometry:
                record["center"] = list(geometry.get("center"))

        if extra_meta:
            record["meta"] = extra_meta

        line = json.dumps(record, ensure_ascii=False)
        self._file.write(line + "\n")
        self._file.flush()
        return True

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_events_logged": self.total_events_logged,
            "unique_codes_count": len(self.unique_codes_seen),
            "log_file": self.log_path,
            "cooldown_sec": self.dedup_cooldown_sec
        }

    def close(self):
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    def __del__(self):
        self.close()

