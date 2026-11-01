"""
Module for saving and loading farm configuration profiles as JSON.
Supports YOLO model inference configuration, detector parameters, color model settings,
and detection mode switching without hardcoded machine-specific absolute paths.
"""

import json
import os
import time
from typing import Any, Dict, Optional


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ProfileManager:
    """Manages serialization and persistence of farm profiles."""

    DEFAULT_PROFILE_DIR = os.path.join(PROJECT_ROOT, "profiles")

    @classmethod
    def save_profile(
        cls,
        filepath: str,
        farm_name: str,
        detector_dict: Dict[str, Any],
        yolo_dict: Optional[Dict[str, Any]] = None,
        color_model_dict: Optional[Dict[str, Any]] = None,
        mode: str = "ai",
        description: str = ""
    ) -> bool:
        """Saves current settings, mode, and model configuration to a JSON file."""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            data = {
                "version": "2.1",
                "farm_name": farm_name,
                "mode": mode,
                "description": description,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "detector": detector_dict,
                "yolo": yolo_dict or {},
                "color_model": color_model_dict or {}
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving profile: {e}")
            return False

    @classmethod
    def load_profile(cls, filepath: str) -> Optional[Dict[str, Any]]:
        """Loads farm profile from JSON file."""
        try:
            if not os.path.exists(filepath):
                return None
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading profile: {e}")
            return None
