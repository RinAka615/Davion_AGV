"""
RobotNongNghiep core package for agricultural greenhouse path detection.
"""

from .video_reader import VideoReader
from .color_model import ColorModel, ColorPatch
from .path_detector import PathDetector, DetectionResult, TrackingState
from .profile_manager import ProfileManager
from .navigation_fusion import FusedNavigationTarget, fuse_navigation_target, check_qr_exceeds_threshold

__all__ = [
    "VideoReader",
    "ColorModel",
    "ColorPatch",
    "PathDetector",
    "DetectionResult",
    "TrackingState",
    "ProfileManager",
    "FusedNavigationTarget",
    "fuse_navigation_target",
    "check_qr_exceeds_threshold"
]

