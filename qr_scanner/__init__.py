"""
QR Scanner Package - Real-time Classical Computer Vision QR Code Detector and Decoder.
Designed for high performance, multi-backend robustness, and edge reliability.
"""

from .detector import QRDetector, QRDetectionResult
from .logger import QRJsonLogger
from .visualizer import QRVisualizer

__all__ = ["QRDetector", "QRDetectionResult", "QRJsonLogger", "QRVisualizer"]

