"""
Qt compatibility layer supporting both PySide6 and PyQt5 seamlessly.
Automatically selects PySide6 when system libraries (like libxcb-cursor on Linux)
are satisfied, or falls back to PyQt5 for rock-solid stability.
"""

import sys
import os

def _check_pyside6_viable() -> bool:
    if os.environ.get("FORCE_PYQT5", "0") == "1":
        return False
    if os.environ.get("FORCE_PYSIDE6", "0") == "1":
        return True
    # Trên Wayland (mặc định của Ubuntu GNOME 22.04+), PySide6 chạy trực tiếp qua wayland plugin không cần libxcb-cursor
    if os.environ.get("XDG_SESSION_TYPE") == "wayland" or bool(os.environ.get("WAYLAND_DISPLAY")):
        return True
    if sys.platform.startswith("linux") and os.environ.get("QT_QPA_PLATFORM") not in ("offscreen", "minimal"):
        try:
            import ctypes
            ctypes.CDLL("libxcb-cursor.so.0")
            return True
        except OSError:
            # On Linux X11 without libxcb-cursor0, Qt 6.5+ crashes on X11, while PyQt5 works cleanly.
            return False
    return True

use_pyside6 = _check_pyside6_viable()

if use_pyside6:
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
        from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QRect, QPoint, QSize
        from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QIcon
        from PySide6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QGridLayout, QSplitter, QLabel, QPushButton, QSlider, QCheckBox,
            QComboBox, QFileDialog, QMessageBox, QGroupBox, QFrame, QScrollArea,
            QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit, QStatusBar,
            QStyle
        )
        QT_BINDING = "PySide6"
    except ImportError:
        use_pyside6 = False

if not use_pyside6:
    import PyQt5
    _pyqt5_plugins = os.path.join(os.path.dirname(PyQt5.__file__), "Qt5", "plugins")
    if os.path.exists(_pyqt5_plugins):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _pyqt5_plugins

    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtCore import (
        Qt, QThread, pyqtSignal as Signal, pyqtSlot as Slot, QTimer,
        QRect, QPoint, QSize
    )
    from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QIcon
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QSplitter, QLabel, QPushButton, QSlider, QCheckBox,
        QComboBox, QFileDialog, QMessageBox, QGroupBox, QFrame, QScrollArea,
        QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit, QStatusBar,
        QStyle
    )
    QT_BINDING = "PyQt5"

__all__ = [
    "QtCore", "QtGui", "QtWidgets", "Qt", "QThread", "Signal", "Slot", "QTimer",
    "QRect", "QPoint", "QSize", "QImage", "QPixmap", "QPainter", "QPen", "QColor",
    "QFont", "QIcon", "QApplication", "QMainWindow", "QWidget", "QVBoxLayout",
    "QHBoxLayout", "QGridLayout", "QSplitter", "QLabel", "QPushButton", "QSlider",
    "QCheckBox", "QComboBox", "QFileDialog", "QMessageBox", "QGroupBox", "QFrame",
    "QScrollArea", "QTableWidget", "QTableWidgetItem", "QHeaderView", "QLineEdit",
    "QStatusBar", "QStyle", "QT_BINDING"
]
