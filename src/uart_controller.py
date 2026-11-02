"""
Module for UART communication with STM32 Motor/Steering Controller.
Supports persistent device symlink (/dev/ttySTM32), rate limiting,
NMEA-style packet generation with XOR checksum, and auto-reconnection.
"""

import glob
import os
import struct
import threading
import time
from typing import List, Optional, Tuple, Union
import serial

try:
    import numpy as np
except ImportError:
    np = None


def calculate_xor_checksum(payload: str) -> str:
    """Calculates 2-hex-digit XOR checksum (NMEA style)."""
    checksum = 0
    for ch in payload:
        checksum ^= ord(ch)
    return f"{checksum:02X}"


class UartController:
    """Manages thread-safe serial transmission to STM32 robot controller.
    Supports both 5-byte binary union packets (4B offset + 1B detect, every 0.1s)
    and NMEA text packets with XOR checksum.
    """

    # Protocols
    PROTOCOL_UNION_5B = "union_5b"    # 5 bytes: 4B union offset + 1B detect
    PROTOCOL_NMEA_TEXT = "nmea_text"  # NMEA text sentence ($NAV,...)

    # Union data types for offset
    UNION_TYPE_FLOAT = "float"  # 32-bit float IEEE-754 Little-Endian
    UNION_TYPE_INT32 = "int32"  # 32-bit signed int Little-Endian

    # Standard state integer mappings for STM32
    STATE_LOST = 0
    STATE_TRACKING = 1
    STATE_LOW_CONF = 2
    STATE_NO_MODEL = 3

    def __init__(
        self,
        port: str = "/dev/ttySTM32",
        baudrate: int = 115200,
        send_rate_hz: float = 10.0,  # 10 Hz = 0.1s interval (mỗi 0.1s)
        is_enabled: bool = False,
        protocol: str = PROTOCOL_UNION_5B,
        union_type: str = UNION_TYPE_FLOAT
    ):
        self._lock = threading.RLock()
        self.port: str = port
        self.baudrate: int = baudrate
        self.send_rate_hz: float = send_rate_hz
        self.min_send_interval_sec: float = 1.0 / max(0.1, send_rate_hz)
        self.protocol: str = protocol
        self.union_type: str = union_type

        self.ser: Optional[serial.Serial] = None
        self.is_connected: bool = False
        self.is_enabled: bool = is_enabled
        self.last_error: str = ""
        self.packets_sent_count: int = 0
        self.errors_count: int = 0
        self.last_packet_str: str = ""
        self.last_packet_bytes: bytes = b""
        self.last_send_time: float = 0.0

        # Auto-reconnection state
        self.last_reconnect_attempt: float = 0.0
        self.reconnect_interval_sec: float = 2.0

    @property
    def last_sent_packet(self) -> str:
        return self.last_packet_str

    @staticmethod
    def list_available_ports() -> List[str]:
        """Scans for available serial ports, placing ttySTM32 at the top."""
        ports = []
        # Check persistent symlinks first
        for p in ("/dev/ttySTM32", "/dev/stm32"):
            if os.path.exists(p) and p not in ports:
                ports.append(p)

        # Check standard USB & Pi hardware serial devices
        for p in sorted(glob.glob("/dev/ttyUSB*")):
            if p not in ports:
                ports.append(p)
        for p in sorted(glob.glob("/dev/ttyACM*")):
            if p not in ports:
                ports.append(p)
        for p in ("/dev/ttyAMA0", "/dev/ttyAMA10", "/dev/serial0"):
            if os.path.exists(p) and p not in ports:
                ports.append(p)

        if not ports:
            ports = ["/dev/ttySTM32", "/dev/ttyUSB0"]
        return ports

    def _create_serial_instance(self) -> serial.Serial:
        if not os.path.exists(self.port):
            avail = self.list_available_ports()
            for candidate in avail:
                if os.path.exists(candidate):
                    self.port = candidate
                    break
        ser = serial.Serial()
        ser.port = self.port
        ser.baudrate = self.baudrate
        ser.timeout = 0.05
        ser.write_timeout = 0.05
        ser.rtscts = False
        ser.dsrdtr = False
        ser.open()
        return ser

    def connect(self, port: Optional[str] = None, baudrate: Optional[int] = None) -> bool:
        """Opens serial port connection with graceful fallback across candidate ports."""
        with self._lock:
            if port is not None:
                self.port = port
            if baudrate is not None:
                self.baudrate = baudrate

            self.disconnect()

            # Build candidate ports to attempt connection
            candidates = [self.port]
            if self.port in ("/dev/ttySTM32", "/dev/stm32"):
                for p in sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*")) + ["/dev/ttyAMA0", "/dev/ttyAMA10", "/dev/serial0"]:
                    if p not in candidates and os.path.exists(p):
                        candidates.append(p)
            elif self.port.startswith("/dev/ttyUSB"):
                if os.path.exists("/dev/ttySTM32") and "/dev/ttySTM32" not in candidates:
                    candidates.insert(0, "/dev/ttySTM32")

            last_err = ""
            for p in candidates:
                try:
                    self.port = p
                    self.ser = self._create_serial_instance()
                    self.is_connected = True
                    self.last_error = ""
                    return True
                except Exception as e:
                    last_err = str(e)

            self.ser = None
            self.is_connected = False
            self.last_error = last_err
            self.errors_count += 1
            return False

    def disconnect(self):
        """Closes serial connection safely."""
        with self._lock:
            if self.ser is not None:
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None
            self.is_connected = False

    def format_union_5b_packet(
        self,
        offset_px: Optional[float],
        is_detected: Union[bool, int],
        union_type: Optional[str] = None
    ) -> Tuple[bytes, str]:
        """Constructs a 5-byte binary packet:
        - Bytes 0..3: 4 bytes from union (lateral offset in px: negative=left, positive=right)
        - Byte 4: 1 byte flag (0: run, 1: stop, 2: turn left, etc.)

        Returns (packet_bytes, display_string)
        """
        u_type = union_type or self.union_type
        off = float(offset_px) if offset_px is not None else 0.0
        if (isinstance(is_detected, int) or (np is not None and isinstance(is_detected, np.integer))) and not isinstance(is_detected, bool):
            det_val = int(is_detected) & 0xFF
        else:
            det_val = 1 if is_detected else 0

        if u_type == self.UNION_TYPE_INT32:
            offset_bytes = struct.pack('<i', int(round(off)))
        else:
            offset_bytes = struct.pack('<f', off)

        detect_byte = bytes([det_val])
        packet_bytes = offset_bytes + detect_byte

        hex_repr = " ".join(f"{b:02X}" for b in packet_bytes)
        display_str = f"[{hex_repr}] (Offset: {off:+.1f}px, Detect: {det_val})"
        return packet_bytes, display_str

    def format_packet(
        self,
        state_code: int,
        offset_px: Optional[float],
        confidence_pct: float,
        target_x: Optional[int],
        target_y: Optional[int]
    ) -> str:
        """Constructs $NAV,<state>,<offset>,<conf>,<tx>,<ty>*<CHK> packet."""
        off_val = f"{offset_px:.1f}" if offset_px is not None else "0.0"
        tx_val = str(target_x) if target_x is not None else "0"
        ty_val = str(target_y) if target_y is not None else "0"
        conf_val = f"{confidence_pct:.1f}"

        # Payload without leading '$' and trailing '*'
        payload = f"NAV,{state_code},{off_val},{conf_val},{tx_val},{ty_val}"
        chk = calculate_xor_checksum(payload)
        return f"${payload}*{chk}\r\n"

    def send_nav_update(
        self,
        state_code: int = STATE_LOST,
        offset_px: Optional[float] = None,
        confidence_pct: float = 0.0,
        target_x: Optional[int] = None,
        target_y: Optional[int] = None,
        is_detected: Optional[Union[bool, int]] = None
    ) -> bool:
        """Thread-safe rate-limited transmission to STM32 (every 0.1s by default)."""
        if not self.is_enabled:
            return False

        now = time.perf_counter()
        # Rate-limiting: only send if interval (e.g. 0.1s) has elapsed
        if (now - self.last_send_time) < self.min_send_interval_sec:
            return False

        if is_detected is None:
            is_detected = (state_code in (self.STATE_TRACKING, self.STATE_LOW_CONF) and offset_px is not None)

        if self.protocol == self.PROTOCOL_UNION_5B:
            packet_bytes, display_str = self.format_union_5b_packet(offset_px, is_detected)
            data_to_send = packet_bytes
            self.last_packet_str = display_str
            self.last_packet_bytes = packet_bytes
        else:
            packet_str = self.format_packet(state_code, offset_px, confidence_pct, target_x, target_y)
            data_to_send = packet_str.encode("ascii")
            self.last_packet_str = packet_str.strip()
            self.last_packet_bytes = data_to_send

        with self._lock:
            if not self.is_connected or self.ser is None:
                # Attempt background reconnect
                if (now - self.last_reconnect_attempt) > self.reconnect_interval_sec:
                    self.last_reconnect_attempt = now
                    try:
                        self.ser = self._create_serial_instance()
                        self.is_connected = True
                        self.last_error = ""
                    except Exception as e:
                        self.ser = None
                        self.is_connected = False
                        self.last_error = str(e)
                        self.errors_count += 1
                        return False
                else:
                    return False

            try:
                self.ser.write(data_to_send)
                self.last_send_time = now
                self.packets_sent_count += 1
                return True
            except Exception as e:
                self.is_connected = False
                self.last_error = str(e)
                self.errors_count += 1
                try:
                    if self.ser:
                        self.ser.close()
                except Exception:
                    pass
                self.ser = None
                return False

    def send_union_update(self, offset_px: Optional[float], is_detected: Union[bool, int]) -> bool:
        """Convenience method specifically for sending the 5-byte union packet."""
        flag_val = int(is_detected) if (isinstance(is_detected, int) and not isinstance(is_detected, bool)) else (1 if is_detected else 0)
        return self.send_nav_update(
            state_code=self.STATE_TRACKING if flag_val != 0 else self.STATE_LOST,
            offset_px=offset_px,
            is_detected=is_detected
        )

    def read_telemetry(self) -> Optional[dict]:
        """Đọc và phân tích gói tin phản hồi trạng thái từ STM32 ($RC,auto_forward,stm_qr_state)."""
        with self._lock:
            if not self.is_connected or self.ser is None:
                return None
            try:
                if self.ser.in_waiting > 0:
                    raw = self.ser.read(self.ser.in_waiting)
                    if not hasattr(self, "_rx_buf"):
                        self._rx_buf = bytearray()
                    self._rx_buf.extend(raw)

                    # Giữ bộ đệm không quá 512 bytes
                    if len(self._rx_buf) > 512:
                        self._rx_buf = self._rx_buf[-256:]

                    latest_dict = None
                    while b"\n" in self._rx_buf:
                        idx = self._rx_buf.index(b"\n")
                        line_bytes = self._rx_buf[:idx].strip()
                        del self._rx_buf[:idx + 1]

                        try:
                            line = line_bytes.decode("ascii", errors="ignore").strip()
                            if line.startswith("$RC,"):
                                parts = line.split(",")
                                if len(parts) >= 3:
                                    auto_fwd = int(parts[1])
                                    qr_state = int(parts[2])
                                    latest_dict = {
                                        "auto_forward": auto_fwd,
                                        "stm_qr_state": qr_state
                                    }
                        except Exception:
                            pass

                    return latest_dict
            except Exception:
                pass
            return None
