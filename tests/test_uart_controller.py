"""
Unit tests for UartController and packet generation.
"""

import unittest
from src.uart_controller import UartController, calculate_xor_checksum


class TestUartController(unittest.TestCase):

    def test_checksum_calculation(self):
        payload = "NAV,1,-12.5,88.0,628,504"
        chk = calculate_xor_checksum(payload)
        self.assertEqual(len(chk), 2)
        # Verify manual XOR
        c = 0
        for ch in payload:
            c ^= ord(ch)
        self.assertEqual(chk, f"{c:02X}")

    def test_packet_formatting(self):
        u = UartController()
        pkt = u.format_packet(
            state_code=UartController.STATE_TRACKING,
            offset_px=-15.3,
            confidence_pct=92.5,
            target_x=625,
            target_y=510
        )
        self.assertTrue(pkt.startswith("$NAV,1,-15.3,92.5,625,510*"))
        self.assertTrue(pkt.endswith("\r\n"))

    def test_packet_formatting_lost_state(self):
        u = UartController()
        pkt = u.format_packet(
            state_code=UartController.STATE_LOST,
            offset_px=None,
            confidence_pct=0.0,
            target_x=None,
            target_y=None
        )
        self.assertTrue(pkt.startswith("$NAV,0,0.0,0.0,0,0*"))
        self.assertTrue(pkt.endswith("\r\n"))

    def test_ports_enumeration(self):
        ports = UartController.list_available_ports()
        self.assertIsInstance(ports, list)
        self.assertGreater(len(ports), 0)
        # /dev/ttySTM32 must be detected on this machine
        self.assertIn("/dev/ttySTM32", ports)

    def test_rate_limiting(self):
        u = UartController(send_rate_hz=10.0)  # 100ms interval (0.1s)
        self.assertAlmostEqual(u.min_send_interval_sec, 0.1, places=3)
        u.is_enabled = False  # Keep disabled so it doesn't actually write
        # When disabled, returns False immediately
        self.assertFalse(u.send_nav_update(1, 0.0, 100.0, 640, 500))

    def test_union_5b_packet_float_left_offset(self):
        import struct
        u = UartController(protocol=UartController.PROTOCOL_UNION_5B, union_type=UartController.UNION_TYPE_FLOAT)
        # Lệch trái 25.5 px (âm), detect = 1
        raw, disp = u.format_union_5b_packet(offset_px=-25.5, is_detected=True)
        self.assertEqual(len(raw), 5)
        # Check byte 0..3 is float -25.5 in Little-Endian
        unpacked_offset = struct.unpack('<f', raw[0:4])[0]
        self.assertAlmostEqual(unpacked_offset, -25.5, places=2)
        # Check byte 4 is 1
        self.assertEqual(raw[4], 1)
        self.assertIn("Detect: 1", disp)

    def test_union_5b_packet_float_right_offset(self):
        import struct
        u = UartController(protocol=UartController.PROTOCOL_UNION_5B, union_type=UartController.UNION_TYPE_FLOAT)
        # Lệch phải 38.25 px (dương), detect = 1
        raw, disp = u.format_union_5b_packet(offset_px=38.25, is_detected=True)
        self.assertEqual(len(raw), 5)
        unpacked_offset = struct.unpack('<f', raw[0:4])[0]
        self.assertAlmostEqual(unpacked_offset, 38.25, places=2)
        self.assertEqual(raw[4], 1)
        self.assertIn("Detect: 1", disp)

    def test_union_5b_packet_float_lost(self):
        import struct
        u = UartController(protocol=UartController.PROTOCOL_UNION_5B, union_type=UartController.UNION_TYPE_FLOAT)
        # Mất đường: offset=0.0, detect = 0
        raw, disp = u.format_union_5b_packet(offset_px=None, is_detected=False)
        self.assertEqual(len(raw), 5)
        unpacked_offset = struct.unpack('<f', raw[0:4])[0]
        self.assertAlmostEqual(unpacked_offset, 0.0, places=2)
        self.assertEqual(raw[4], 0)
        self.assertIn("Detect: 0", disp)

    def test_union_5b_packet_int32(self):
        import struct
        u = UartController(protocol=UartController.PROTOCOL_UNION_5B, union_type=UartController.UNION_TYPE_INT32)
        # Lệch trái -45 px (âm), int32
        raw, disp = u.format_union_5b_packet(offset_px=-45.0, is_detected=True)
        self.assertEqual(len(raw), 5)
        unpacked_offset = struct.unpack('<i', raw[0:4])[0]
        self.assertEqual(unpacked_offset, -45)
        self.assertEqual(raw[4], 1)


if __name__ == "__main__":
    unittest.main()

