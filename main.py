"""
Main entry point for RobotNongNghiep Path Detection Application.
Supports both interactive GUI (PyQt5/PySide6) and headless automated verification mode.
Supports DUAL MODES: AI (YOLO Segmentation) and Image Processing (CIELAB Color Model).

Usage:
    ./start.sh
    python main.py
    python main.py --mode color
    python main.py --mode ai
    python main.py --headless --mode ai --frames 60
    python main.py --headless --mode color --frames 60
"""

import argparse
import json
import os
import sys
import time
import cv2
import numpy as np

from src.ui_compat import QApplication, QtCore, QtGui, QColor, QT_BINDING
from src.main_window import MainWindow, draw_dashed_polyline
from src.video_reader import VideoReader
from src.yolo_segmenter import YoloSegmenter
from src.color_model import ColorModel
from src.path_detector import PathDetector, TrackingState
from src.profile_manager import ProfileManager, PROJECT_ROOT
from src.uart_controller import UartController


def run_headless_pipeline(
    video_path: str,
    output_dir: str,
    mode: str = "ai",
    model_path: str = "",
    conf: float = 0.25,
    imgsz: int = 416,
    device: str = "cpu",
    center_method: str = "box",
    num_frames_to_test: int = 60,
    uart_enabled: bool = False,
    uart_port: str = "/dev/ttySTM32",
    uart_baud: int = 115200,
    uart_protocol: str = UartController.PROTOCOL_UNION_5B,
    uart_union_type: str = UartController.UNION_TYPE_FLOAT,
    profile_path: str = ""
) -> bool:
    """Headless batch execution for automated verification, telemetry collection, and proof artifact generation."""
    is_ai = (mode.lower() in ("ai", "yolo"))
    mode_name = "AI (YOLO Segmentation)" if is_ai else "Xử lý ảnh (CIELAB Color Model)"

    print(f"\n========================================================")
    print(f"[Headless Mode] Starting automated pipeline verification")
    print(f"Mode  : {mode_name}")
    print(f"Video : {video_path}")
    if is_ai:
        print(f"Model : {model_path} (imgsz={imgsz}, conf={conf}, device={device})")
    if uart_enabled:
        print(f"UART  : Enabled -> {uart_port} @ {uart_baud} bps (Protocol: {uart_protocol}, Union: {uart_union_type})")
    print(f"Output: {output_dir}")
    print(f"========================================================")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize Video Reader
    reader = VideoReader(video_path)
    if not reader.is_opened:
        print(f"[Headless Error] Could not open video: {video_path}")
        return False

    print(f"[Headless] Video opened: {reader.width}x{reader.height}, {reader.fps:.1f} FPS, {reader.total_frames} frames.")

    # 2. Initialize Models based on Mode
    segmenter = None
    color_model = None
    detector = PathDetector()

    # Load default profile if exists
    default_prof = profile_path or os.path.join(ProfileManager.DEFAULT_PROFILE_DIR, "default_profile.json")
    prof_data = None
    if os.path.exists(default_prof):
        print(f"[Headless] Loading profile settings from: {default_prof}")
        prof_data = ProfileManager.load_profile(default_prof)
        if prof_data and "detector" in prof_data:
            detector.from_dict(prof_data["detector"])
    detector.center_method = center_method

    if is_ai:
        segmenter = YoloSegmenter(
            model_path=model_path,
            conf_threshold=conf,
            imgsz=imgsz,
            device=device
        )
        if not segmenter.is_loaded:
            print(f"[Headless Error] Failed to load YOLO model: {segmenter.load_error}")
            return False
        print(f"[Headless] YOLO model loaded: {segmenter.class_names} on {segmenter.device}")
    else:
        color_model = ColorModel()
        if prof_data and "color_model" in prof_data:
            color_model.from_dict(prof_data["color_model"])

        # If still not trained, sample from frame 0
        if not color_model.is_trained:
            ret, frame0, _, _ = reader.read_frame()
            if ret and frame0 is not None:
                H0, W0 = frame0.shape[:2]
                rx, ry, rw, rh = detector.compute_roi((H0, W0))
                # Sample representative path from ROI center
                p_rect = (rx + int(rw * 0.25), ry + int(rh * 0.20), max(10, int(rw * 0.50)), max(10, int(rh * 0.55)))
                color_model.add_patch(frame0, p_rect, "path", label="Bạt ROI chuẩn")
                color_model.train(frame0)
                print(f"[Headless] Auto-sampled initial color patch from frame 0: {p_rect}")
            reader.seek(0)

    # Initialize UART if requested
    uart = None
    if uart_enabled:
        uart = UartController(
            port=uart_port,
            baudrate=uart_baud,
            send_rate_hz=10.0,
            is_enabled=True,
            protocol=uart_protocol,
            union_type=uart_union_type
        )
        ok = uart.connect()
        if ok:
            print(f"[Headless] UART connected to {uart.port} ({uart.baudrate} bps)")
        else:
            print(f"[Headless Warning] UART connect failed: {uart.last_error}")

    processed_count = 0
    states_count = {}
    path_confidences = []
    lat_offsets = []
    inference_times_ms = []

    test_indices = {0, 10, 20, 30, 40, 50, num_frames_to_test - 1}

    while processed_count < num_frames_to_test:
        ret, frame, f_idx, ts = reader.read_frame()
        if not ret or frame is None:
            break

        H, W = frame.shape[:2]
        cam_cx = W / 2.0
        roi_rect = detector.compute_roi((H, W))

        t0 = time.perf_counter()
        if is_ai:
            yolo_res = segmenter.predict(
                frame_bgr=frame,
                conf=conf,
                imgsz=imgsz,
                roi_rect=roi_rect
            )
            det_res = detector.process(
                frame_bgr=frame,
                mask_or_score=yolo_res.chosen_mask,
                is_source_valid=yolo_res.is_valid
            )
            score_map = None
        else:
            yolo_res = None
            if color_model.is_trained:
                score_map = color_model.score_frame(frame, roi_rect)
                is_valid = True
            else:
                score_map = np.zeros((roi_rect[3], roi_rect[2]), dtype=np.uint8)
                is_valid = False

            det_res = detector.process(
                frame_bgr=frame,
                mask_or_score=score_map,
                is_source_valid=is_valid
            )

        t_infer = (time.perf_counter() - t0) * 1000.0
        inference_times_ms.append(t_infer)

        # UART transmission if enabled
        if uart and uart.is_enabled:
            state_code = UartController.STATE_LOST
            if det_res.state == TrackingState.TRACKING:
                state_code = UartController.STATE_TRACKING
            elif det_res.state == TrackingState.LOW_CONFIDENCE:
                state_code = UartController.STATE_LOW_CONF
            elif det_res.state == TrackingState.NO_MODEL:
                state_code = UartController.STATE_NO_MODEL

            target_pt = det_res.path_center if (det_res.center_method == "path" and det_res.path_center is not None) else det_res.box_center
            tx = int(target_pt[0]) if target_pt else None
            ty = int(target_pt[1]) if target_pt else None

            is_detected = bool(
                (det_res.state == TrackingState.TRACKING or 
                 (det_res.state == TrackingState.LOW_CONFIDENCE and det_res.lateral_offset_px is not None))
            )

            uart.send_nav_update(
                state_code=state_code,
                offset_px=det_res.lateral_offset_px,
                confidence_pct=det_res.confidence,
                target_x=tx,
                target_y=ty,
                is_detected=is_detected
            )

        states_count[det_res.state.value] = states_count.get(det_res.state.value, 0) + 1
        path_confidences.append(det_res.confidence)
        if det_res.lateral_offset_px is not None:
            lat_offsets.append(det_res.lateral_offset_px)

        # Export sample proof frames
        if f_idx in test_indices or processed_count % 20 == 0:
            rx, ry, rw, rh = roi_rect

            # 1. View 1: Original + ROI
            v1 = frame.copy()
            cv2.rectangle(v1, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
            cv2.putText(v1, f"ROI [{rw}x{rh}]", (rx + 8, ry + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.line(v1, (int(cam_cx), ry), (int(cam_cx), ry + rh), (200, 200, 200), 1)

            # 2. View 2: Overlay (Strictly clipped inside ROI)
            v2 = frame.copy()
            if det_res.binary_mask is not None and np.any(det_res.binary_mask > 0) and rw > 0 and rh > 0:
                full_mask = np.zeros((H, W), dtype=bool)
                full_mask[ry:ry + rh, rx:rx + rw] = (det_res.binary_mask > 0)
                tint_color = np.array([210, 220, 20], dtype=np.float32) if is_ai else np.array([0, 220, 220], dtype=np.float32)
                overlay = v2.copy()
                overlay[full_mask] = (0.55 * overlay[full_mask] + 0.45 * tint_color).astype(np.uint8)
                cv2.addWeighted(overlay, 0.70, v2, 0.30, 0, v2)

            cv2.rectangle(v2, (rx, ry), (rx + rw, ry + rh), (100, 180, 255), 1)
            cv2.line(v2, (int(cam_cx), ry), (int(cam_cx), ry + rh), (240, 240, 240), 1)

            # Centering Overlay (Path Centerline vs Bounding Box Midpoint)
            if det_res.center_method == "path":
                # Fitted Centerline (Solid Red)
                if det_res.centerline_points:
                    pts_arr = np.array(det_res.centerline_points, dtype=np.int32)
                    cv2.polylines(v2, [pts_arr], False, (0, 0, 255), 3, cv2.LINE_AA)

                # Extrapolated Centerline (Dashed Magenta)
                if det_res.extrapolated_points:
                    draw_dashed_polyline(v2, det_res.extrapolated_points, (255, 0, 255), 2, dash_len=8, gap_len=6)

                # Reference Scanline Slice (Cyan/Orange)
                if det_res.ref_y_global is not None:
                    ref_s = next((s for s in det_res.scanlines if s.y_global == det_res.ref_y_global and s.is_valid), None)
                    if ref_s and ref_s.x_left is not None and ref_s.x_right is not None:
                        cv2.line(v2, (ref_s.x_left, det_res.ref_y_global), (ref_s.x_right, det_res.ref_y_global), (0, 220, 255), 2)
                        cv2.circle(v2, (ref_s.x_left, det_res.ref_y_global), 5, (0, 140, 255), -1)
                        cv2.circle(v2, (ref_s.x_right, det_res.ref_y_global), 5, (0, 140, 255), -1)

                # Path Center Point & Deviation Arrow
                if det_res.path_center is not None:
                    px, py = det_res.path_center
                    cv2.circle(v2, (px, py), 8, (0, 255, 255), -1)
                    cv2.circle(v2, (px, py), 12, (0, 0, 255), 2)
                    cv2.putText(v2, f"Tam path ({px}, {py})", (min(W - 190, px + 15), py + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                    if det_res.lateral_offset_px is not None:
                        cv2.circle(v2, (int(cam_cx), py), 5, (255, 255, 255), -1)
                        cv2.arrowedLine(v2, (int(cam_cx), py), (px, py), (0, 255, 255), 2, tipLength=0.20)
                        lbl_x = max(20, min(W - 280, (int(cam_cx) + px) // 2 - 60))
                        cv2.putText(v2, det_res.offset_direction, (lbl_x, py - 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if det_res.bounding_box is not None:
                    bx, by, bw, bh = det_res.bounding_box
                    cv2.rectangle(v2, (bx, by), (bx + bw, by + bh), (0, 200, 0), 1)

            else:
                # Bounding Box Midpoint mode
                if det_res.bounding_box is not None and det_res.box_center is not None:
                    bx, by, bw, bh = det_res.bounding_box
                    cx, cy = det_res.box_center

                    # 1. Bounding box (Green, 2px)
                    cv2.rectangle(v2, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                    cv2.putText(v2, f"Vung loi di [{bw}x{bh} px]", (bx + 8, max(25, by + 22)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                    # 2. Straight vertical center axis along box
                    cv2.line(v2, (cx, by), (cx, by + bh), (0, 0, 255), 2, cv2.LINE_AA)

                    # 3. Center point
                    cv2.circle(v2, (cx, cy), 8, (0, 255, 255), -1)
                    cv2.circle(v2, (cx, cy), 12, (0, 0, 255), 2)
                    cv2.putText(v2, f"Tam box ({cx}, {cy})", (min(W - 160, cx + 15), cy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                    # 4. Camera reference point & Deviation arrow at y = cy
                    if det_res.lateral_offset_px is not None:
                        cv2.circle(v2, (int(cam_cx), cy), 5, (255, 255, 255), -1)
                        cv2.arrowedLine(v2, (int(cam_cx), cy), (cx, cy), (0, 255, 255), 2, tipLength=0.20)
                        lbl_x = max(20, min(W - 220, (int(cam_cx) + cx) // 2 - 50))
                        cv2.putText(v2, det_res.offset_direction, (lbl_x, cy - 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            method_tag = "PATH" if det_res.center_method == "path" else "BOX"
            badge_text = f"Mode: {mode.upper()} [{method_tag}] | State: {det_res.state.value} | Conf: {det_res.confidence:.1f}%"
            cv2.putText(v2, badge_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            # 3. View 3: Binary Mask (Clipped strictly inside ROI)
            v3 = np.zeros((H, W), dtype=np.uint8)
            if det_res.binary_mask is not None and det_res.binary_mask.shape == (rh, rw):
                v3[ry:ry + rh, rx:rx + rw] = det_res.binary_mask
            cv2.rectangle(v3, (rx, ry), (rx + rw, ry + rh), (100, 100, 255), 1)

            # 4. View 4: Mode Dependent (Debug / Heatmap)
            if is_ai and yolo_res:
                v4 = (frame // 2).astype(np.uint8)
                inst_colors = [(255, 130, 0), (255, 0, 255), (0, 255, 255)]
                for inst in yolo_res.all_instances:
                    col = inst_colors[inst.instance_id % len(inst_colors)]
                    mask_i = inst.mask > 0
                    if np.any(mask_i):
                        v4[mask_i] = (0.5 * v4[mask_i] + 0.5 * np.array(col, dtype=np.float32)).astype(np.uint8)
                    bx, by, bw, bh = inst.bbox
                    cv2.rectangle(v4, (bx, by), (bx + bw, by + bh), col, 2)
                    cv2.putText(v4, f"#{inst.instance_id} ({inst.confidence:.2f})", (bx + 4, max(18, by + 16)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

                # Draw scanlines in path mode
                if det_res.center_method == "path" and det_res.scanlines:
                    for s in det_res.scanlines:
                        if s.is_valid and s.x_left is not None and s.x_right is not None and s.x_mid is not None:
                            cv2.circle(v4, (s.x_left, s.y_global), 3, (0, 255, 0), -1)
                            cv2.circle(v4, (s.x_right, s.y_global), 3, (0, 255, 0), -1)
                            cv2.circle(v4, (s.x_mid, s.y_global), 3, (0, 255, 255), -1)
                        elif s.left_clipped and s.x_left is not None:
                            cv2.circle(v4, (s.x_left, s.y_global), 3, (0, 0, 255), -1)
                        elif s.right_clipped and s.x_right is not None:
                            cv2.circle(v4, (s.x_right, s.y_global), 3, (0, 0, 255), -1)

                target_pt = det_res.path_center if (det_res.center_method == "path" and det_res.path_center is not None) else det_res.box_center
                if target_pt is not None:
                    cx, cy = target_pt
                    cv2.circle(v4, (cx, cy), 6, (0, 255, 255), -1)
            else:
                full_score = np.zeros((H, W), dtype=np.uint8)
                if score_map is not None and score_map.shape == (rh, rw):
                    full_score[ry:ry + rh, rx:rx + rw] = score_map
                v4 = cv2.applyColorMap(full_score, cv2.COLORMAP_JET)
                v4[0:ry, :] = v4[0:ry, :] // 3
                v4[ry + rh:, :] = v4[ry + rh:, :] // 3

            # Save proof images
            out_v1_path = os.path.join(output_dir, f"proof_f{f_idx:04d}_orig.jpg")
            out_v2_path = os.path.join(output_dir, f"proof_f{f_idx:04d}_overlay.jpg")
            out_v3_path = os.path.join(output_dir, f"proof_f{f_idx:04d}_mask.jpg")
            v4_name = "debug" if is_ai else "heatmap"
            out_v4_path = os.path.join(output_dir, f"proof_f{f_idx:04d}_{v4_name}.jpg")

            cv2.imwrite(out_v1_path, v1)
            cv2.imwrite(out_v2_path, v2)
            cv2.imwrite(out_v3_path, v3)
            cv2.imwrite(out_v4_path, v4)

            # Save proof telemetry JSON
            telemetry_data = {
                "frame_index": f_idx,
                "timestamp_sec": ts,
                "mode": mode,
                "state": det_res.state.value,
                "path_confidence_pct": round(det_res.confidence, 1),
                "lateral_offset_px": round(det_res.lateral_offset_px, 1) if det_res.lateral_offset_px is not None else None,
                "offset_direction": det_res.offset_direction,
                "ref_y_global": det_res.ref_y_global,
                "inference_time_ms": round(t_infer, 1),
                "bounding_box": list(det_res.bounding_box) if det_res.bounding_box else None,
                "box_center": list(det_res.box_center) if det_res.box_center else None,
                "valid_scanlines": det_res.valid_scanlines_count,
                "total_scanlines": det_res.total_scanlines_count
            }
            if is_ai and yolo_res:
                telemetry_data["yolo_instance_confidence"] = round(yolo_res.chosen_confidence, 3) if yolo_res.chosen_confidence else None

            json_path = os.path.join(output_dir, f"proof_f{f_idx:04d}_telemetry.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(telemetry_data, f, indent=2, ensure_ascii=False)

            print(f"[Headless] Frame {f_idx:03d}: State={det_res.state.value} | "
                  f"Offset={det_res.offset_direction} | Infer={t_infer:.1f}ms")

        processed_count += 1

    reader.close()

    avg_infer = np.mean(inference_times_ms) if inference_times_ms else 0.0
    avg_path_conf = np.mean(path_confidences) if path_confidences else 0.0
    fps_estimate = (1000.0 / avg_infer) if avg_infer > 0 else 0.0

    print(f"\n========================================================")
    print(f"[Headless Result Summary - {mode_name}]")
    print(f"Total frames processed : {processed_count}")
    print(f"State breakdown        : {states_count}")
    print(f"Average processing time: {avg_infer:.1f} ms (~{fps_estimate:.1f} FPS)")
    print(f"Average Path Confidence: {avg_path_conf:.1f}%")
    if lat_offsets:
        print(f"Lateral offset mean    : {np.mean(lat_offsets):.1f} px, std: {np.std(lat_offsets):.1f} px")
    if uart:
        print(f"UART packets sent      : {uart.packets_sent_count} (Errors: {uart.errors_count})")
        if uart.last_sent_packet:
            print(f"UART last packet       : {uart.last_sent_packet.strip()}")
        uart.disconnect()
    print(f"Proof artifacts saved to: {output_dir}")
    print(f"========================================================\n")
    return True


def main():
    nhamang_model = os.path.join(PROJECT_ROOT, "models", "nhamang_batden_seg.pt")
    default_model = nhamang_model if os.path.exists(nhamang_model) else os.path.join(PROJECT_ROOT, "models", "farm_path_bootstrap.pt")
    default_output = os.path.join(PROJECT_ROOT, "output")

    parser = argparse.ArgumentParser(description="RobotNongNghiep Path Detection System [Dual-Engine: AI & Color]")
    parser.add_argument("--mode", type=str, default="ai", choices=["ai", "color", "yolo"],
                        help="Detection engine mode: 'ai' (YOLO Segmentation) or 'color' (Image Processing)")
    parser.add_argument("--video", "-v", type=str, default=None, help="Path to input video file")
    parser.add_argument("--model", "-m", type=str, default=default_model, help="Path to YOLO segmentation model (.pt)")
    parser.add_argument("--profile", type=str, default=None, help="Optional JSON profile for model and ROI settings")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=416, help="YOLO input image size (default: 416)")
    try:
        from src.yolo_segmenter import is_cuda_available_and_working
        default_device = "cuda" if is_cuda_available_and_working() else "cpu"
    except Exception:
        default_device = "cpu"
    parser.add_argument("--device", type=str, default=default_device, help=f"Device for YOLO inference: cuda, cpu, or auto (default: {default_device})")
    parser.add_argument("--center-method", type=str, default="box", choices=["box", "path"],
                        help="Centering method: 'box' (Bounding box midpoint) or 'path' (Generated centerline path)")
    parser.add_argument("--uart", action="store_true", help="Enable UART transmission to STM32 controller")
    parser.add_argument("--uart-port", type=str, default="/dev/ttySTM32", help="UART serial port path (default: /dev/ttySTM32)")
    parser.add_argument("--uart-baud", type=int, default=115200, help="UART serial baud rate (default: 115200)")
    parser.add_argument("--uart-protocol", type=str, default="union_5b", choices=["union_5b", "nmea_text"],
                        help="UART protocol: 'union_5b' (5 bytes: 4B union offset + 1B detect, every 0.1s) or 'nmea_text' ($NAV,...)")
    parser.add_argument("--uart-union-type", type=str, default="float", choices=["float", "int32"],
                        help="Union type for offset: 'float' (IEEE-754 4-byte) or 'int32' (signed 32-bit)")
    parser.add_argument("--headless", action="store_true", help="Run in headless verification mode without GUI")
    parser.add_argument("--output", "-o", type=str, default=default_output, help="Output directory for snapshots")
    parser.add_argument("--frames", "-n", type=int, default=60, help="Number of frames to process in headless mode")
    args = parser.parse_args()
    if args.profile and not os.path.isfile(args.profile):
        parser.error(f"Profile does not exist: {args.profile}")

    # Locate default video if not specified
    video_path = args.video
    is_cam = bool(video_path and (str(video_path).strip().isdigit() or str(video_path).strip().startswith("/dev/video")))
    if not video_path:
        nhamang_video = os.path.join(PROJECT_ROOT, "nhamang_20261025_030235.mp4")
        if os.path.isfile(nhamang_video):
            video_path = nhamang_video
        else:
            candidate_dir = os.path.join(PROJECT_ROOT, "Quỳnh Robotics [13-10-2026 20_33]")
            if os.path.exists(candidate_dir):
                files = sorted(os.path.join(candidate_dir, f) for f in os.listdir(candidate_dir) if f.endswith(".mp4"))
                if files:
                    video_path = files[0]

    if args.headless:
        if not video_path or (not is_cam and not os.path.exists(video_path)):
            print("Error: No video file or camera device found for headless mode.")
            sys.exit(1)

        ok = run_headless_pipeline(
            video_path=video_path,
            output_dir=args.output,
            mode=args.mode,
            model_path=args.model,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            center_method=args.center_method,
            num_frames_to_test=args.frames,
            uart_enabled=args.uart,
            uart_port=args.uart_port,
            uart_baud=args.uart_baud,
            uart_protocol=args.uart_protocol,
            uart_union_type=args.uart_union_type,
            profile_path=args.profile or ""
        )
        sys.exit(0 if ok else 1)

    # Desktop GUI mode
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    print(f"[RobotNongNghiep] Starting GUI with {QT_BINDING} in '{args.mode.upper()}' mode (Center: {args.center_method})...")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    sig_timer = QtCore.QTimer()
    sig_timer.timeout.connect(lambda: None)
    sig_timer.start(300)

    # Dark theme palette
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QColor(40, 42, 46))
    palette.setColor(QtGui.QPalette.WindowText, QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.Base, QColor(25, 27, 30))
    palette.setColor(QtGui.QPalette.AlternateBase, QColor(40, 42, 46))
    palette.setColor(QtGui.QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QtGui.QPalette.Text, QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.Button, QColor(50, 52, 56))
    palette.setColor(QtGui.QPalette.ButtonText, QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QtGui.QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    window = MainWindow(
        initial_video_path=video_path,
        initial_model_path=args.model,
        initial_mode=args.mode,
        initial_center_method=args.center_method,
        initial_device=args.device,
        initial_uart=args.uart,
        initial_uart_port=args.uart_port,
        initial_uart_baud=args.uart_baud,
        initial_uart_protocol=args.uart_protocol,
        initial_uart_union_type=args.uart_union_type
    )
    if args.profile:
        window._load_profile_path(args.profile)
    window.show()
    exec_func = getattr(app, "exec", getattr(app, "exec_", None))
    sys.exit(exec_func())


if __name__ == "__main__":
    main()
