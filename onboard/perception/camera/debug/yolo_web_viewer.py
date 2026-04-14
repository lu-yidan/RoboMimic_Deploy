"""Web viewer for all YOLO detections from a RealSense camera.

This is a debugging utility for real-robot bring-up:
- detects all YOLO classes by default
- draws every box with class name and confidence
- streams the annotated frames via MJPEG in a browser

It intentionally does not publish DDS or apply target-selection logic.

Usage:
    bash onboard/perception/camera/run_yolo_web.sh
    bash onboard/perception/camera/run_yolo_web.sh --list-cameras
    bash onboard/perception/camera/run_yolo_web.sh --camera-serial 244622070281
    bash onboard/perception/camera/run_yolo_web.sh --conf-threshold 0.15 --imgsz 320
    bash onboard/perception/camera/run_yolo_web.sh --class-filter bottle,cup,vase

What to look for on the real robot:
1. Whether the target object gets any box at all.
2. Whether the class label is stable or jumps between similar classes.
3. Whether clutter produces stronger boxes than the actual target.
4. Whether detection quality drops at specific chest-camera angles or distances.
"""

import sys
import warnings
import numpy as _np_compat

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    if not hasattr(_np_compat, "bool"):
        _np_compat.bool = bool
    if not hasattr(_np_compat, "int"):
        _np_compat.int = int
    if not hasattr(_np_compat, "float"):
        _np_compat.float = float
    if not hasattr(_np_compat, "object"):
        _np_compat.object = object
del _np_compat

from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent.absolute()))

import argparse
import socket
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


class _FPS:
    def __init__(self, window=30):
        self._timestamps = []
        self._window = window

    def tick(self):
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) > self._window:
            self._timestamps.pop(0)

    @property
    def fps(self):
        if len(self._timestamps) < 2:
            return 0.0
        return (len(self._timestamps) - 1) / (self._timestamps[-1] - self._timestamps[0])


def _parse_names(raw_values):
    if not raw_values:
        return None
    parsed = []
    for raw in raw_values:
        for item in raw.split(","):
            name = item.strip().lower()
            if name:
                parsed.append(name)
    return parsed or None


def _start_camera_pipeline(args):
    ctx = rs.context()
    devs = ctx.query_devices()
    all_sns = [d.get_info(rs.camera_info.serial_number) for d in devs]
    all_names = [d.get_info(rs.camera_info.name) for d in devs]

    print("[INFO] Connected RealSense devices:")
    for idx, (sn, nm) in enumerate(zip(all_sns, all_names)):
        print(f"  [{idx}] serial={sn}  {nm}")

    if args.list_cameras:
        return None, None, None
    if len(all_sns) == 0:
        raise RuntimeError("No RealSense device found.")

    serial = args.camera_serial or all_sns[0]
    print(f"[INFO] YOLO web camera -> serial {serial}")

    pipeline = rs.pipeline()
    fps_tries = [(60, 60), (30, 30), (15, 15)]
    last_err = None
    for color_fps, depth_fps in fps_tries:
        rs_cfg = rs.config()
        rs_cfg.enable_device(serial)
        rs_cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, color_fps)
        rs_cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, depth_fps)
        for attempt in range(2):
            try:
                print(
                    f"[INFO] Starting RealSense serial={serial} "
                    f"(color {color_fps} Hz, depth {depth_fps} Hz, attempt {attempt + 1})..."
                )
                profile = pipeline.start(rs_cfg)
            except RuntimeError as exc:
                msg = str(exc).lower()
                if "resolve" in msg or "couldn't" in msg:
                    last_err = exc
                    print(f"[WARN] Stream profile not supported: {exc}")
                    break
                raise
            try:
                pipeline.wait_for_frames(timeout_ms=5000)
                print(
                    f"[INFO] RealSense pipeline OK for serial={serial} "
                    f"(color {color_fps} Hz, depth {depth_fps} Hz)"
                )
                return pipeline, profile, serial
            except RuntimeError:
                print("[WARN] Frame timeout. Performing hardware reset...")
                pipeline.stop()
                reset_ctx = rs.context()
                matched = [d for d in reset_ctx.query_devices()
                           if d.get_info(rs.camera_info.serial_number) == serial]
                if not matched:
                    raise RuntimeError(f"RealSense serial {serial} disappeared during reset.")
                matched[0].hardware_reset()
                time.sleep(3)

    msg = f"RealSense failed to start for serial={serial}. Tried FPS pairs: {fps_tries}."
    if last_err is not None:
        msg += f" Last resolve error: {last_err!r}"
    raise RuntimeError(msg)


def _start_mjpeg_server(port):
    import http.server
    import socketserver

    mjpeg_frame = [None]
    mjpeg_lock = threading.Lock()

    class _MJPEGHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/":
                self.send_response(302)
                self.send_header("Location", "/stream")
                self.end_headers()
                return
            if self.path != "/stream":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                last_sent = None
                while True:
                    with mjpeg_lock:
                        jpg = mjpeg_frame[0]
                    if jpg is None or jpg is last_sent:
                        time.sleep(0.02)
                        continue
                    last_sent = jpg
                    self.wfile.write(
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                    )
            except Exception:
                pass

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer(("0.0.0.0", port), _MJPEGHandler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, mjpeg_frame, mjpeg_lock


def _get_stream_url(port, path="/stream"):
    host = "127.0.0.1"
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("192.168.123.1", 1))
        host = sock.getsockname()[0]
    except OSError:
        try:
            host = socket.gethostbyname(socket.gethostname())
        except OSError:
            pass
    finally:
        if sock is not None:
            sock.close()
    return f"http://{host}:{port}{path}"


def _resolve_filter_ids(model_names, filter_names):
    if filter_names is None:
        return None
    if isinstance(model_names, dict):
        pairs = model_names.items()
    else:
        pairs = enumerate(model_names)
    by_name = {str(name).lower(): int(class_id) for class_id, name in pairs}
    ids = []
    missing = []
    for name in filter_names:
        if name in by_name:
            ids.append(by_name[name])
        else:
            missing.append(name)
    if missing:
        raise ValueError(
            f"Class filters not found in model labels: {missing}. "
            f"Available labels include: {sorted(list(by_name.keys()))[:20]}"
        )
    return set(ids)


def _color_for_class(class_id):
    rng = np.random.default_rng(class_id)
    color = rng.integers(64, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def main():
    parser = argparse.ArgumentParser(
        description="RealSense + YOLO web viewer for all detected objects"
    )
    parser.add_argument("--model", default="onboard/perception/camera/models/yolo11m.pt")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-serial", default=None,
                        help="RealSense serial to use.")
    parser.add_argument("--list-cameras", action="store_true",
                        help="Print connected RealSense serials and exit.")
    parser.add_argument("--port", type=int, default=8081,
                        help="MJPEG web port. Default avoids colliding with other camera tools.")
    parser.add_argument("--conf-threshold", type=float, default=0.20)
    parser.add_argument("--max-det", type=int, default=30,
                        help="Maximum number of detections to draw per frame.")
    parser.add_argument("--class-filter", action="append", dest="class_filters",
                        help="Optional class labels to keep. Repeat or pass comma-separated names.")
    parser.add_argument("--line-width", type=int, default=2)
    args = parser.parse_args()

    pipeline, _profile, serial = _start_camera_pipeline(args)
    if args.list_cameras:
        return

    filter_names = _parse_names(args.class_filters)
    if filter_names:
        print(f"[INFO] Applying class filter: {filter_names}")
    else:
        print("[INFO] Showing all YOLO classes.")

    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = args.model
    if model_path.endswith(".pt"):
        engine_path = model_path.replace(".pt", ".engine")
        if __import__("os").path.exists(engine_path):
            model_path = engine_path
            print(f"[INFO] TensorRT engine found, using: {engine_path}")
        else:
            print(f"[INFO] No .engine found at {engine_path}, using .pt")

    print(f"[INFO] Loading YOLO model: {model_path} (device={device})")
    model = YOLO(model_path)
    is_trt = str(model_path).endswith(".engine")
    infer_kw = {} if is_trt else dict(device=device, half=True)

    filter_ids = _resolve_filter_ids(model.names, filter_names)
    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    print("[INFO] YOLO warming up...")
    for idx in range(3):
        t0 = time.perf_counter()
        model(dummy, verbose=False, **infer_kw)
        print(f"[INFO]   warmup[{idx}]: {(time.perf_counter() - t0) * 1000:.1f}ms")

    httpd, mjpeg_frame, mjpeg_lock = _start_mjpeg_server(args.port)
    print(f"[INFO] MJPEG stream started -> open {_get_stream_url(args.port)}")
    print("[INFO] Camera running. Press Ctrl+C to stop.")

    yolo_fps = _FPS()
    try:
        while True:
            frames = pipeline.wait_for_frames()
            cf = frames.get_color_frame()
            if not cf:
                continue

            color = np.asanyarray(cf.get_data()).copy()
            orig_h, orig_w = color.shape[:2]
            color_small = cv2.resize(color, (args.imgsz, args.imgsz))
            sx = orig_w / args.imgsz
            sy = orig_h / args.imgsz

            results = model(color_small, conf=args.conf_threshold, max_det=args.max_det,
                            verbose=False, **infer_kw)

            vis = color.copy()
            det_lines = []
            det_count = 0

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    if filter_ids is not None and class_id not in filter_ids:
                        continue
                    conf = float(box.conf[0])
                    x1s, y1s, x2s, y2s = box.xyxy[0]
                    x1 = int(x1s * sx)
                    y1 = int(y1s * sy)
                    x2 = int(x2s * sx)
                    y2 = int(y2s * sy)
                    cls_name = str(model.names[class_id])
                    color_box = _color_for_class(class_id)

                    cv2.rectangle(vis, (x1, y1), (x2, y2), color_box, args.line_width)
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(
                        vis,
                        label,
                        (x1, max(24, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color_box,
                        2,
                    )

                    det_count += 1
                    if len(det_lines) < 5:
                        det_lines.append(label)

            yolo_fps.tick()

            summary = "none" if not det_lines else " | ".join(det_lines)
            print(
                f"\r[YOLO] serial={serial} det={det_count:02d} fps={yolo_fps.fps:4.1f} top={summary}" + " " * 12,
                end="",
                flush=True,
            )

            cv2.putText(vis, f"YOLO {yolo_fps.fps:.1f} fps", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(vis, f"detections {det_count}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
            filter_text = "all classes" if filter_names is None else f"filter: {', '.join(filter_names)}"
            cv2.putText(vis, filter_text, (10, 76),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            _, jpg_buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 65])
            with mjpeg_lock:
                mjpeg_frame[0] = jpg_buf.tobytes()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        pipeline.stop()
        httpd.shutdown()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
