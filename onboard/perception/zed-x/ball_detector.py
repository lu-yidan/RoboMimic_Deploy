"""
ball_detector.py — ZED X + YOLO TensorRT 球体 3D 检测

硬件：Unitree G1，ZED X 安装在胸部（GMSL，Jetson AGX Orin）
模型：YOLO11m / YOLO11n → TensorRT FP16 engine

当前阶段（MJPEG 验证）：
  - 三线程：ZED 采集 / YOLO 推理 / MJPEG 编码
  - 输出球心 body 系坐标（相对 ZED X 光轴）
  - MJPEG HTTP 流，浏览器可访问

TODO (P1):
  - 加 ROS2 _JointListener 订阅 /lowstate（参考 camera/ball_detector.py）
  - 加 BallStatePublisher DDS 发布 rt/ball_state
  - 调用 zed_to_base.transform_point_zed_to_base() 输出 pelvis 系坐标
  - 标定 zed_to_base._ZED_XYZ / _ZED_RPY 胸部安装外参

用法：
    cd RoboMimic_Deploy
    bash onboard/perception/zed-x/run.sh --ball soccer-m --show
    bash onboard/perception/zed-x/run.sh --ball tennis --show
    # 浏览器打开 http://<robot-ip>:8080/stream
"""

import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*torchvision.*")

import argparse
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import pyzed.sl as sl
    HAS_ZED = True
except ImportError:
    HAS_ZED = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# 工程根目录（RoboMimic_Deploy/）
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_THIS_DIR  = Path(__file__).resolve().parent


# ── 球类配置 ─────────────────────────────────────────────────────────────────
BALL_CONFIGS = {
    "tennis":   {"class_id": 0,  "radius": 0.034, "conf": 0.30,
                 "model": str(_THIS_DIR / "weights/yolov8_best.engine")},
    "soccer":   {"class_id": 32, "radius": 0.11,  "conf": 0.20,
                 "model": str(_THIS_DIR / "weights/yolo11n.engine")},
    "soccer-m": {"class_id": 32, "radius": 0.11,  "conf": 0.15,
                 "model": str(_THIS_DIR / "weights/yolo11m.engine")},
}

CONF_THRESHOLD      = 0.30
DEPTH_SAMPLE_RADIUS = 8      # px，中值采样半径（17×17 patch）
DEPTH_MIN           = 0.3    # m
DEPTH_MAX           = 15.0   # m
EMA_ALPHA           = 0.6
EMA_GATE            = 0.5    # m，跳变超过此值时重置 EMA
COAST_FRAMES        = 10


# ── FPS 统计 ──────────────────────────────────────────────────────────────────

class _FPS:
    def __init__(self, window=30):
        self._t = []
        self._w = window

    def tick(self):
        now = time.perf_counter()
        self._t.append(now)
        if len(self._t) > self._w:
            self._t.pop(0)

    @property
    def fps(self):
        if len(self._t) < 2:
            return 0.0
        return (len(self._t) - 1) / (self._t[-1] - self._t[0])


# ── ZED 初始化 ────────────────────────────────────────────────────────────────

def _open_zed():
    if not HAS_ZED:
        raise RuntimeError("pyzed 未安装")

    devices = sl.Camera.get_device_list()
    if not devices:
        raise RuntimeError(
            "未检测到 ZED 设备！请检查 GMSL 线缆与供电，"
            "并执行: sudo systemctl restart zed_x_daemon.service"
        )
    print(f"[ZED] 检测到 {len(devices)} 个设备:")
    for d in devices:
        print(f"  SN={d.serial_number}  型号={d.camera_model}  状态={d.camera_state}")

    zed = sl.Camera()
    # ZED X 支持的分辨率（SVGA 优先，速度最快）：
    #   SVGA   960×600  @ 120fps
    #   HD1200 1920×1200 @ 60fps
    #   HD1080 1920×1080 @ 30fps
    # 注：SDK 5.x 中 PERFORMANCE 是真正的传统立体匹配，
    #     QUALITY/NEURAL 均走 GPU 神经网络，会与 YOLO TRT 争 GPU。
    candidates = [
        (sl.RESOLUTION.SVGA,   120),
        (sl.RESOLUTION.HD1200,  60),
        (sl.RESOLUTION.HD1080,  30),
        (sl.RESOLUTION.HD1200,  30),
    ]

    for res, fps in candidates:
        init = sl.InitParameters()
        init.camera_resolution      = res
        init.camera_fps             = fps
        init.depth_mode             = sl.DEPTH_MODE.PERFORMANCE  # 传统立体匹配，不占 GPU
        init.coordinate_units       = sl.UNIT.METER
        init.depth_minimum_distance = DEPTH_MIN
        init.depth_maximum_distance = DEPTH_MAX
        err = zed.open(init)
        if err == sl.ERROR_CODE.SUCCESS:
            info = zed.get_camera_information()
            cfg  = info.camera_configuration
            r    = cfg.resolution
            print(f"[ZED] 已打开: {r.width}×{r.height} @ {cfg.fps}fps  SN={info.serial_number}")
            return zed, info
        print(f"[ZED] {res} @ {fps}fps 失败: {repr(err)}")

    raise RuntimeError("ZED 所有分辨率均失败，请检查 SDK 版本与相机固件")


# ── YOLO 初始化 ───────────────────────────────────────────────────────────────

def _load_yolo(model_path: str):
    if not HAS_YOLO:
        raise RuntimeError("ultralytics 未安装")
    if model_path.endswith(".pt"):
        engine = model_path.replace(".pt", ".engine")
        if os.path.exists(engine):
            print(f"[YOLO] 找到 TRT engine: {engine}")
            model_path = engine
        else:
            print(f"[YOLO] 未找到 .engine，使用 .pt（建议先导出 TRT）")
    print(f"[YOLO] 加载: {model_path}")
    model  = YOLO(model_path)
    is_trt = model_path.endswith(".engine")
    return model, is_trt


def _warmup(model, imgsz: int, is_trt: bool, n: int = 5):
    import torch
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    kw = dict(imgsz=imgsz)
    if not is_trt:
        kw.update(device="cuda:0" if torch.cuda.is_available() else "cpu", half=True)
    print(f"[YOLO] Warmup {n} 次 (imgsz={imgsz}) ...")
    for i in range(n):
        t = time.perf_counter()
        model(dummy, verbose=False, **kw)
        print(f"  [{i}] {(time.perf_counter()-t)*1000:.1f}ms")
    t0 = time.perf_counter()
    for _ in range(10):
        model(dummy, verbose=False, **kw)
    ms = (time.perf_counter() - t0) / 10 * 1000
    print(f"[YOLO] 稳定推理: {ms:.1f}ms/帧  (上限 {1000/ms:.0f} FPS)")
    return kw


# ── 深度采样 ──────────────────────────────────────────────────────────────────

def _sample_depth(depth_arr: np.ndarray, cx: int, cy: int) -> float:
    """
    ZED X 深度图已对齐左目，(cx, cy) 直接对应左目像素，无需 color→depth 映射。
    取 (2R+1)×(2R+1) patch 中值，返回前表面深度（米），无效时返回 0.0。
    """
    H, W = depth_arr.shape
    x0 = max(0, cx - DEPTH_SAMPLE_RADIUS)
    x1 = min(W, cx + DEPTH_SAMPLE_RADIUS + 1)
    y0 = max(0, cy - DEPTH_SAMPLE_RADIUS)
    y1 = min(H, cy + DEPTH_SAMPLE_RADIUS + 1)
    patch = depth_arr[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > DEPTH_MIN) & (patch < DEPTH_MAX)]
    return float(np.median(valid)) if len(valid) > 0 else 0.0


# ── 3D 反投影 ─────────────────────────────────────────────────────────────────

def _deproject(cx_px, cy_px, depth_m, fx, fy, cam_cx, cam_cy):
    """像素 + 深度 → 光学坐标系 3D 点（Z-forward, X-right, Y-down）"""
    x = (cx_px - cam_cx) / fx * depth_m
    y = (cy_px - cam_cy) / fy * depth_m
    return np.array([x, y, depth_m], dtype=np.float64)


def _optical_to_body(p):
    """光学系(Z-forward,X-right,Y-down) → body系(X-forward,Y-left,Z-up)"""
    x, y, z = p
    return np.array([z, -x, -y], dtype=np.float64)


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ZED X + YOLO TRT 球体检测（胸部相机）")
    parser.add_argument("--ball",  default="soccer-m", choices=list(BALL_CONFIGS),
                        help="球类 tennis|soccer|soccer-m")
    parser.add_argument("--model", default=None,
                        help="覆盖模型路径（默认由 --ball 决定）")
    parser.add_argument("--imgsz", type=int, default=320,
                        help="YOLO 输入尺寸（默认 320）")
    parser.add_argument("--show",  action="store_true",
                        help="启用 MJPEG HTTP 流（端口 8080）")
    args = parser.parse_args()

    ball_cfg   = BALL_CONFIGS[args.ball]
    class_id   = ball_cfg["class_id"]
    ball_r     = ball_cfg["radius"]
    model_path = args.model or ball_cfg["model"]
    print(f"[BALL] 类型={args.ball}  class_id={class_id}  半径={ball_r}m  模型={model_path}")

    # ── ZED ──────────────────────────────────────────────────────────────
    zed, cam_info = _open_zed()
    left_cam      = cam_info.camera_configuration.calibration_parameters.left_cam
    fx, fy        = left_cam.fx, left_cam.fy
    cam_cx, cam_cy = left_cam.cx, left_cam.cy
    print(f"[ZED] 内参: fx={fx:.1f}  fy={fy:.1f}  cx={cam_cx:.1f}  cy={cam_cy:.1f}")

    zed_image = sl.Mat()
    zed_depth = sl.Mat()
    runtime   = sl.RuntimeParameters()
    runtime.confidence_threshold = 50

    # ── YOLO ─────────────────────────────────────────────────────────────
    model, is_trt = _load_yolo(model_path)
    infer_kw      = _warmup(model, args.imgsz, is_trt)

    # ── 共享缓冲（主线程写，YOLO 线程 / encode 线程读）────────────────────
    buf_lock    = threading.Lock()
    buf_small   = [None]     # imgsz×imgsz BGR，YOLO 线程读
    buf_color   = [None]     # 原始 BGR，encode 线程读
    buf_depth   = [None]     # H×W float32
    buf_scale   = [1.0, 1.0] # [sx, sy] 原图→imgsz 缩放比
    buf_updated = threading.Event()
    stop_flag   = threading.Event()

    vis_lock  = threading.Lock()
    vis_state = {"bbox": None, "conf": 0.0, "valid": False,
                 "pos": (0.0, 0.0, 0.0), "coast": 0, "fps": 0.0}

    # ── YOLO 推理线程 ─────────────────────────────────────────────────────
    def yolo_worker():
        center_ema = None
        last_bbox  = None
        miss_count = 0
        yolo_fps   = _FPS()

        while not stop_flag.is_set():
            if not buf_updated.wait(timeout=1.0):
                continue
            buf_updated.clear()

            with buf_lock:
                small  = buf_small[0]
                depth  = buf_depth[0]
                sx, sy = buf_scale
            if small is None:
                continue

            results = model(small, conf=ball_cfg["conf"], verbose=False, **infer_kw)

            best_box  = None
            best_conf = 0.0
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == class_id:
                        c = float(box.conf[0])
                        if c > best_conf:
                            best_conf, best_box = c, box

            if best_box is not None:
                miss_count = 0
                x1s, y1s, x2s, y2s = best_box.xyxy[0]
                last_bbox = (int(x1s * sx), int(y1s * sy),
                             int(x2s * sx), int(y2s * sy))
            else:
                miss_count += 1

            published_valid = False
            x_b = y_b = z_b = 0.0
            if last_bbox is not None and miss_count <= COAST_FRAMES:
                x1, y1, x2, y2 = last_bbox
                cx_px = (x1 + x2) // 2
                cy_px = (y1 + y2) // 2

                depth_surface = _sample_depth(depth, cx_px, cy_px)
                if depth_surface > 0:
                    depth_m = depth_surface + ball_r
                    p_opt   = _deproject(cx_px, cy_px, depth_m, fx, fy, cam_cx, cam_cy)
                    p_body  = _optical_to_body(p_opt)

                    # TODO (P1): 调用 zed_to_base.transform_point_zed_to_base(p_body, q_wy, q_wr, q_wp)
                    #            将 body 系坐标变换到 pelvis 系

                    if center_ema is None:
                        center_ema = p_body.copy()
                    else:
                        gate_dist = np.linalg.norm(p_body - center_ema)
                        if gate_dist < EMA_GATE:
                            center_ema = EMA_ALPHA * p_body + (1 - EMA_ALPHA) * center_ema
                        else:
                            center_ema = p_body.copy()

                    x_b, y_b, z_b = center_ema
                    published_valid = True

            if not published_valid:
                center_ema = None

            with vis_lock:
                vis_state["bbox"]  = last_bbox if miss_count <= COAST_FRAMES else None
                vis_state["conf"]  = best_conf
                vis_state["valid"] = published_valid
                vis_state["pos"]   = (x_b, y_b, z_b)
                vis_state["coast"] = miss_count
                vis_state["fps"]   = yolo_fps.fps

            status = "BALL " if best_box is not None else ("COAST" if published_valid else "     ")
            print(
                f"\r[{status}] body=({x_b:+.3f},{y_b:+.3f},{z_b:+.3f})m  "
                f"conf={best_conf:.2f}  YOLO={yolo_fps.fps:.1f}fps",
                end="", flush=True,
            )
            yolo_fps.tick()

    yolo_thread = threading.Thread(target=yolo_worker, daemon=True, name="yolo")
    yolo_thread.start()

    # ── MJPEG HTTP 服务 ───────────────────────────────────────────────────
    _httpd = None
    if args.show:
        import http.server, socketserver

        _mjpeg_frame = [None]
        _mjpeg_lock  = threading.Lock()

        class _MJPEGHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_GET(self):
                if self.path not in ('/', '/stream'):
                    self.send_error(404); return
                self.send_response(200)
                self.send_header('Content-Type',
                                 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                try:
                    last = None
                    while True:
                        with _mjpeg_lock:
                            jpg = _mjpeg_frame[0]
                        if jpg is None or jpg is last:
                            time.sleep(0.005); continue
                        last = jpg
                        self.wfile.write(
                            b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                            + jpg + b'\r\n')
                except Exception:
                    pass

        socketserver.ThreadingTCPServer.allow_reuse_address = True
        _httpd = socketserver.ThreadingTCPServer(('0.0.0.0', 8080), _MJPEGHandler)
        _httpd.daemon_threads = True
        threading.Thread(target=_httpd.serve_forever, daemon=True, name="mjpeg-http").start()

        # ── encode 线程：@ 60fps 读最新帧 + 叠加检测框 → JPEG ──────────
        def _encode_worker():
            while not stop_flag.is_set():
                time.sleep(1 / 60)
                with buf_lock:
                    frame = buf_color[0]
                if frame is None:
                    continue
                with vis_lock:
                    bbox  = vis_state["bbox"]
                    conf  = vis_state["conf"]
                    valid = vis_state["valid"]
                    x_b, y_b, z_b = vis_state["pos"]
                    coast = vis_state["coast"]
                    yfps  = vis_state["fps"]

                # 缩到 640×400 再编码，降低 imencode GIL 开销
                vis = cv2.resize(frame, (640, 400))
                H, W = vis.shape[:2]
                sx_d = W / frame.shape[1]
                sy_d = H / frame.shape[0]

                if bbox is not None:
                    x1d = int(bbox[0]*sx_d); y1d = int(bbox[1]*sy_d)
                    x2d = int(bbox[2]*sx_d); y2d = int(bbox[3]*sy_d)
                    clr = (0, 255, 0) if coast == 0 else (0, 165, 255)
                    cv2.rectangle(vis, (x1d, y1d), (x2d, y2d), clr, 2)
                    cv2.circle(vis, ((x1d+x2d)//2, (y1d+y2d)//2), 5, clr, -1)
                    label = f"ball {conf:.2f}" if coast == 0 else f"coast {coast}"
                    cv2.putText(vis, label, (x1d, max(y1d-8, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 2)
                    if valid:
                        cv2.putText(vis,
                                    f"body ({x_b:+.2f},{y_b:+.2f},{z_b:+.2f})m",
                                    (10, H - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                cv2.putText(vis, f"YOLO {yfps:.0f}fps",
                            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                ok, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with _mjpeg_lock:
                        _mjpeg_frame[0] = buf.tobytes()

        threading.Thread(target=_encode_worker, daemon=True, name="encode").start()

        # 获取本机 IP 用于打印访问地址
        import socket
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            ip = "localhost"
        print(f"[MJPEG] 流媒体已启动 → 浏览器打开 http://{ip}:8080/stream")

    # ── 主线程：ZED 采集 ──────────────────────────────────────────────────
    print("[INFO] 开始采集，按 Ctrl+C 停止 ...")
    cam_fps = _FPS()
    try:
        while not stop_flag.is_set():
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(zed_image,   sl.VIEW.LEFT)
            zed.retrieve_measure(zed_depth, sl.MEASURE.DEPTH)

            color_bgra = zed_image.get_data()
            depth_raw  = zed_depth.get_data()

            color_bgr = cv2.cvtColor(color_bgra, cv2.COLOR_BGRA2BGR)
            H, W      = color_bgr.shape[:2]
            small     = cv2.resize(color_bgr, (args.imgsz, args.imgsz))

            with buf_lock:
                buf_color[0] = color_bgr
                buf_small[0] = small
                buf_depth[0] = depth_raw.copy()
                buf_scale[:] = [W / args.imgsz, H / args.imgsz]
            buf_updated.set()
            cam_fps.tick()

    except KeyboardInterrupt:
        print("\n[INFO] 收到 Ctrl+C，正在退出...")
    finally:
        stop_flag.set()
        yolo_thread.join(timeout=2.0)
        zed.close()
        print("[INFO] ZED 已关闭。")
        if _httpd:
            _httpd.shutdown()
        print(f"[INFO] 采集 FPS: {cam_fps.fps:.1f}")


if __name__ == "__main__":
    main()
