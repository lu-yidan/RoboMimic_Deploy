# Camera Ball Detector — 排障与调优记录

> 适用硬件：**Unitree G1** 机载电脑（NVIDIA Jetson Orin NX 16 GB，JetPack 5.1.2）  
> 相机：**Intel RealSense D435**  
> 模型：**YOLOv8n**（ultralytics）

---

## 一、环境配置

### 1.1 硬件规格

| 项目 | 规格 |
|------|------|
| SoC | NVIDIA Jetson Orin NX 16 GB |
| CPU | 8× ARM Cortex-A78AE（1.5 GHz） |
| GPU | 1024-core Ampere（最高 918 MHz） |
| 内存 | 16 GB LPDDR5，CPU/GPU 统一内存 |
| JetPack | 5.1.2（CUDA 11.4，cuDNN 8.6） |
| Python | 3.8（conda 环境 `robomimic`） |
| ROS2 | Foxy + CycloneDDS |

### 1.2 conda 环境依赖

```bash
conda activate robomimic

# RealSense（必须用 conda-forge，pip 版不含 ARM so 文件）
conda install -c conda-forge pyrealsense2 -y

# ultralytics（YOLOv8）
pip install ultralytics

# PyTorch（见 1.3，不能直接 pip install）
# torchvision（见 1.4，必须源码编译）
```

### 1.3 为 Jetson 安装正确的 PyTorch

**问题：** `pip install torch` 默认从 PyPI 拉取 x86_64 wheel，没有 aarch64+CUDA 版本；
即使安装成功，`torch.cuda.is_available()` 返回 `False`，YOLO 只能跑 CPU（~670 ms/帧）。

**解法：** 从 NVIDIA 开发者网站下载 Jetson 专属 wheel。

```bash
# JetPack 5.1.2 对应 PyTorch 2.1.0
wget https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/\
torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

pip install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

# 验证
python -c "import torch; print(torch.cuda.is_available())"
# 应输出 True
```

> NVIDIA Jetson PyTorch 下载页：  
> https://developer.nvidia.com/embedded/downloads#?search=pytorch

### 1.4 编译 torchvision（必须对齐 PyTorch 版本）

`pip install torchvision` 从 PyPI 拉取的版本（如 0.19.1）与 `torch 2.1.0` **不兼容**，
会报 `RuntimeError: Couldn't load custom C++ ops`。
必须从源码编译 `v0.16.0`。

```bash
# 指定与 JetPack 一致的 CUDA 版本
export CUDA_HOME=/usr/local/cuda-11.4
export PATH=$CUDA_HOME/bin:$PATH

# 卸载可能存在的不兼容版本
pip uninstall torchvision -y

# 克隆并编译（约需 20-30 分钟，全程在机器人上完成）
git clone --branch v0.16.0 --depth 1 https://github.com/pytorch/vision
cd vision
python setup.py install
cd ..
rm -rf vision

# 验证
python -c "import torchvision; print(torchvision.__version__)"
# 应输出 0.16.0
```

> **为什么不能直接 pip install？**  
> PyPI 上只有 x86 预编译 wheel；aarch64+JetPack CUDA 组合无对应官方包，
> 必须本地编译链接到机器上的 CUDA 11.4。

---

## 二、遇到的问题与解决方案

### 问题 1：EMA 公式写反，球心位置疯狂跳变

**现象**：`check_ball_state.py` 观察到 x/y/z 在大范围内快速跳变，噪声极大。

**根因**：`lidar/ball_detector.py` 中 EMA 权重写反：
```python
# 错误：alpha=0.9 让新测量值占 90%，完全没有平滑效果
self.center_ema = alpha * center_lidar + (1 - alpha) * self.center_ema
```

**修复**：
```python
# 正确：alpha 对应"历史权重"，值越大跟踪越平滑
self.center_ema = alpha * self.center_ema + (1 - alpha) * center_lidar
```

同时将 `alpha` 从 0.9 调整到 0.5（更快响应新位置）。

---

### 问题 2：本地 check_ball_state.py 显示全零 / 数据过期

**现象**：机器人上 `ball_detector.py` 显示正常坐标，本地 `check_ball_state.py` 显示
`x=0 y=0 z=0 age=11000ms`（过期）。

**根因**：`BallStateSubscriber.latest()` 用 **机器人时钟戳** 与 **本地时钟** 做差判断新鲜度，
两台机器未做时钟同步，差值常达数秒，导致所有消息都被判为"过期"。

**修复**（`common/ball_state_dds.py`）：
```python
# 收到消息时记录本地接收时刻
self._received_at = int(time.time() * 1e6)

# 用本地接收时刻判断新鲜度，不依赖机器人时钟
age_us = now_us - self._received_at   # 而非 now_us - s.timestamp_us
```

---

### 问题 3：YOLO 推理极慢（~670 ms/帧），torch.cuda.is_available() = False

**现象**：`[INFO] YOLO inference: 671.4 ms/frame (~ 1 FPS upper bound)`

**根因**：安装的是 PyPI 通用 PyTorch，不含 ARM+CUDA 支持，YOLO 跑在 CPU 上。

**修复**：见 §1.3，安装 NVIDIA Jetson 专属 PyTorch wheel。

修复后 warmup 输出：
```
warmup[0]: 3845.8ms   ← 首次触发 CUDA JIT 编译，正常
warmup[1]:   26.9ms
warmup[2]:   24.9ms
[INFO] YOLO inference: 23.4 ms/frame (≈ 43 FPS upper bound)  ✅
```

---

### 问题 4：GPU 频率被限速（115 MHz → 918 MHz）

**现象**：安装 GPU PyTorch 后，YOLO 仍显示 ~260 ms/帧。

**诊断**：
```bash
cat /sys/class/devfreq/*/cur_freq
# 输出 115200000（115 MHz，超低功耗模式）
```

**修复**：
```bash
# 切换到最大性能模式（需 sudo，密码 123）
echo "123" | sudo -S nvpmodel -m 0
echo "123" | sudo -S jetson_clocks

# 验证
cat /sys/class/devfreq/*/cur_freq
# 应输出 918400000（918 MHz）
```

---

### 问题 5：ROS2 /lowstate 500 Hz 订阅饱和 GIL，YOLO 降速 10×

**现象**：`nvpmodel` + `jetson_clocks` 后独立测试 YOLO 为 23 ms，
但在 `ball_detector.py` 完整流程中仍显示 ~260 ms。

**根因**：`/lowstate` 以 500 Hz 高频发布，`rclpy.spin(joint)` 在 ROS2 线程
持续反序列化，几乎全时占用 Python GIL，导致 YOLO CUDA 和 Python 切换之间
的 GIL 等待时间极长。

**修复**：将 `rclpy.spin()` 替换为受控 50 Hz 轮询：
```python
def _spin_loop():
    while True:
        rclpy.spin_once(joint, timeout_sec=0.0)
        time.sleep(0.02)   # 50 Hz，让出 GIL 给 YOLO

threading.Thread(target=_spin_loop, daemon=True).start()
```

修复后 YOLO 恢复 23 ms/帧，完整流程 FPS 从 ~4 提升到预期水平。

---

### 问题 6：整体 FPS 仍为 0.1（YOLO 线程实际卡死 34 秒/帧）

**现象**：YOLO warmup 显示 23 ms，但运行中 `YOLO= 0.1fps`，
TIMING 日志显示 `track=33979ms`（34 秒！）。

**诊断过程**：

1. 加入每帧 TIMING 打印，发现 TIMING 长期不出现 → YOLO 线程根本没完成一帧
2. 加入 `[LOOP]` 调试打印：
   ```
   [LOOP] iter=0 waiting...
   [LOOP] iter=0 got=True wait=0ms      ← 事件拿到了
   [LOOP] acquiring buf_lock...         ← 等 lock
   [LOOP] copy done 102.4ms, resizing...← copy 花了 102 ms！
   [LOOP] resize done 97.3ms, tracking... ← resize 花了 97 ms！
   （之后无输出 → model.track() 卡死）
   ```
3. 发现两个叠加问题（见下）

#### 子问题 6a：depth 帧直接引用 DMA 缓冲区，ARM 读速只有 8 MB/s

**根因**：相机线程：
```python
buf_depth = np.asanyarray(df.get_data())   # ← 无 .copy()，是 DMA 内存的视图！
```
YOLO 线程从该 DMA 地址 `copy()` 时，ARM 访问未缓存 DMA 区速度极慢（约 8 MB/s），
0.8 MB 深度图需要 ~100 ms。

**临时修复**：
```python
buf_depth = np.asanyarray(df.get_data()).copy()  # 在相机线程立即 copy 到 Python 堆
```

#### 子问题 6b：`align.process()` 持有 GIL 126 ms，使 model.track() 中每次 Python/CUDA 切换等待 126 ms

**诊断**：
```python
t0 = time.perf_counter()
aligned = align.process(frames)
print(f"align.process: {(time.perf_counter()-t0)*1e3:.1f}ms")
# 输出：align.process: 126.1ms
```

`align.process()` 是 Intel RealSense SDK 的 C 扩展，在 ARM 上做深度-颜色投影对齐，
CPU 计算量大，且**不释放 GIL**（Intel 未实现 GIL 友好的接口）。

原来的架构：
```
主线程：pipeline.wait_for_frames()（释放 GIL）
      → align.process()（持有 GIL 126 ms）  ← 每 126 ms 阻塞 YOLO 线程一次
      → buf_lock + copy

YOLO 线程：model.track()
  内部每次 Python/CUDA 切换需要 GIL
  → 每次等待 126 ms
  → 30 ms 实际计算被切割成数百个 GIL 等待片段
  → 总耗时 34 秒
```

**根本修复**：将 `align.process()` 从相机主线程移到 YOLO 线程，顺序执行，彻底消除 GIL 争抢：

```python
# ── 相机主线程（修复后）────────────────────────────────────
while True:
    frames = pipeline.wait_for_frames()   # C扩展，自动释放 GIL
    with buf_lock:
        buf_frames = frames               # 只交换引用，不做任何计算
    buf_updated.set()

# ── YOLO 线程（修复后）─────────────────────────────────────
while not stop_flag.is_set():
    buf_updated.wait(timeout=1.0)
    buf_updated.clear()
    with buf_lock:
        frames = buf_frames

    # align 和 copy 在 YOLO 线程顺序执行，无 GIL 竞争
    aligned = align.process(frames)       # 126 ms，但不再和 model.track() 并发
    cf = aligned.get_color_frame()
    df = aligned.get_depth_frame()
    color = np.asanyarray(cf.get_data()).copy()
    depth = np.asanyarray(df.get_data()).copy()

    # YOLO 推理，此时相机主线程只在 pipeline.wait_for_frames() 阻塞，不持有 GIL
    results = model.track(color_small, ...)
```

**修复前后对比**：

| 步骤 | 修复前 | 修复后 |
|------|--------|--------|
| copy（含 DMA 读取）| 102 ms | ~100 ms（align 本身耗时，不可避免） |
| resize | 97 ms | 1 ms |
| model.track() | **34,000 ms** | **44 ms** |
| 单帧总计 | ∞（卡死） | **~150 ms** |
| **实际 FPS** | **0.1 FPS** | **~7 FPS** |

---

### 问题 7：MJPEG 浏览器流卡顿（端口冲突 / busy-loop）

**现象**：`--show` 模式启动时报 `OSError: [Errno 98] Address already in use`；
浏览器刷新图片极慢。

**修复**：
```python
# 允许端口快速复用（上次进程崩溃留下 TIME_WAIT 状态）
socketserver.ThreadingTCPServer.allow_reuse_address = True

# MJPEG 推送线程避免 busy-loop，仅发送新帧
last_sent = None
while True:
    with _mjpeg_lock:
        jpg = _mjpeg_frame[0]
    if jpg is None or jpg is last_sent:
        time.sleep(0.02)   # 50 Hz 轮询
        continue
    last_sent = jpg
    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
```

---

## 三、性能调优汇总

### 最终运行参数

```bash
# 启动前解锁 GPU/CPU 频率
echo "123" | sudo -S nvpmodel -m 0
echo "123" | sudo -S jetson_clocks

# 启动球检测（推荐参数）
python onboard/perception/camera/ball_detector.py --imgsz 320

# 带 MJPEG 视频流（浏览器访问 http://192.168.123.164:8080）
python onboard/perception/camera/ball_detector.py --imgsz 320 --show
```

### 关键参数影响

| 参数 | 值 | FPS 约估 | 备注 |
|------|----|----------|------|
| `--imgsz` | 640 | ~4 FPS | 精度高，球在远处时更准 |
| `--imgsz` | 320 | ~7 FPS | 推荐，速度与精度平衡 |
| `--model` | yolov8n.pt | ~7 FPS | 默认，最快 |
| `--model` | yolov8s.pt | ~5 FPS | 稍慢，精度略高 |

### 瓶颈分布（`--imgsz 320`，稳定运行时）

```
[TIMING] copy=100ms  resize=1ms  track=44ms  post=2ms  total=148ms  fps=7
```

- **`copy=100ms`**：`align.process()` 的固有 CPU 耗时（Intel RealSense ARM 对齐）
- **`track=44ms`**：YOLO 实际推理（GPU 23ms + ByteTracker + 开销）
- 如需更高帧率，可将相机分辨率从 848×480 降至 640×480

---

## 四、已知限制

| 限制 | 说明 |
|------|------|
| `align.process()` 100 ms | Intel RealSense ARM CPU 对齐，不用 GPU，是当前主瓶颈 |
| 首次 CUDA JIT ~4 s | `warmup[0]` 触发 PyTorch CUDA 内核编译，属正常现象 |
| Segmentation fault on exit | `rclpy.shutdown()` 与 pyrealsense2 同时析构时偶发，不影响运行时 |
| `lap` 依赖自动安装 | 首次运行 ultralytics 会联网下载 `lap`，需要网络 |

---

## 五、诊断命令速查

```bash
# 查看 GPU 当前频率（918400000 = 918 MHz = 最大）
cat /sys/class/devfreq/*/cur_freq

# 查看 GPU/CPU 实时功率和温度
sudo tegrastats

# 确认 PyTorch 使用 CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"

# 确认 YOLO 使用 GPU（看 device=cuda:0）
python -c "
from ultralytics import YOLO
import numpy as np, time
m = YOLO('yolov8n.pt')
dummy = np.zeros((320,320,3), dtype=np.uint8)
for _ in range(3): m(dummy, verbose=False, device='cuda:0', half=True)
t = time.perf_counter()
for _ in range(10): m(dummy, verbose=False, device='cuda:0', half=True)
print(f'avg: {(time.perf_counter()-t)/10*1000:.1f} ms')
"
# 应输出约 23 ms；若 >100 ms 则 GPU 未启用或频率被限速

# 查看 align.process() 耗时
python -c "
import pyrealsense2 as rs, time
p = rs.pipeline(); c = rs.config()
c.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
c.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
p.start(c); a = rs.align(rs.stream.color)
for i in range(5):
    f = p.wait_for_frames()
    t0 = time.perf_counter()
    a.process(f)
    print(f'align[{i}]: {(time.perf_counter()-t0)*1e3:.1f}ms')
p.stop()
"
# 预期约 120-130 ms（ARM CPU，正常）
```
