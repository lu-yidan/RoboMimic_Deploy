# Onboard Install Guide for Agents

This guide is for installing the onboard perception/runtime stack on another
Unitree G1 Jetson robot as quickly as possible. It records the working path from
the JetPack 6 / Orin install in this repo.

## Target Machine

First identify the robot. Do not blindly follow older JetPack 5 instructions.

```bash
cat /etc/nv_tegra_release
cat /etc/os-release
nvcc --version
```

Known-good setup from this install:

- Ubuntu 22.04 / JetPack 6, L4T R36.4.3
- CUDA 12.6
- TensorRT 10.3
- Python 3.10 conda environment named `robomimic`
- RealSense D435I on USB 3.2

If the robot is JetPack 5.1.2 / CUDA 11.4, use the older notes in
`onboard/perception/camera/TROUBLESHOOTING.md` instead of the PyTorch commands
below.

## Fast Install

Use `uv` for pip packages. Keep the env name `robomimic` because repo scripts
call `conda run -n robomimic`.

```bash
source /home/unitree/miniconda3/etc/profile.d/conda.sh
conda create -n robomimic -c conda-forge python=3.10 pip -y

conda install -n robomimic -c conda-forge \
  numpy scipy pyyaml hydra-core omegaconf pygame pillow opencv pyrealsense2 -y

uv pip install --python /home/unitree/miniconda3/envs/robomimic/bin/python \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
  "torch==2.8.0" "torchvision==0.23.0"

uv pip install --python /home/unitree/miniconda3/envs/robomimic/bin/python \
  "numpy<2" ultralytics onnx onnxruntime mujoco pupil-apriltags
```

Do not use the default PyPI `torch` on Jetson. It will either be CPU-only or
pull a wheel that expects libraries missing from the robot.

## TensorRT Python Binding

JetPack installs TensorRT bindings for system Python 3.10 at
`/usr/lib/python3.10/dist-packages`. Link only TensorRT into conda; do not prepend
the whole system dist-packages directory to `PYTHONPATH`, because that can make
conda load system `cv2`, `protobuf`, or `sympy`.

```bash
CONDA_SITE=$(/home/unitree/miniconda3/envs/robomimic/bin/python -c 'import site; print(site.getsitepackages()[0])')
for name in \
  tensorrt tensorrt_lean tensorrt_dispatch \
  tensorrt-10.3.0.dist-info \
  tensorrt_lean-10.3.0.dist-info \
  tensorrt_dispatch-10.3.0.dist-info
do
  if [ -e "/usr/lib/python3.10/dist-packages/$name" ] && [ ! -e "$CONDA_SITE/$name" ]; then
    ln -s "/usr/lib/python3.10/dist-packages/$name" "$CONDA_SITE/$name"
  fi
done
```

## CycloneDDS Python Binding

Ubuntu Jammy's `cyclonedds-dev` is too old for `cyclonedds==0.10.5`. Build
CycloneDDS 0.10.5 into a user prefix, then compile the Python binding against it.
Disable shared memory support to avoid missing `iceoryx` headers.

```bash
mkdir -p /home/unitree/share/src /home/unitree/share/opt
git clone --branch 0.10.5 --depth 1 \
  https://github.com/eclipse-cyclonedds/cyclonedds.git \
  /home/unitree/share/src/cyclonedds-0.10.5

cmake -S /home/unitree/share/src/cyclonedds-0.10.5 \
  -B /home/unitree/share/src/cyclonedds-0.10.5/build \
  -DCMAKE_INSTALL_PREFIX=/home/unitree/share/opt/cyclonedds-0.10.5 \
  -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DENABLE_SHM=OFF \
  -DCMAKE_BUILD_TYPE=Release

LD_LIBRARY_PATH=/home/unitree/share/src/cyclonedds-0.10.5/build/lib \
  cmake --build /home/unitree/share/src/cyclonedds-0.10.5/build -j"$(nproc)"

LD_LIBRARY_PATH=/home/unitree/share/src/cyclonedds-0.10.5/build/lib \
  cmake --install /home/unitree/share/src/cyclonedds-0.10.5/build

CYCLONEDDS_HOME=/home/unitree/share/opt/cyclonedds-0.10.5 \
LD_LIBRARY_PATH=/home/unitree/share/opt/cyclonedds-0.10.5/lib:/home/unitree/miniconda3/envs/robomimic/lib \
  uv pip install --python /home/unitree/miniconda3/envs/robomimic/bin/python \
  --no-build-isolation --no-cache "cyclonedds==0.10.5"
```

Verify that Python links to the user-built library:

```bash
source onboard/perception/setup_runtime_env.sh
conda run -n robomimic --no-capture-output python -c \
  "from cyclonedds.domain import DomainParticipant; DomainParticipant(0); print('cyclonedds ok')"

ldd $(conda run -n robomimic python -c \
  "import sysconfig, glob; print(glob.glob(sysconfig.get_path('platlib') + '/cyclonedds/_clayer*.so')[0])") | grep ddsc
```

Expected `ldd` path:

```text
/home/unitree/share/opt/cyclonedds-0.10.5/lib/libddsc.so.0
```

## Runtime Path Setup

Use the shared script added in this repo:

```bash
source onboard/perception/setup_runtime_env.sh
```

It keeps conda's `libstdc++` before Ubuntu's copy, adds CycloneDDS 0.10.5, and
adds Jetson CUDA/TensorRT runtime library paths. This avoids common errors like:

- `CXXABI_1.3.15 not found` from OpenCV
- `undefined symbol: DDS_XTypes_TypeObject_desc` from CycloneDDS
- accidental import of system `protobuf` or `cv2`

## RealSense Setup

Install udev rules if device access fails:

```bash
curl -L -o /tmp/99-realsense-libusb.rules \
  https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
echo "123" | sudo -S cp /tmp/99-realsense-libusb.rules /etc/udev/rules.d/99-realsense-libusb.rules
echo "123" | sudo -S udevadm control --reload-rules
echo "123" | sudo -S udevadm trigger
```

Unplug/replug the camera, then verify:

```bash
source onboard/perception/setup_runtime_env.sh
conda run -n robomimic --no-capture-output python - <<'PY'
import pyrealsense2 as rs
ctx = rs.context()
print("devices", len(ctx.devices))
for d in ctx.devices:
    print("name", d.get_info(rs.camera_info.name))
    print("serial", d.get_info(rs.camera_info.serial_number))
    print("usb", d.get_info(rs.camera_info.usb_type_descriptor))
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipe.start(cfg)
frames = pipe.wait_for_frames(5000)
print("pipeline ok", bool(frames.get_color_frame()), bool(frames.get_depth_frame()))
pipe.stop()
PY
```

Expected: one D435/D435I/D455 device, USB `3.x`, and `pipeline ok True True`.

## ROS2 / LiDAR / Unitree Messages

The AprilTag target publisher, camera ball publisher, and LiDAR detector import:

- `rclpy`
- `unitree_hg.msg`
- `sensor_msgs`
- `livox_ros_driver2`

If `/opt/ros` and `ws_livox` are missing, those scripts will not fully start.
Install or restore the robot's ROS2 + Unitree message workspace before running:

```bash
bash onboard/perception/camera/run_apriltag_target.sh --show
bash onboard/perception/lidar/run.sh
bash onboard/perception/run_sensor_dashboard.sh
```

After ROS2 is present, keep:

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="eth0" priority="default" multicast="default" /></Interfaces></General></Domain></CycloneDDS>'
```

This avoids FastDDS shared-memory OOM behavior on 16 GB Jetson systems.

## Final Checklist

```bash
source onboard/perception/setup_runtime_env.sh

conda run -n robomimic --no-capture-output python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

conda run -n robomimic --no-capture-output python -c \
  "import cv2, pyrealsense2; print('vision stack ok')"

conda run -n robomimic --no-capture-output python -c \
  "from cyclonedds.domain import DomainParticipant; DomainParticipant(0); print('dds ok')"

conda run -n robomimic --no-capture-output python -c \
  "from common.ball_state_dds import BallStatePublisher; from common.target_state_dds import TargetStatePublisher; print('repo dds ok')"
```

If all four pass, the conda/runtime side is installed. Then validate hardware
with the RealSense smoke test and, if ROS2 is available, the onboard perception
scripts.
