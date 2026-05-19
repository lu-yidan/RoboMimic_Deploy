#!/usr/bin/env bash
# ============================================================
# 启动灰度/IR 相机 AprilTag + bright-ball 检测
#
# 适用场景：
#   - librealsense RGB 是黑屏
#   - V4L2 IR/灰度画面正常
#   - 需要 AprilTag 目标 + 白球/反光球检测
#
# 注意：
#   默认由单个进程独占 V4L2 GREY 节点，并在内部用 latest-frame
#   worker 发布 rt/cam_ball_state，避免多个进程抢同一相机。
#   这里不启用 RealSense depth-only。实测它会和 V4L2 UVC
#   抢同一台 D435I，导致 /dev/video* 重新枚举甚至灰度画面消失。
# ============================================================
set -e
cd "$(dirname "$0")/../../.."

exec bash onboard/perception/camera/run_apriltag_target.sh \
    --camera-profile gray-ir \
    --color-backend v4l2 \
    --v4l2-device "${V4L2_DEVICE:-/dev/video3}" \
    --v4l2-fourcc "${V4L2_FOURCC:-GREY}" \
    --v4l2-fps "${V4L2_FPS:-30}" \
    --ball-bright \
    --ball-bright-max-hz "${BALL_BRIGHT_MAX_HZ:-0}" \
    "$@"
