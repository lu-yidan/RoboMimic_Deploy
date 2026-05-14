#!/usr/bin/env bash
# ============================================================
# 启动彩色 V4L2 相机 AprilTag + bright-ball 检测
#
# 适用场景：
#   - 需要正常彩色视频预览
#   - 当前 D435I 彩色 UVC 节点通常是 /dev/video4
#   - 仍使用 bright-ball 的形状/几何过滤发布 rt/cam_ball_state
#
# 与 run_apriltag_gray_ball.sh 的区别：
#   - 本脚本固定走 YUYV 彩色流
#   - 灰度/IR 脚本固定走 GREY 流
# ============================================================
set -e
cd "$(dirname "$0")/../../.."

exec bash onboard/perception/camera/run_apriltag_target.sh \
    --camera-profile color-v4l2 \
    --color-backend v4l2 \
    --v4l2-device "${V4L2_DEVICE:-/dev/video4}" \
    --v4l2-fourcc "${V4L2_FOURCC:-YUYV}" \
    --v4l2-fps "${V4L2_FPS:-30}" \
    --ball-bright \
    "$@"
