#!/usr/bin/env bash
# ============================================================
# 启动球位置网页可视化器（DDS rt/ball_state -> browser）
#
# 用法：
#   bash onboard/perception/run_ball_web_viewer.sh
#   bash onboard/perception/run_ball_web_viewer.sh --port 8090
#   bash onboard/perception/run_ball_web_viewer.sh --topic rt/ball_state
#
# 默认网页端口：8090
# 启动后访问终端打印出的 http://<robot-ip>:8090/
# ============================================================
set -e
cd "$(dirname "$0")/../.."

export PYTHONPATH=/usr/lib/python3.8/dist-packages:${PYTHONPATH:-}

conda run -n robomimic --no-capture-output \
    python -u onboard/perception/debug/ball_web_viewer.py "$@"
