#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-robomimic}"
REPO_DIR="/home/unitree/yixuan/yichao-deploy/RoboMimic_Deploy"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not in PATH" >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  tmux attach-session -t "${SESSION_NAME}"
  exit 0
fi

tmux new-session -d -s "${SESSION_NAME}" -n main -c "${REPO_DIR}"

# Layout (2 columns × 3 rows):
#   left-top    | right-top     bridge/cpp  | apriltag target detection (camera only)
#   left-mid    | right-mid     deploy_policy | lidar ball detection → rt/ball_state
#   left-bottom | right-bottom  sensor dashboard (port 8091) | (free shell)
#
# Data flow:
#   right-top  -> rt/target_state   (AprilTag)
#   right-mid  -> rt/ball_state     (lidar直接输出，无 fuser 中间层)
#   left-bot   subscribes rt/target_state + rt/ball_state -> browser http://<robot>:8091/
LEFT_TOP="$(tmux display-message -p -t "${SESSION_NAME}:0.0" '#{pane_id}')"
RIGHT_TOP="$(tmux split-window -h -P -F '#{pane_id}' -t "${LEFT_TOP}" -c "${REPO_DIR}")"
LEFT_MID="$(tmux split-window -v -P -F '#{pane_id}' -t "${LEFT_TOP}" -c "${REPO_DIR}")"
RIGHT_MID="$(tmux split-window -v -P -F '#{pane_id}' -t "${RIGHT_TOP}" -c "${REPO_DIR}")"
LEFT_BOTTOM="$(tmux split-window -v -P -F '#{pane_id}' -t "${LEFT_MID}" -c "${REPO_DIR}")"
RIGHT_BOTTOM="$(tmux split-window -v -P -F '#{pane_id}' -t "${RIGHT_MID}" -c "${REPO_DIR}")"

# Pre-fill commands — press Enter in each pane to start
tmux send-keys -t "${RIGHT_TOP}"    -l "cd ${REPO_DIR} && ./onboard/perception/camera/run_apriltag_target.sh --show"
tmux send-keys -t "${RIGHT_MID}"    -l "cd ${REPO_DIR} && ./onboard/perception/lidar/run.sh --show --base-y-bias 0.00 --dds-topic rt/ball_state"
tmux send-keys -t "${LEFT_TOP}"     -l "cd ${REPO_DIR} && cmake -S bridge -B bridge/build && cmake --build bridge/build -j2 && BRIDGE_NETWORK_INTERFACE=eth0 bridge/build/cpp_bridge_main"
tmux send-keys -t "${LEFT_MID}"     -l "cd ${REPO_DIR} && python bridge/python/deploy_policy.py"
tmux send-keys -t "${LEFT_BOTTOM}"  -l "cd ${REPO_DIR} && bash onboard/perception/run_sensor_dashboard.sh"
# RIGHT_BOTTOM is left as a free shell for ad-hoc commands

tmux select-pane -t "${LEFT_TOP}"
tmux attach-session -t "${SESSION_NAME}"
