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

# Build a 2x2 layout:
#   left-top      right-top
#   left-bottom   right-bottom
LEFT_TOP="$(tmux display-message -p -t "${SESSION_NAME}:0.0" '#{pane_id}')"
RIGHT_TOP="$(tmux split-window -h -P -F '#{pane_id}' -t "${LEFT_TOP}" -c "${REPO_DIR}")"
LEFT_BOTTOM="$(tmux split-window -v -P -F '#{pane_id}' -t "${LEFT_TOP}" -c "${REPO_DIR}")"
RIGHT_BOTTOM="$(tmux split-window -v -P -F '#{pane_id}' -t "${RIGHT_TOP}" -c "${REPO_DIR}")"

tmux send-keys -t "${RIGHT_TOP}" -l "cd ${REPO_DIR} && ./onboard/perception/camera/run_apriltag_target.sh --tag-id 0 --tag-size 0.15 --show"
tmux send-keys -t "${RIGHT_BOTTOM}" -l "cd ${REPO_DIR} && ./onboard/perception/lidar/run.sh --show"
tmux send-keys -t "${LEFT_TOP}" -l "cd ${REPO_DIR} && cmake -S bridge -B bridge/build && cmake --build bridge/build -j2 && BRIDGE_NETWORK_INTERFACE=eth0 bridge/build/cpp_bridge_main"
tmux send-keys -t "${LEFT_BOTTOM}" -l "cd ${REPO_DIR} && python bridge/python/deploy_policy.py"

tmux select-pane -t "${LEFT_TOP}"
tmux attach-session -t "${SESSION_NAME}"
