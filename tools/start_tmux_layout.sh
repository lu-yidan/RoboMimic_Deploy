#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-robomimic}"
REPO_DIR="/home/unitree/yichao/RoboMimic_Deploy"
TMUX_SHELL="bash --noprofile --norc"
ROBOT_IFACE="${ROBOT_IFACE:-enP8p1s0}"
UNITREE_SDK2_DIR="${UNITREE_SDK2_DIR:-/home/unitree/unitree_sdk2-main}"
CYCLONEDDS_IDLC="${CYCLONEDDS_IDLC:-/opt/ros/humble/bin/idlc}"
SCORE_ONNX="${SCORE_ONNX:-${REPO_DIR}/policy/score/model/policy-obs-utf8-486.onnx}"
PYTHON_BIN="${PYTHON_BIN:-/home/unitree/miniconda3/envs/robomimic/bin/python}"
CONDA_ENV_LIB="${CONDA_ENV_LIB:-/home/unitree/miniconda3/envs/robomimic/lib}"
UNITREE_DDS_LIB="${UNITREE_SDK2_DIR}/thirdparty/lib/$(uname -m)"
PY_CYCLONEDDS_LIB="${PY_CYCLONEDDS_LIB:-/home/unitree/share/opt/cyclonedds-0.10.5/lib}"
BRIDGE_CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"${ROBOT_IFACE}\" priority=\"default\" multicast=\"default\" /></Interfaces></General><SharedMemory><Enable>false</Enable></SharedMemory></Domain></CycloneDDS>"
PY_BRIDGE_CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"${ROBOT_IFACE}\" priority=\"default\" multicast=\"default\" /></Interfaces></General></Domain></CycloneDDS>"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not in PATH" >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  tmux attach-session -t "${SESSION_NAME}"
  exit 0
fi

tmux new-session -d -s "${SESSION_NAME}" -n main -c "${REPO_DIR}" "${TMUX_SHELL}"

# Layout (2 columns × 3 rows):
#   left-top    | right-top     bridge/cpp  | gray camera target/ball + ball fuser
#   left-mid    | right-mid     deploy_policy | lidar ball detection → rt/lidar_ball_state
#   left-bottom | right-bottom  sensor dashboard (port 8091) | (free shell)
#
# Data flow:
#   right-top  -> rt/target_state + rt/cam_ball_state; fuser -> rt/ball_state
#   right-mid  -> rt/lidar_ball_state
#   left-bot   subscribes target/camera/lidar/fused topics -> browser http://<robot>:8091/
LEFT_TOP="$(tmux display-message -p -t "${SESSION_NAME}:0.0" '#{pane_id}')"
RIGHT_TOP="$(tmux split-window -h -P -F '#{pane_id}' -t "${LEFT_TOP}" -c "${REPO_DIR}" "${TMUX_SHELL}")"
LEFT_MID="$(tmux split-window -v -P -F '#{pane_id}' -t "${LEFT_TOP}" -c "${REPO_DIR}" "${TMUX_SHELL}")"
RIGHT_MID="$(tmux split-window -v -P -F '#{pane_id}' -t "${RIGHT_TOP}" -c "${REPO_DIR}" "${TMUX_SHELL}")"
LEFT_BOTTOM="$(tmux split-window -v -P -F '#{pane_id}' -t "${LEFT_MID}" -c "${REPO_DIR}" "${TMUX_SHELL}")"
RIGHT_BOTTOM="$(tmux split-window -v -P -F '#{pane_id}' -t "${RIGHT_MID}" -c "${REPO_DIR}" "${TMUX_SHELL}")"

# Pre-fill commands — press Enter in each pane to start
tmux send-keys -t "${RIGHT_TOP}"    -l "cd ${REPO_DIR} && ./onboard/perception/camera/run_gray_perception.sh --with-fuser --show"
tmux send-keys -t "${RIGHT_MID}"    -l "cd ${REPO_DIR} && ./onboard/perception/lidar/run.sh --show --base-y-bias 0.05 --dds-topic rt/lidar_ball_state"
tmux send-keys -t "${LEFT_TOP}"     -l "cd ${REPO_DIR} && rm -rf bridge/build && cmake -S bridge -B bridge/build -DUNITREE_SDK2_DIR=${UNITREE_SDK2_DIR} -DCYCLONEDDS_IDLC=${CYCLONEDDS_IDLC} && cmake --build bridge/build -j2 && LD_LIBRARY_PATH=${UNITREE_DDS_LIB}:\${LD_LIBRARY_PATH:-} CYCLONEDDS_URI='${BRIDGE_CYCLONEDDS_URI}' BRIDGE_NETWORK_INTERFACE=${ROBOT_IFACE} bridge/build/cpp_bridge_main"
tmux send-keys -t "${LEFT_MID}"     -l "cd ${REPO_DIR} && if [[ -f \"${SCORE_ONNX}\" ]]; then LD_LIBRARY_PATH=${PY_CYCLONEDDS_LIB}:${CONDA_ENV_LIB}:/opt/onnxruntime/lib:\${LD_LIBRARY_PATH:-} CYCLONEDDS_URI='${PY_BRIDGE_CYCLONEDDS_URI}' ROS_LOCALHOST_ONLY=0 \"${PYTHON_BIN}\" bridge/python/deploy_policy.py; else echo \"[deploy_policy] Missing ONNX model: ${SCORE_ONNX}\"; echo \"[deploy_policy] Copy the trained policy .onnx into policy/score/model/ or set SCORE_ONNX / policy/score/config/score.yaml.\"; fi"
tmux send-keys -t "${LEFT_BOTTOM}"  -l "cd ${REPO_DIR} && bash onboard/perception/run_sensor_dashboard.sh"
# RIGHT_BOTTOM is left as a free shell for ad-hoc commands

tmux select-pane -t "${LEFT_TOP}"
tmux attach-session -t "${SESSION_NAME}"
