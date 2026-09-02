#!/usr/bin/env bash
set -euo pipefail

# Real-robot bring-up reference topology.
#
# Keep onboarding docs aligned with this script: each tmux pane maps to one
# runtime responsibility, and rt/ball_state must have exactly one fuser
# publisher.

SESSION_NAME="${1:-robomimic}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
TMUX_SHELL="bash --noprofile --norc"
if [[ -z "${ROBOT_IFACE:-}" ]]; then
  # Use the canonical sysfs name. Some G1 boards expose enP8p1s0 only as
  # an altname of eth0; CycloneDDS must bind the real interface name.
  if [[ -d /sys/class/net/eth0 ]]; then
    ROBOT_IFACE=eth0
  elif [[ -d /sys/class/net/enP8p1s0 ]]; then
    ROBOT_IFACE=enP8p1s0
  else
    ROBOT_IFACE=eth0
  fi
fi
if [[ -z "${UNITREE_SDK2_DIR:-}" ]]; then
  for _sdk in \
    /home/unitree/unitree_sdk2-main \
    /home/unitree/Documents/zhoujiacheng/BFM-Zero-deploy/unitree_sdk2 \
    /home/unitree/unitree_sdk2; do
    if [[ -f "$_sdk/include/unitree/idl/hg/LowCmd_.hpp" ]]; then
      UNITREE_SDK2_DIR="$_sdk"
      break
    fi
  done
  UNITREE_SDK2_DIR="${UNITREE_SDK2_DIR:-/home/unitree/unitree_sdk2-main}"
fi
if [[ -z "${CYCLONEDDS_IDLC:-}" ]]; then
  for _idlc in /opt/ros/humble/bin/idlc /home/unitree/cyclonedds/install/bin/idlc /home/unitree/cyclonedds_ws/install/cyclonedds/bin/idlc; do
    if [[ -x "$_idlc" ]]; then
      CYCLONEDDS_IDLC="$_idlc"
      break
    fi
  done
  CYCLONEDDS_IDLC="${CYCLONEDDS_IDLC:-/opt/ros/humble/bin/idlc}"
fi
PYTHON_BIN="${PYTHON_BIN:-/home/unitree/miniconda3/envs/robomimic/bin/python}"
CONDA_ENV_LIB="${CONDA_ENV_LIB:-/home/unitree/miniconda3/envs/robomimic/lib}"
UNITREE_DDS_LIB="${UNITREE_SDK2_DIR}/thirdparty/lib/$(uname -m)"
if [[ -z "${PY_CYCLONEDDS_LIB:-}" ]]; then
  if [[ -d /home/unitree/share/opt/cyclonedds-0.10.5/lib ]]; then
    PY_CYCLONEDDS_LIB=/home/unitree/share/opt/cyclonedds-0.10.5/lib
  else
    PY_CYCLONEDDS_LIB=/home/unitree/cyclonedds/install/lib
  fi
fi
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
#   left-top    | right-top     bridge/cpp  | gray camera target/ball -> raw topics
#   left-mid    | right-mid     deploy_policy | lidar ball detection → rt/lidar_ball_state
#   left-bottom | right-bottom  sensor dashboard (port 8091) | ball fuser -> rt/ball_state
#
# Data flow:
#   right-top  -> rt/target_state + rt/cam_ball_state
#   right-mid  -> rt/lidar_ball_state
#   right-bot  -> rt/ball_state
#   left-bot   subscribes target/camera/lidar/fused topics -> browser http://<robot>:8091/
LEFT_TOP="$(tmux display-message -p -t "${SESSION_NAME}:0.0" '#{pane_id}')"
RIGHT_TOP="$(tmux split-window -h -P -F '#{pane_id}' -t "${LEFT_TOP}" -c "${REPO_DIR}" "${TMUX_SHELL}")"
LEFT_MID="$(tmux split-window -v -P -F '#{pane_id}' -t "${LEFT_TOP}" -c "${REPO_DIR}" "${TMUX_SHELL}")"
RIGHT_MID="$(tmux split-window -v -P -F '#{pane_id}' -t "${RIGHT_TOP}" -c "${REPO_DIR}" "${TMUX_SHELL}")"
LEFT_BOTTOM="$(tmux split-window -v -P -F '#{pane_id}' -t "${LEFT_MID}" -c "${REPO_DIR}" "${TMUX_SHELL}")"
RIGHT_BOTTOM="$(tmux split-window -v -P -F '#{pane_id}' -t "${RIGHT_MID}" -c "${REPO_DIR}" "${TMUX_SHELL}")"

# Pre-fill commands — press Enter in each pane to start
tmux send-keys -t "${RIGHT_TOP}"    -l "cd ${REPO_DIR} && { pkill -f 'onboard/perception/camera/target_ball_detector.py.*--v4l2-device /dev/video3' || true; pkill -f 'onboard/perception/camera/_launch.sh.*--v4l2-device /dev/video3' || true; if command -v fuser >/dev/null 2>&1; then fuser -k 8080/tcp || true; fuser -k /dev/video3 || true; fi; } && ./onboard/perception/camera/run_gray.sh --show"
tmux send-keys -t "${RIGHT_MID}"    -l "cd ${REPO_DIR} && ./onboard/perception/lidar/run.sh --show --base-y-bias 0.00 --dds-topic rt/lidar_ball_state"
tmux send-keys -t "${LEFT_TOP}"     -l "cd ${REPO_DIR} && rm -rf bridge/build && cmake -S bridge -B bridge/build -DUNITREE_SDK2_DIR=${UNITREE_SDK2_DIR} -DCYCLONEDDS_IDLC=${CYCLONEDDS_IDLC} && cmake --build bridge/build -j2 && LD_LIBRARY_PATH=${UNITREE_DDS_LIB}:\${LD_LIBRARY_PATH:-} CYCLONEDDS_URI='${BRIDGE_CYCLONEDDS_URI}' BRIDGE_NETWORK_INTERFACE=${ROBOT_IFACE} bridge/build/cpp_bridge_main"
tmux send-keys -t "${LEFT_MID}"     -l "cd ${REPO_DIR} && LD_LIBRARY_PATH=${PY_CYCLONEDDS_LIB}:${CONDA_ENV_LIB}:/opt/onnxruntime/lib:\${LD_LIBRARY_PATH:-} CYCLONEDDS_URI='${PY_BRIDGE_CYCLONEDDS_URI}' ROS_LOCALHOST_ONLY=0 \"${PYTHON_BIN}\" bridge/python/deploy_policy.py"
tmux send-keys -t "${LEFT_BOTTOM}"  -l "cd ${REPO_DIR} && bash onboard/perception/run_sensor_dashboard.sh"
tmux send-keys -t "${RIGHT_BOTTOM}" -l "cd ${REPO_DIR} && bash onboard/perception/run_ball_fuser.sh"

# Extra monitor window: check final rt/ball_state stream quickly.
tmux new-window -d -t "${SESSION_NAME}" -n monitor -c "${REPO_DIR}" "${TMUX_SHELL}"
tmux send-keys -t "${SESSION_NAME}:1.0" -l "cd ${REPO_DIR} && ${PYTHON_BIN} tools/check_ball_state.py"

tmux select-pane -t "${LEFT_TOP}"
tmux attach-session -t "${SESSION_NAME}"
