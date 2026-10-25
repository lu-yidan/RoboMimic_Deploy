#!/usr/bin/env bash
set -euo pipefail

# Real-robot bring-up reference topology.
#
# Keep onboarding docs aligned with this script: each tmux pane maps to one
# runtime responsibility, and rt/ball_state must have exactly one fuser
# publisher.

SESSION_NAME="${1:-robomimic}"
DETACH="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
printf -v TMUX_SHELL 'bash --noprofile --rcfile %q -i' "${REPO_DIR}/tools/deploy/tmux.bashrc"
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

if tmux has-session -t "=${SESSION_NAME}" 2>/dev/null; then
  [[ "$DETACH" == "--detach" ]] && exit 0
  if [[ -n "${TMUX:-}" ]]; then
    tmux switch-client -t "${SESSION_NAME}"
  else
    tmux attach-session -t "${SESSION_NAME}"
  fi
  exit 0
fi

tmux new-session -d -s "${SESSION_NAME}" -n main -c "${REPO_DIR}" "${TMUX_SHELL}"

# Layout (2 columns × 3 rows):
#   left-top    | right-top     bridge/cpp  | camera.mode: gray / PHP depth
#   left-mid    | right-mid     deploy_policy | lidar ball detection → rt/lidar_ball_state
#   left-bottom | right-bottom  sensor dashboard (port 8091) | ball fuser -> rt/ball_state
#
# Data flow:
#   right-top  -> gray DDS topics OR PHP depth IPC + Dashboard preview
#   right-mid  -> rt/lidar_ball_state
#   right-bot  -> rt/ball_state
#   left-bot   subscribes target/camera/lidar/fused topics -> browser http://<robot>:8091/
LEFT_TOP="$(tmux display-message -p -t "${SESSION_NAME}:0.0" '#{pane_id}')"
RIGHT_TOP="$(tmux split-window -h -P -F '#{pane_id}' -t "${LEFT_TOP}" -c "${REPO_DIR}" "${TMUX_SHELL}")"
LEFT_MID="$(tmux split-window -v -P -F '#{pane_id}' -t "${LEFT_TOP}" -c "${REPO_DIR}" "${TMUX_SHELL}")"
RIGHT_MID="$(tmux split-window -v -P -F '#{pane_id}' -t "${RIGHT_TOP}" -c "${REPO_DIR}" "${TMUX_SHELL}")"
LEFT_BOTTOM="$(tmux split-window -v -P -F '#{pane_id}' -t "${LEFT_MID}" -c "${REPO_DIR}" "${TMUX_SHELL}")"
RIGHT_BOTTOM="$(tmux split-window -v -P -F '#{pane_id}' -t "${RIGHT_MID}" -c "${REPO_DIR}" "${TMUX_SHELL}")"

# One layout for every policy. PHP enablement is read from LYD_DEPLOY_CONFIG.
printf -v repo_q '%q' "${REPO_DIR}"
printf -v python_q '%q' "${PYTHON_BIN}"
printf -v config_q '%q' "${LYD_DEPLOY_CONFIG:-${REPO_DIR}/config/deploy.yaml}"
printf -v sdk_q '%q' "${UNITREE_SDK2_DIR}"
printf -v idlc_q '%q' "${CYCLONEDDS_IDLC}"
printf -v cpp_lib_q '%q' "${UNITREE_DDS_LIB}"
printf -v py_lib_q '%q' "${PY_CYCLONEDDS_LIB}:${CONDA_ENV_LIB}:/opt/onnxruntime/lib"
printf -v cpp_uri_q '%q' "${BRIDGE_CYCLONEDDS_URI}"
printf -v py_uri_q '%q' "${PY_BRIDGE_CYCLONEDDS_URI}"
printf -v iface_q '%q' "${ROBOT_IFACE}"
# Literal prefill only; no Enter and no process-killing or device cleanup.
tmux send-keys -t "${LEFT_TOP}" -l "cd ${repo_q} && cmake -S bridge -B bridge/build -DUNITREE_SDK2_DIR=${sdk_q} -DCYCLONEDDS_IDLC=${idlc_q} && cmake --build bridge/build -j2 && LD_LIBRARY_PATH=${cpp_lib_q}:\${LD_LIBRARY_PATH:-} CYCLONEDDS_URI=${cpp_uri_q} BRIDGE_NETWORK_INTERFACE=${iface_q} bridge/build/cpp_bridge_main"
tmux send-keys -t "${LEFT_MID}" -l "cd ${repo_q} && LYD_DEPLOY_CONFIG=${config_q} LD_LIBRARY_PATH=${py_lib_q}:\${LD_LIBRARY_PATH:-} CYCLONEDDS_URI=${py_uri_q} ROS_LOCALHOST_ONLY=0 ${python_q} bridge/python/deploy_policy.py"
tmux send-keys -t "${LEFT_BOTTOM}" -l "cd ${repo_q} && LYD_DEPLOY_CONFIG=${config_q} LD_LIBRARY_PATH=${py_lib_q}:\${LD_LIBRARY_PATH:-} CYCLONEDDS_URI=${py_uri_q} ${python_q} onboard/perception/debug/sensor_dashboard.py"
tmux send-keys -t "${RIGHT_TOP}" -l "cd ${repo_q} && LYD_DEPLOY_CONFIG=${config_q} LD_LIBRARY_PATH=${py_lib_q}:\${LD_LIBRARY_PATH:-} CYCLONEDDS_URI=${py_uri_q} ${python_q} tools/capture/run_camera.py"
tmux send-keys -t "${RIGHT_MID}" -l "cd ${repo_q} && ./onboard/perception/lidar/run.sh --show --base-y-bias 0.00 --dds-topic rt/lidar_ball_state"
tmux send-keys -t "${RIGHT_BOTTOM}" -l "cd ${repo_q} && bash onboard/perception/run_ball_fuser.sh"

# Existing extra monitoring window (the main window still has six panes).
tmux new-window -d -t "${SESSION_NAME}" -n monitor -c "${REPO_DIR}" "${TMUX_SHELL}"
tmux send-keys -t "${SESSION_NAME}:1.0" -l "cd ${repo_q} && LD_LIBRARY_PATH=${py_lib_q}:\${LD_LIBRARY_PATH:-} CYCLONEDDS_URI=${py_uri_q} ${python_q} tools/diagnostics/check_ball_state.py"

# Stable pane-local labels: terminal title escape sequences cannot overwrite them.
label_pane() {
  tmux set-option -p -t "$1" @purpose "$2"
}
label_pane "${LEFT_TOP}" '左上 Bridge | 电机通信'
label_pane "${LEFT_MID}" '左中 Policy | 策略与手柄'
label_pane "${LEFT_BOTTOM}" '左下 Dashboard | 网页 :8091'
label_pane "${RIGHT_TOP}" '右上 Camera | gray / php_depth'
label_pane "${RIGHT_MID}" '右中 LiDAR | 雷达球检测'
label_pane "${RIGHT_BOTTOM}" '右下 Fuser | 球位置融合'
label_pane "${SESSION_NAME}:1.0" 'Monitor | 最终球状态（DDS）'
for window in "${SESSION_NAME}:0" "${SESSION_NAME}:1"; do
  tmux set-option -w -t "${window}" pane-border-status top
  tmux set-option -w -t "${window}" pane-border-format ' #{pane_index} | #{@purpose} '
done

tmux select-pane -t "${LEFT_TOP}"
if [[ "$DETACH" != "--detach" ]]; then
  if [[ -n "${TMUX:-}" ]]; then
    tmux switch-client -t "${SESSION_NAME}"
  else
    tmux attach-session -t "${SESSION_NAME}"
  fi
fi
