#!/usr/bin/env bash
# Run on a LOCAL Linux graphical desktop. ROS and RViz run on G1, GUI forwards.
set -euo pipefail
[[ -n "${DISPLAY:-}" ]] || { echo 'Run this in a local graphical desktop terminal with DISPLAY set.' >&2; exit 2; }
HOST="${1:-g1-wifi}"
# A previously opened non-X11 master cannot acquire a forwarded DISPLAY.
exec ssh -Y -o ControlMaster=no -o ControlPath=none -o ExitOnForwardFailure=yes "$HOST" \
  'env -u CONDA_PREFIX -u LD_LIBRARY_PATH -u PYTHONPATH bash /home/unitree/RoboMimic_Deploy/onboard/localization/dlio/view.sh'
