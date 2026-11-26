#!/usr/bin/env bash
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${DISPLAY:-}" ]]; then
  echo 'No graphical DISPLAY in this SSH/tmux pane. Run view_remote.sh in your LOCAL desktop terminal (or connect with ssh -Y).' >&2
  exit 2
fi
# Forwarded X11 uses software GL and cannot use shared-memory X images.
if [[ "$DISPLAY" == localhost:* || "$DISPLAY" == 127.0.0.1:* ]]; then
  export QT_X11_NO_MITSHM=1
  export LIBGL_ALWAYS_SOFTWARE=1
  export __GLX_VENDOR_LIBRARY_NAME=mesa
fi
source /opt/ros/foxy/setup.bash
source "$HERE/transport_env.sh"
# Rendering is diagnostic work on the robot, including forwarded X11.
# Limit Mesa/BLAS worker proliferation and yield CPU to localization/control.
export LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec nice -n 10 rviz2 -d "$HERE/config/acp.rviz"
