#!/usr/bin/env bash
# ============================================================
# Start grayscale camera perception:
#   - AprilTag target -> rt/target_state
#   - bright camera ball -> rt/cam_ball_state
#   - optional fuser -> rt/ball_state
#
# By default this uses the single-process detector so one owner opens the
# RealSense GREY V4L2 node. Pass --with-fuser to also start ball_fuser.py.
# ============================================================
set -e
cd "$(dirname "$0")/../../.."

with_fuser=0
args=()
for arg in "$@"; do
    case "$arg" in
        --with-fuser)
            with_fuser=1
            ;;
        *)
            args+=("$arg")
            ;;
    esac
done

if [[ "$with_fuser" -eq 1 ]]; then
    camera_pid=""
    fuser_pid=""
    shutting_down=0

    cleanup() {
        local status=$?
        if [[ "$shutting_down" -eq 1 ]]; then
            return "$status"
        fi
        shutting_down=1

        echo
        echo "[run_gray_perception.sh] Stopping managed perception processes..."
        for pid in "${camera_pid:-}" "${fuser_pid:-}"; do
            if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
                # Children are started with setsid, so -PID targets the whole
                # process group (bash/conda/python and multiprocessing workers).
                kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
            fi
        done

        sleep 1
        for pid in "${camera_pid:-}" "${fuser_pid:-}"; do
            if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
                kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
            fi
        done

        wait "${camera_pid:-}" 2>/dev/null || true
        wait "${fuser_pid:-}" 2>/dev/null || true
        echo "[run_gray_perception.sh] Done."
        return "$status"
    }
    trap cleanup INT TERM EXIT

    fuser_log="${FUSER_LOG:-/tmp/ball_fuser.log}"
    echo "[run_gray_perception.sh] Starting fuser... log=$fuser_log"
    : > "$fuser_log"
    setsid bash onboard/perception/run_ball_fuser.sh > "$fuser_log" 2>&1 &
    fuser_pid=$!
    echo "[run_gray_perception.sh] fuser pid=$fuser_pid pgid=$fuser_pid"

    echo "[run_gray_perception.sh] Starting AprilTag/bright-ball camera..."
    setsid bash onboard/perception/camera/run_apriltag_gray_ball.sh \
        "${args[@]}" &
    camera_pid=$!
    echo "[run_gray_perception.sh] camera pid=$camera_pid pgid=$camera_pid"

    echo "[run_gray_perception.sh] Press Ctrl+C to stop camera and fuser."
    wait -n "$camera_pid" "$fuser_pid"
    exit $?
fi

exec bash onboard/perception/camera/run_apriltag_gray_ball.sh \
    "${args[@]}"
