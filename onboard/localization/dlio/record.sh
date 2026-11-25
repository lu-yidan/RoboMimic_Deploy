#!/usr/bin/env bash
# Diagnostic subscribers only. The operator separately starts/stops DLIO.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'Usage: bash onboard/localization/dlio/record.sh [seconds: 5-120, default 90]'
  exit 0
fi
DURATION="${1:-90}"
[[ $# -le 1 && "$DURATION" =~ ^[0-9]+$ ]] && (( 10#$DURATION >= 5 && 10#$DURATION <= 120 )) || {
  echo 'Recording duration must be an integer from 5 to 120 seconds' >&2; exit 2;
}
[[ -z "${CONDA_PREFIX:-}" ]] || { echo 'Use env -u CONDA_PREFIX -u LD_LIBRARY_PATH -u PYTHONPATH bash .../record.sh' >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
set +u
source /opt/ros/foxy/setup.bash
set -u
source "$HERE/transport_env.sh"
RECORD_ROOT="${ACP_DLIO_RECORD_ROOT:-$ROOT/logs/acp-dlio}"
mkdir -p "$RECORD_ROOT"
OUT="$(mktemp -d "$RECORD_ROOT/record-$(date +%Y%m%d_%H%M%S)-XXXXXX")"

snapshot() {
  /usr/bin/python3 - "$OUT" "$ROOT" "$1" <<'PY'
import json, os, shutil, subprocess, sys
from pathlib import Path
out, root, stage = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
ipc = Path(os.environ.get('ACP_DLIO_IPC_DIR', '/dev/shm/robomimic-%d' % os.getuid()))
errors = []
if stage == 'start' and shutil.disk_usage(out).free < 2*1024**3:
    raise SystemExit('Need at least 2 GiB free for this diagnostic capture')
def copy(source, name):
    try:
        shutil.copyfile(source, out/name)
    except OSError as error:
        errors.append(str(error))
copy(ipc/'acp-dlio-status.json', stage+'-status.json')
copy(ipc/'acp-waist.json', stage+'-waist.json')
if stage == 'start':
    site = Path(os.environ.get('ACP_DLIO_SITE', '~/.config/robomimic/acp-dlio/site.json')).expanduser()
    copy(site, 'site.json')
    copy(site.with_name('MID360_config.json'), 'MID360_config.json')
    for name in ('dlio.yaml', 'cyclonedds.xml', 'record_qos.yaml'):
        copy(root/'onboard/localization/dlio/config'/name, name)
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(root), text=True, capture_output=True)
    metadata = dict(commit=result.stdout.strip(), ros_domain=os.environ['ROS_DOMAIN_ID'],
                    rmem_max=int(Path('/proc/sys/net/core/rmem_max').read_text()),
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    note='Recorder only; no physical sensor or motor processes started by this script')
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
else:
    try:
        status = json.loads((out/'end-status.json').read_text())
        trace = Path(status.get('diagnostic_trace', '')).resolve()
        trace.relative_to((root/'logs/acp-dlio').resolve())
        copy(trace, 'relay.jsonl')
    except (OSError, ValueError) as error:
        errors.append(str(error))
(out/(stage+'-snapshot-errors.json')).write_text(json.dumps(errors, indent=2)+'\n')
PY
}

snapshot start
echo "Localization recording only, ${DURATION}s: $OUT"
echo 'Start right-middle DLIO now; keep the robot still. Leave the policy stopped.'
trap ':' INT
set +e
timeout --signal=INT --kill-after=10s "${DURATION}s" ros2 bag record \
  -o "$OUT/bag" --qos-profile-overrides-path "$HERE/config/record_qos.yaml" \
  /livox/lidar /livox/imu /acp/livox/points_checked /acp/livox/imu_si \
  /dlio/odom_node/odom /dlio/odom_node/pointcloud/deskewed /acp/pelvis/odom /rosout \
  2>&1 | tee "$OUT/record.log"
record_rc=${PIPESTATUS[0]}
set -e
snapshot end
case "$record_rc" in
  0|124|130) ros2 bag info "$OUT/bag" | tee "$OUT/bag-info.txt" ;;
  *) echo "Recording failed (exit $record_rc); see $OUT/record.log" >&2; exit "$record_rc" ;;
esac
echo "Capture saved: $OUT"
