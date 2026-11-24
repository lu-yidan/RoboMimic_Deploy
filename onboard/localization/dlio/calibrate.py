"""Set a measured flat floor for the CURRENT relay session; no actuator interfaces."""
import argparse
import math
import os
from pathlib import Path
import sys
import time

if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from onboard.localization.dlio.core import atomic_json, read_json
from policy.acp.goal_ipc import validate_status


def validate_calibration(status, height, now):
    validate_status(status, now)
    if not math.isfinite(height) or not .3 < height < 1.2:
        raise ValueError('Measured pelvis height must be between 0.3 and 1.2 m')
    delay = now-status['sampled_at']
    for name, key, limit in [('IMU', 'imu_age_s', .15), ('raw cloud', 'raw_cloud_age_s', .3),
                             ('registration', 'registration_age_s', .25)]:
        age = status.get(key)
        if age is None or not math.isfinite(float(age)) or not 0 <= float(age)+delay <= limit:
            raise ValueError('%s not fresh for calibration: source_age=%s s, status_age=%.3f s, limit=%.3f s' % (
                name, age, delay, limit))
    speed, turn = status.get('speed_mps'), status.get('turn_rate_rps')
    if speed is None or turn is None or not all(math.isfinite(float(v)) for v in (speed, turn)):
        raise ValueError('Stationarity estimate not initialized or nonfinite')
    if not 0 <= speed <= .05 or not 0 <= turn <= .1:
        raise ValueError('Standing motion estimate exceeds limit: speed=%.4f m/s (<=0.05), '
                         'turn_rate=%.4f rad/s (<=0.10); keep still and inspect localization' % (speed, turn))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pelvis-height', required=True, type=float, help='Measured pelvis origin height above flat floor, metres')
    parser.add_argument('--confirm-mount', action='store_true', required=True,
                        help='Confirm configured head/mount angles, URDF chain and SDK waist signs were reviewed onsite')
    parser.add_argument('--ipc-dir', default='/dev/shm/robomimic-%d' % os.getuid())
    args = parser.parse_args()
    directory = Path(args.ipc_dir)
    try:
        status = read_json(directory / 'acp-dlio-status.json')
        now = time.monotonic()
        validate_calibration(status, args.pelvis_height, now)
        result = dict(session=status['session'], calibrated_at=now, mount_reviewed=True,
                      measured_pelvis_height=args.pelvis_height,
                      ground_height=status['pelvis_pos_w'][2]-args.pelvis_height,
                      mount_angles_rad=status['mount_angles_rad'])
        atomic_json(directory / 'acp-dlio-calibration.json', result)
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.error(str(error))
    print(result)


if __name__ == '__main__':
    main()
