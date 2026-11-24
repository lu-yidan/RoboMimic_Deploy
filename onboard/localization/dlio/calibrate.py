"""Set a measured flat floor for the CURRENT relay session; no actuator interfaces."""
import argparse
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from onboard.localization.dlio.core import atomic_json, read_json
from policy.acp.state_ipc import boot_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pelvis-height', required=True, type=float, help='Measured pelvis origin height above flat floor, metres')
    parser.add_argument('--confirm-mount', action='store_true', required=True,
                        help='Confirm configured head/mount angles, URDF chain and SDK waist signs were reviewed onsite')
    parser.add_argument('--ipc-dir', default='/dev/shm/robomimic-%d' % os.getuid())
    args = parser.parse_args()
    directory = Path(args.ipc_dir)
    status = read_json(directory / 'acp-dlio-status.json')
    now = time.monotonic()
    if (not status['pose_valid'] or status['boot_id'] != boot_id() or status['frame_id'] != 'acp_odom'
            or not 0 <= now-status['pose_captured_at'] <= .15 or not .3 < args.pelvis_height < 1.2
            or status.get('speed_mps') is None or status['speed_mps'] > .05
            or status.get('turn_rate_rps') is None or status['turn_rate_rps'] > .1):
        parser.error('Need a fresh stationary standing pose and a measured 0.3–1.2 m pelvis height')
    result = dict(session=status['session'], calibrated_at=now, mount_reviewed=True,
                  measured_pelvis_height=args.pelvis_height,
                  ground_height=status['pelvis_pos_w'][2]-args.pelvis_height,
                  mount_angles_rad=status['mount_angles_rad'])
    atomic_json(directory / 'acp-dlio-calibration.json', result)
    print(result)


if __name__ == '__main__':
    main()
