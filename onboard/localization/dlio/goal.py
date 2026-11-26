"""Set a live ACP target from a fresh DLIO pose; the chosen world point stays fixed."""
import argparse
import os
from pathlib import Path
import sys
import time
if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from policy.acp.goal_ipc import read_packet, relative_goal, write_goal, validate_status, wait_for_acceptance
from policy.acp.observations import yaw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--relative', nargs=2, type=float, metavar=('FORWARD_M', 'LEFT_M'))
    group.add_argument('--world', nargs=2, type=float, metavar=('X_M', 'Y_M'))
    group.add_argument('--stop', action='store_true', help='Latch current position and heading')
    parser.add_argument('--heading-deg', type=float, default=0., help='Relative yaw change for --relative, absolute yaw for --world')
    parser.add_argument('--ipc-dir', default='/dev/shm/robomimic-%d' % os.getuid())
    args = parser.parse_args()
    try:
        status = read_packet(Path(args.ipc_dir) / 'acp-dlio-status.json')
        validate_status(status, time.monotonic())
        angle = float(np.deg2rad(args.heading_deg))
        if args.relative is not None:
            target, heading = relative_goal(status, *args.relative, angle)
        elif args.world is not None:
            target, heading = args.world, angle
        else:
            target, heading = status['pelvis_pos_w'][:2], yaw(status['pelvis_quat_w'])
        packet = write_goal(args.ipc_dir, status, target, heading)
        print('Goal written; waiting for policy acceptance: ' + packet['request_id'], flush=True)
        ack = wait_for_acceptance(args.ipc_dir, packet)
        print('ACCEPTED by policy PID %s: %s. Target is fixed; Y still requires valid inputs.'
              % (ack['policy_pid'], packet), flush=True)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error) + '; run check.py ON G1 to see the blocking stage')


if __name__ == '__main__':
    main()
