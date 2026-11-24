"""Explain the current ACP bring-up blocker using host-local IPC; no ROS/motor I/O."""
import argparse
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from onboard.localization.dlio.core import read_json
from policy.acp.state_ipc import boot_id


def assess(status, now, host_boot):
    """Return a stage name and human-readable next action, without relaxing gates."""
    if status.get('boot_id') != host_boot:
        return 'host', 'Run this on G1, in the same host/boot as the relay.'
    delay = now-float(status.get('sampled_at', 0.))
    if not 0 <= delay <= .5:
        return 'relay', 'Status is old: the right-middle DLIO/relay process has stopped or stalled.'
    pose_fresh = status.get('pose_valid') and 0 <= now-float(status.get('pose_captured_at', 0.)) <= .15
    counts = status.get('counts', {})
    if counts and not counts.get('raw_imu', 0):
        return 'imu', 'No raw IMU received: check Livox driver/IP and its log.'
    if counts and not status.get('imu_ready'):
        return 'imu', 'IMU startup check pending: keep still and check acceleration units/norm.'
    if status.get('imu_age_s', 1e9)+delay > .15:
        return 'imu', 'Valid SI IMU is stale (>150 ms); check IMU stage errors.'
    if counts and not counts.get('raw_cloud', 0):
        return 'raw_cloud', 'No raw cloud received: check Livox point output and its log.'
    raw_age = status.get('raw_cloud_age_s')
    if raw_age is not None and raw_age+delay > .3:
        return 'raw_cloud', 'Checked cloud is stale (>300 ms); inspect raw-cloud timing/errors.'
    if counts and not counts.get('checked_cloud', 0):
        return 'raw_cloud', 'Raw clouds were rejected: inspect timestamp units/header/frame error.'
    if status.get('registration_age_s', 1e9)+delay > .25:
        return 'registration', 'Registered DLIO cloud is stale/not initialized (>250 ms); inspect DLIO log.'
    if not pose_fresh:
        return 'pelvis', 'No fresh pelvis: inspect waist export/synchronization and odometry error.'
    if status.get('acp_output_enabled') and not status.get('acp_output_valid'):
        if not status.get('calibration_ready'):
            return 'calibration', 'Fresh pelvis is available. Review mount and measure standing pelvis height, then run calibrate.py.'
        return 'acp_state', 'ACP state is blocked; inspect current error and state file.'
    if not status.get('acp_output_enabled'):
        return 'diagnostic_only', 'Pose-only relay: ACP_DLIO_PUBLISH=1 is needed for the read-only policy test.'
    return 'ready', 'Fresh ACP state available. Start left-middle READ-ONLY policy, set --stop, then B -> Y.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ipc-dir', default='/dev/shm/robomimic-%d' % os.getuid())
    args = parser.parse_args()
    directory = Path(args.ipc_dir).expanduser()
    try:
        status = read_json(directory/'acp-dlio-status.json')
        now = time.monotonic()
        stage, action = assess(status, now, boot_id())
        print('Blocking stage: ' + stage + '\nNext: ' + action)
        print('Error: ' + str(status.get('error', '')))
        print('Stage errors: ' + str(status.get('stage_errors', {})))
        print('Counts: ' + str(status.get('counts', {})))
        print('Age (ms): status=%.1f, IMU=%.1f, registration=%.1f, pose=%.1f' % (
            1000*(now-status['sampled_at']), 1000*(now-status['sampled_at']+status.get('imu_age_s', 0)),
            1000*(now-status['sampled_at']+status.get('registration_age_s', 0)), 1000*(now-status.get('pose_captured_at', 0))))
        print('pose_valid=%s acp_output_valid=%s' % (status.get('pose_valid'), status.get('acp_output_valid')))
        try:
            waist = read_json(directory/'acp-waist.json')
            print('Waist unique tick=%s, age=%.1f ms' % (waist['rows'][-1][1], 1000*(now-waist['rows'][-1][0])))
        except (OSError, KeyError, IndexError) as error:
            print('Waist export unavailable: ' + str(error))
        print('ACP state file exists: ' + str((directory/'acp-state.npz').exists()))
        print('Diagnostic trace: ' + str(status.get('diagnostic_trace', '(update relay to enable)')))
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error) + '; run this ON G1 after starting bridge, waist export and DLIO')


if __name__ == '__main__':
    main()
