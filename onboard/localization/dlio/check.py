"""Explain the current ACP bring-up blocker using host-local IPC; no ROS/motor I/O."""
import argparse
from collections import Counter
import math
import os
from pathlib import Path
import sys
import time

if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
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
    if status.get('localization_fault') or 'DLIO discontinuity/reset' in str(status.get('stage_errors', {}).get('odom', '')):
        return 'localization_reset', 'DLIO jump/reset is latched. Restart right-middle DLIO; review the fault details. Recalibrate ground/targets only after stable fresh poses.'
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
    parser.add_argument('--duration', type=float, default=0., help='Observe 0–10 seconds; report intermittent blockers as well as the final sample')
    args = parser.parse_args()
    if not math.isfinite(args.duration) or not 0 <= args.duration <= 10:
        parser.error('--duration must be between 0 and 10 seconds')
    directory = Path(args.ipc_dir).expanduser()
    try:
        host = boot_id()
        started, stages, fresh_poses = time.monotonic(), Counter(), 0
        while True:
            status = read_json(directory/'acp-dlio-status.json')
            now = time.monotonic()
            stage, action = assess(status, now, host)
            stages[stage] += 1
            fresh_poses += bool(status.get('boot_id') == host and status.get('pose_valid')
                                and 0 <= now-status.get('pose_captured_at', 0.) <= .15)
            if now-started >= args.duration:
                break
            time.sleep(min(.02, args.duration-(now-started)))
        if args.duration:
            print('Window %.2f s: %s; fresh pelvis %.1f%% (%d samples)' % (
                now-started, dict(stages), 100*fresh_poses/sum(stages.values()), sum(stages.values())))
        print('Blocking stage: ' + stage + '\nNext: ' + action)
        print('Error: ' + str(status.get('error', '')))
        print('Stage errors: ' + str(status.get('stage_errors', {})))
        if status.get('localization_fault_details'):
            print('First localization fault: ' + str(status['localization_fault_details']))
        print('Counts: ' + str(status.get('counts', {})))
        delay = now-status['sampled_at']
        ages = dict(status=delay, IMU=delay+status.get('imu_age_s', 0),
                    raw_cloud=None if status.get('raw_cloud_age_s') is None else delay+status['raw_cloud_age_s'],
                    registration=delay+status.get('registration_age_s', 0), pose=now-status.get('pose_captured_at', 0))
        print('Age (ms): ' + ', '.join(k+'='+('n/a' if v is None else '%.1f' % (1000*v)) for k, v in ages.items()))
        print('pose_valid=%s acp_output_valid=%s' % (status.get('pose_valid'), status.get('acp_output_valid')))
        print('Motion estimate: speed=%s m/s (calibration <=0.05), turn_rate=%s rad/s (<=0.10)' % (
            status.get('speed_mps'), status.get('turn_rate_rps')))
        try:
            waist = read_json(directory/'acp-waist.json')
            waist_now = time.monotonic()  # Sample after reading: the exporter can update during the read.
            print('Waist unique tick=%s, age=%.1f ms' % (waist['rows'][-1][1], 1000*(waist_now-waist['rows'][-1][0])))
        except (OSError, KeyError, IndexError) as error:
            print('Waist export unavailable: ' + str(error))
        print('ACP state file exists: ' + str((directory/'acp-state.npz').exists()))
        print('Diagnostic trace: ' + str(status.get('diagnostic_trace', '(update relay to enable)')))
        if 'raw_cloud_receive_age_s' in status:
            print('Cloud ages AT ARRIVAL (s): raw=%s, registration=%s; max arrival gaps (s): raw=%s, registration=%s' % (
                status.get('raw_cloud_receive_age_s'), status.get('registration_receive_age_s'),
                status.get('raw_cloud_max_gap_s'), status.get('registered_cloud_max_gap_s')))
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error) + '; run this ON G1 after starting bridge, waist export and DLIO')


if __name__ == '__main__':
    main()
