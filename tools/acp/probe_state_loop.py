"""Bounded synthetic IPC + real ACP inference check, with no ROS/DDS or actuators.

Uses a private temporary directory and a separate writer process. Does not
construct PolicyRuntime, FSM, subscribers or publishers. This is not a full
control-loop or physical-motion acceptance test.
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import yaml

from bridge.python.policy_runtime import PolicyRuntime
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.utils import FSMCommand, FSMStateName
from policy.acp.ACP import ACP
from policy.acp.robot_history import RobotHistory
from policy.acp.state_ipc import StateReader, StateWriter


def produce(path, stop, ready):
    writer = StateWriter(path, 'acp_odom')
    sequence = 0
    while not stop.is_set():
        then = time.monotonic()
        writer.write(pelvis_pos_w=[0., 0., .78], pelvis_quat_w=[1., 0., 0., 0.],
                     captured_at=then-.02, sequence=sequence, ground_height=0.)
        ready.set()
        sequence += 1
        stop.wait(max(0., .02-(time.monotonic()-then)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--mode', choices=['cached', 'sync'], default='sync')
    args = parser.parse_args()
    if not 2 <= args.seconds <= 60:
        parser.error('seconds must be within [2, 60]')
    config = yaml.safe_load((REPO/'config/deploy_acp_dlio.yaml').read_text())['acp']
    config['model_dir'] = str(args.model_dir.resolve())
    names = yaml.safe_load((REPO/'policy/php_parkour/config/php.yaml').read_text())['joint_names']
    runtime = PolicyRuntime.__new__(PolicyRuntime)
    runtime.acp_enabled, runtime.acp_config, runtime.connect = True, config, False
    runtime.state_cmd, runtime.policy_output = StateAndCmd(29), PolicyOutput(29)
    policy = ACP(runtime.state_cmd, runtime.policy_output, config, names)
    policy.warmup()
    policy.enter()
    runtime.FSM_controller = SimpleNamespace(acp_policy=policy, cur_policy=policy)
    runtime._acp_robot_history = RobotHistory(config['max_age_s'], config['max_sync_skew_s'])
    runtime._acp_goal_reader = runtime._logger = runtime._acp_error = None
    runtime._acp_poll_in_step = args.mode == 'sync'
    runtime._acp_effort_limits = policy.effort_limits
    runtime._cmd_seq = 0
    records, failures = [], []
    context = mp.get_context('spawn')
    stop, ready = context.Event(), context.Event()
    with tempfile.TemporaryDirectory(prefix='acp-private-probe-') as folder:
        path = Path(folder)/'state.npz'
        runtime.acp_source = StateReader(config, path)
        producer = context.Process(target=produce, args=(path, stop, ready))
        producer.start()
        try:
            if not ready.wait(10):
                raise RuntimeError('Private writer did not start')
            if args.mode == 'cached':
                runtime.acp_source.start()
            tick, started = 0, time.monotonic()
            while time.monotonic()-started < args.seconds:
                then = time.monotonic()
                tick += 1
                state = SimpleNamespace(tick=tick, q=policy.default.copy(), dq=np.zeros(29),
                                        imu_quat_wxyz=[1., 0., 0., 0.], imu_gyro=[0., 0., 0.],
                                        _local_received_monotonic=then)
                runtime._acp_tick_time = then
                runtime._apply_robot_state(state)
                runtime._apply_acp_state(state)
                read_done = time.monotonic()
                # Fill the 20 ms synthetic localization history before scoring.
                if then-started > .2:
                    policy.run()
                    done = time.monotonic()
                    if policy.failed:
                        failures.append(dict(at=then-started, reason=policy.last_error))
                        break
                    assert not runtime._guard_acp_output()
                    records.append([then, (read_done-then)*1000, (time.monotonic()-then)*1000,
                                    (done-runtime.state_cmd.acp_pose_time)*1000, policy.last_inference_ms])
                time.sleep(max(0., .02-(time.monotonic()-then)))
            stopped_at = time.monotonic()
            stop.set()
            detected = None
            while time.monotonic()-stopped_at < .5:
                now = time.monotonic()
                tick += 1
                state.tick, state._local_received_monotonic = tick, now
                runtime._acp_tick_time = now
                runtime._apply_acp_state(state)
                policy.run()
                if policy.failed:
                    detected = dict(after_stop_s=time.monotonic()-stopped_at, reason=policy.last_error,
                                    source_error=runtime._acp_error,
                                    passive_requested=runtime.state_cmd.skill_cmd == FSMCommand.PASSIVE,
                                    zero_kp=bool(np.all(runtime.policy_output.kps == 0)),
                                    damping_kd=bool(np.all(runtime.policy_output.kds == 8)))
                    break
                time.sleep(.01)
        finally:
            stop.set()
            producer.join(5)
            if producer.is_alive():
                producer.terminate()
                producer.join(5)
            runtime.acp_source.close()
    a = np.asarray(records)
    summary = dict(synthetic=True,hardware_started=False,ros_dds_started=False,motor_control_started=False,
                   mode=args.mode, requested_duration_s=args.seconds,frames=len(records),failures=failures,
                   stopped_producer=detected,
                   limitations='Synthetic inputs and partial runtime; excludes full FSM, DDS, live localization, normal logging and physical motion.')
    if len(a):
        summary.update(percentiles=[50,90,99,100],
                       timing_ms={name:np.percentile(a[:,i],[50,90,99,100]).tolist()
                                  for i,name in enumerate(['state_apply','compute','pose_age','inference'],1)},
                       compute_over_20ms_fraction=float(np.mean(a[:,2]>20)))
    summary['passed'] = bool(len(records)>args.seconds*40 and not failures and detected and
                             detected['after_stop_s'] < .25 and detected['passive_requested'] and
                             detected['zero_kp'] and detected['damping_kd'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
