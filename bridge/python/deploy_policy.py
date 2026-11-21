import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import time
import argparse

from bridge.python.bridge_cmd_dds import BridgeCmdPublisher
from deploy_real.config import Config
from bridge.python.policy_runtime import PolicyRuntime


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the bridge policy/FSM process")
    parser.add_argument('--read-only', action='store_true',
                        help='Infer and log without creating a BridgeCmd publisher')
    args = parser.parse_args(argv)
    config = Config()
    config.bridge_read_only = args.read_only
    runtime = PolicyRuntime(config)
    cmd_pub = None if args.read_only else BridgeCmdPublisher(
        domain_id=config.bridge_domain_id,
        topic_name=config.bridge_cmd_topic,
    )
    if args.read_only:
        print('[Bridge] Read-only policy: inference/logging only; no BridgeCmd publisher. '
              'The C++ bridge must also use BRIDGE_READ_ONLY=1.')

    try:
        while True:
            loop_start = time.time()
            frame = runtime.step()
            if cmd_pub is not None:
                cmd_pub.publish(
                    seq=frame.seq,
                    q_des=frame.q_des,
                    kp=frame.kp,
                    kd=frame.kd,
                    request_damping=frame.request_damping,
                    exit_requested=frame.exit_requested,
                )
            if frame.exit_requested:
                break

            loop_dt = time.time() - loop_start
            if loop_dt < config.control_dt:
                time.sleep(config.control_dt - loop_dt)
    except KeyboardInterrupt:
        if runtime._logger is not None:
            runtime._logger.event("keyboard_interrupt", time_s=time.monotonic()-runtime._log_start)
    except Exception as error:
        if runtime._logger is not None:
            runtime._logger.event("runtime_exception", time_s=time.monotonic()-runtime._log_start,
                                  error_type=type(error).__name__, message=str(error))
        raise
    finally:
        try:
            runtime.close()
        finally:
            if cmd_pub is not None:
                cmd_pub.publish(
                    seq=0,
                    q_des=[0.0] * config.num_joints,
                    kp=[0.0] * config.num_joints,
                    kd=[8.0] * config.num_joints,
                    request_damping=True,
                    exit_requested=True,
                )
        print("Exit")


if __name__ == "__main__":
    main()
