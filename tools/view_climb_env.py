"""Inspect the mounted scene; optionally run the exported support-pose actor.

Starts paused at the training initial state. Space toggles physical stepping;
R resets. Each episode stops after its declared horizon or an early failure.
This viewer never communicates with robot hardware.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import queue
import time
import mujoco.viewer
from policy.climb.Climb import OnnxActor
from training.climb_env import ClimbSupportEnv, DEFAULT_BUNDLE

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument('--export', type=Path, default=ROOT/'outputs/climb_ppo_smoke_v1/export')
    args = parser.parse_args()
    env = ClimbSupportEnv(args.bundle)
    actor = OnnxActor(args.export/'policy.onnx', args.export/'metadata.json', env.contract)
    observation, _ = env.reset(seed=20261009)
    events = queue.SimpleQueue()
    playing = False
    try:
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=events.put,
                show_left_ui=False, show_right_ui=False) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [.29, 0., .88]
                viewer.cam.distance = 2.45
                viewer.cam.azimuth = -36.
                viewer.cam.elevation = -12.
            viewer.sync()
            print('READY: mounted-hook training scene; paused at initial support pose.', flush=True)
            print('SPACE: run/pause ONNX policy. R: reset. Episode stops at 4 seconds.', flush=True)
            print('Mouse: rotate/pan/zoom. No robot hardware connection.', flush=True)
            while viewer.is_running():
                started = time.perf_counter()
                while not events.empty():
                    key = events.get_nowait()
                    if key in (82, 114):
                        with viewer.lock():
                            observation, _ = env.reset(seed=20261009)
                        playing = False
                        print('RESET: paused at initial pose.', flush=True)
                    elif key == 32:
                        if env.finished:
                            print('Episode finished. Press R to reset.', flush=True)
                        else:
                            playing = not playing
                if playing:
                    with viewer.lock():
                        action = actor(observation)
                        observation, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        playing = False
                        print(json.dumps(info), flush=True)
                viewer.sync()
                time.sleep(max(0., env.contract.control_dt-(time.perf_counter()-started)))
    finally:
        env.close()


if __name__ == '__main__':
    main()
