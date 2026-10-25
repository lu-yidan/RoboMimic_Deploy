"""Headless PHP: 2 seconds standing, then 8 seconds forward, on flat/steps.

Run with MUJOCO_GL=egl in robomimic. This never opens DDS or robot interfaces.
"""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
import torch
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.deployment_config import load_php_config
from policy.php_parkour.PHPParkour import PHPParkour
from policy.php_parkour.depth import DepthCamera
from deploy_mujoco.scene_builder import load_model


def run_scene(scene):
    model = load_model(ROOT / 'g1_description/scene.xml', {'name': scene})
    model.opt.timestep = .002
    data = mujoco.MjData(model)
    state, output = StateAndCmd(29), PolicyOutput(29)
    names = [model.joint(int(model.actuator_trnid[i, 0])).name for i in range(29)]
    policy = PHPParkour(state, output, load_php_config(), names)
    camera = DepthCamera(model)
    data.qpos[7:36] = policy.default
    data.qpos[2] = .78
    mujoco.mj_forward(model, data)
    policy.enter()
    limits = np.array([88, 139, 88, 139, 50, 50] * 2 + [88, 50, 50]
                      + [25, 25, 25, 25, 25, 5, 5] * 2)
    start = time.monotonic()
    frames, min_height, max_height, failed = 0, 10., 0., False
    try:
        for tick in range(5000):
            if tick % 10 == 0:
                mujoco.mj_forward(model, data)
                state.q = data.qpos[7:36].copy()
                state.dq = data.qvel[6:35].copy()
                state.ang_vel = data.qvel[3:6].copy()
                state.torso_quat_w = data.body('torso_link').xquat.copy()
                state.vel_cmd[:] = [1 if tick >= 1000 else 0, 0, 0]
                camera.capture(data, state)
                policy.run()
                frames += 1
                if policy.failed:
                    failed = True
                    break
            torque = output.kps * (output.actions - data.qpos[7:36]) - output.kds * data.qvel[6:35]
            data.ctrl[:] = np.clip(torque, -limits, limits)
            mujoco.mj_step(model, data)
            min_height = min(min_height, data.qpos[2])
            max_height = max(max_height, data.qpos[2])
            if not np.isfinite(data.qpos).all() or data.qpos[2] < .3:
                failed = True
                break
    finally:
        camera.close()
    return {'scene': scene, 'sim_seconds': data.time, 'position': data.qpos[:3].tolist(),
            'min_height': min_height, 'max_height': max_height, 'failed_or_fell': failed,
            'frames': frames, 'wall_seconds': time.monotonic() - start}


if __name__ == '__main__':
    torch.set_num_threads(1)
    results = [run_scene(scene) for scene in ('none', 'php_steps')]
    for result in results:
        print(json.dumps(result))
    sys.exit(1 if any(r['failed_or_fell'] for r in results) else 0)
