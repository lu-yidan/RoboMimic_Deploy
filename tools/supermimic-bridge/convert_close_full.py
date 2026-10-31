"""Build the closing-segment clip WITH all bodies (the tracker's reward indexes bodies by name).

Root cause found (2026-10-31): the earlier clip stored only the pelvis. SuperMimic's reward terms
(motion_body_pos / motion_body_quat / ...) track a fixed named set (`TRACKED_BODIES` in
tracking_env_cfg.py, standard G1 names), so a single-body clip gives the policy no tracking target:
imag_ret froze at ~0.345 and surv stayed at 0.000 across both clip lengths.

This version runs MuJoCo forward kinematics per frame and stores every body, with names in body-id order.
Quaternions stay wxyz (MuJoCo / mjlab convention, confirmed in src/supermimic/data/motion_convert.py).
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(r'D:\G1Climbing\external\RoboMimic_Deploy')
sys.path.insert(0, str(REPO))

from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv  # noqa: E402

BASE = REPO / 'outputs/calibrated_geometry_training_baseline_20261027_v1'
PLAN = REPO / 'outputs/swing_lift_phase_sine_plan_20261029_v2/plan.npz'
OUT_DIR = Path(r'D:\G1Climbing\supermimic\data\npz\climb_close_full')
CLIP = 'climb_close_full_v1'
T0 = 70.2
FPS = 50.0


def main():
    manifest = json.loads((BASE / 'manifest.json').read_text())
    env = StrictEnv(plan=str(PLAN), cases={'x': {}},
                    normal_bias_y_deg=manifest['normal_bias_y_deg'],
                    memory_enabled=manifest['force_history_memory'])
    model = env.env.model
    scratch = mujoco.MjData(model)

    with np.load(PLAN) as data:
        time = data['time'].astype(np.float64)
        qpos = data['qpos'].astype(np.float64)
    keep = time >= T0
    time, qpos = time[keep] - time[keep][0], qpos[keep]

    dt = 1.0 / FPS
    dense = np.arange(0.0, time[-1], dt)
    resampled = np.empty((dense.size, qpos.shape[1]))
    for k in range(qpos.shape[1]):
        resampled[:, k] = np.interp(dense, time, qpos[:, k])

    nbody = model.nbody
    positions = np.empty((dense.size, nbody, 3))
    quats = np.empty((dense.size, nbody, 4))
    joints = np.empty((dense.size, len(env.env.qadr)))
    joint_vel = np.empty_like(joints)
    lin_vel = np.empty((dense.size, 3))
    ang_vel = np.empty((dense.size, 3))

    for i, state in enumerate(resampled):
        scratch.qpos[:] = state
        mujoco.mj_forward(model, scratch)
        positions[i] = scratch.xpos
        quats[i] = scratch.xquat
        joints[i] = state[env.env.qadr]
    joint_vel = np.gradient(joints, dt, axis=0)
    lin_vel = np.gradient(positions[:, 1], dt, axis=0)      # body 1 = pelvis (body 0 = world)

    def quat_mul(a, b):
        w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                         w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)

    pelvis_quat = quats[:, 1]
    conjugate = pelvis_quat * np.array([1.0, -1.0, -1.0, -1.0])
    delta = quat_mul(pelvis_quat[1:], conjugate[:-1])
    sign = np.where(delta[:, 0:1] < 0, -1.0, 1.0)
    ang_vel = np.vstack([2.0 * delta[:, 1:4] * sign / dt, np.zeros(3)])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / (CLIP + '.npz')
    np.savez_compressed(target,
                        body_pos_w=positions, body_quat_w=quats,
                        joint_pos=joints, joint_vel=joint_vel,
                        body_lin_vel_w=np.repeat(lin_vel[:, None, :], nbody, axis=1),
                        body_ang_vel_w=np.repeat(ang_vel[:, None, :], nbody, axis=1),
                        body_names=np.array([model.body(i).name for i in range(nbody)]))
    meta = dict(clip=CLIP, frames=int(dense.size), fps=FPS, duration_s=float(time[-1] + dt),
                t0=T0, joints=int(joints.shape[1]), bodies=int(nbody))
    (OUT_DIR / (CLIP + '.json')).write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
