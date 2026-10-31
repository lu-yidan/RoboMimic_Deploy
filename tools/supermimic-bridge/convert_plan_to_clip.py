"""Convert our climbing reference (plan.npz) into the SuperMimic-Reproduced clip format.

Required npz keys (from scripts/data/build_library.py:27):
    body_pos_w[:, r]   (T,3)   root position, world
    body_quat_w[:, r]  (T,4)   root quaternion, world
    joint_pos          (T,29)  joint positions
    body_lin_vel_w[:, r] (T,3) root linear velocity
    body_ang_vel_w[:, r] (T,3) root angular velocity
    joint_vel          (T,29)  joint velocities
The library packs these into 71-d reference states (3+4+29+3+3+29).

Our plan.npz stores qpos (T,36) = [base_pos(3), base_quat(4), joints(29)] at dt = 0.02 s, so velocities are
finite differences. NOTE: 29 joints is the G1 layout used by this project; verify it matches the motion
library's joint order before training.

Status: written from the format spec, NOT yet exercised against the reproduction's trainer.
"""
import json
from pathlib import Path

import numpy as np

PLAN = Path(r'D:\G1Climbing\external\RoboMimic_Deploy\outputs\swing_lift_phase_sine_plan_20261029_v2\plan.npz')
OUT_DIR = Path(r'D:\G1Climbing\supermimic\data\npz\climb')
CLIP = 'climb_cycle_v1'


def main():
    with np.load(PLAN) as data:
        time = data['time'].astype(np.float64)
        qpos = data['qpos'].astype(np.float64)
    assert qpos.shape[1] == 36, qpos.shape
    src_dt = float(np.median(np.diff(time)))
    # 计划节点只有 ~10 Hz（1036 帧 / 113.66 s），跟踪器要 50 Hz 的密集目标 -> 重采样
    target_fps = 50.0
    dt = 1.0 / target_fps
    fps = target_fps
    dense_time = np.arange(time[0], time[-1], dt)
    resampled = np.empty((dense_time.size, qpos.shape[1]))
    for k in range(qpos.shape[1]):
        resampled[:, k] = np.interp(dense_time, time, qpos[:, k])
    qpos = resampled
    time = dense_time

    pos = qpos[:, 0:3]
    quat = qpos[:, 3:7]           # wxyz, as stored by MuJoCo
    joints = qpos[:, 7:36]
    lin_vel = np.gradient(pos, dt, axis=0)
    joint_vel = np.gradient(joints, dt, axis=0)

    # angular velocity from quaternion finite difference (wxyz)
    def quat_mul(a, b):
        w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return np.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], axis=-1)

    conjugate = quat * np.array([1.0, -1.0, -1.0, -1.0])
    delta = quat_mul(quat[1:], conjugate[:-1])          # rotation from t to t+1
    vec_part = delta[:, 1:4]
    sign = np.where(delta[:, 0:1] < 0, -1.0, 1.0)
    ang_vel = 2.0 * vec_part * sign / dt
    ang_vel = np.vstack([ang_vel, ang_vel[-1:]])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / (CLIP + '.npz')
    np.savez_compressed(
        target,
        body_pos_w=pos[:, None, :],
        body_quat_w=quat[:, None, :],
        joint_pos=joints,
        body_lin_vel_w=lin_vel[:, None, :],
        body_ang_vel_w=ang_vel[:, None, :],
        joint_vel=joint_vel,
        body_names=np.array(["pelvis"]),
    )
    meta = dict(clip=CLIP, frames=int(qpos.shape[0]), fps=fps, duration_s=float(time[-1]),
                joints=int(joints.shape[1]), source=str(PLAN))
    (OUT_DIR / (CLIP + '.json')).write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))
    print('wrote', target)


if __name__ == '__main__':
    main()
