"""Build a short clip from the closing segment of our climbing reference.

Why: the 113.66 s clip is 20x longer than the reproduction's median training clip (5.8 s), and the policy
never survives the imagination horizon (surv = 0 at 3.7k iterations). The closing phase (70.2-113.66 s) is
both short enough to learn and exactly the segment the endpoint-closure metric cares about.

Same npz layout as convert_plan_to_clip.py (the 6 keys build_library.py expects + body_names).
"""
import json
from pathlib import Path

import numpy as np

PLAN = Path(r'D:\G1Climbing\external\RoboMimic_Deploy\outputs\swing_lift_phase_sine_plan_20261029_v2\plan.npz')
OUT_DIR = Path(r'D:\G1Climbing\supermimic\data\npz\climb_close')
CLIP = 'climb_close_v1'
T0 = 70.2
FPS = 50.0


def main():
    with np.load(PLAN) as data:
        time = data['time'].astype(np.float64)
        qpos = data['qpos'].astype(np.float64)
    keep = time >= T0
    time, qpos = time[keep], qpos[keep]
    time = time - time[0]
    dt = 1.0 / FPS
    dense = np.arange(time[0], time[-1], dt)
    resampled = np.empty((dense.size, qpos.shape[1]))
    for k in range(qpos.shape[1]):
        resampled[:, k] = np.interp(dense, time, qpos[:, k])
    qpos, time = resampled, dense

    pos, quat, joints = qpos[:, 0:3], qpos[:, 3:7], qpos[:, 7:36]
    lin_vel = np.gradient(pos, dt, axis=0)
    joint_vel = np.gradient(joints, dt, axis=0)

    def quat_mul(a, b):
        w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                         w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)

    conjugate = quat * np.array([1.0, -1.0, -1.0, -1.0])
    delta = quat_mul(quat[1:], conjugate[:-1])
    sign = np.where(delta[:, 0:1] < 0, -1.0, 1.0)
    ang_vel = 2.0 * delta[:, 1:4] * sign / dt
    ang_vel = np.vstack([ang_vel, ang_vel[-1:]])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / (CLIP + '.npz')
    np.savez_compressed(target,
                        body_pos_w=pos[:, None, :], body_quat_w=quat[:, None, :],
                        joint_pos=joints, body_lin_vel_w=lin_vel[:, None, :],
                        body_ang_vel_w=ang_vel[:, None, :], joint_vel=joint_vel,
                        body_names=np.array(['pelvis']))
    meta = dict(clip=CLIP, frames=int(qpos.shape[0]), fps=FPS, duration_s=float(time[-1] + dt),
                t0=T0, joints=int(joints.shape[1]))
    (OUT_DIR / (CLIP + '.json')).write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
