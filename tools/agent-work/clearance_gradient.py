"""Which direction opens the swing clearance? Finite-difference probe of d(clearance)/d(foot offset)."""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(r'D:\G1Climbing\external\RoboMimic_Deploy')
sys.path.insert(0, str(REPO))

from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv  # noqa: E402

BASE = REPO / 'outputs/calibrated_geometry_training_baseline_20261027_v1'
CASE = 'training_13'
PROBE_TIMES = [76.0, 77.5, 79.0, 80.5]
STEP_MM = 2.0

manifest = json.loads((BASE / 'manifest.json').read_text())
plan_path = Path(manifest['reference'])
with np.load(plan_path) as data:
    times, qpos = data['time'].copy(), data['qpos'].copy()
env = StrictEnv(plan=str(plan_path), cases={CASE: manifest['evaluation_cases'][CASE]},
                normal_bias_y_deg=manifest['normal_bias_y_deg'],
                memory_enabled=manifest['force_history_memory'])
env.reset(options={'case': CASE})
physical = env.env
model = physical.model
qadr, vadr = np.asarray(physical.qadr), np.asarray(physical.vadr)
joint_ids = np.asarray(physical.joints)
names = [model.joint(int(j)).name for j in joint_ids]
leg = [i for i, n in enumerate(names) if n.startswith('left_')
       and any(k in n for k in ('hip', 'knee', 'ankle'))]
leg_qadr, leg_vadr = qadr[leg], vadr[leg]
lo = model.jnt_range[joint_ids, 0][leg] + 1e-3
hi = model.jnt_range[joint_ids, 1][leg] - 1e-3
foot = model.body('left_ankle_roll_link').id
gname = [model.geom(g).name or '' for g in range(model.ngeom)]
foot_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == foot
              and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]
rung_geoms = [g for g in range(model.ngeom) if gname[g].startswith('rung_')
              and 'arm' not in gname[g] and 'brace' not in gname[g]]
scratch = mujoco.MjData(model)


def setup(state):
    scratch.qpos[:] = state
    mujoco.mj_forward(model, scratch)
    mujoco.mj_collision(model, scratch)


def clearance():
    return min(mujoco.mj_geomDistance(model, scratch, g1, g2, 0.30, None)
               for g1 in foot_geoms for g2 in rung_geoms)


def nearest_rung():
    best = min(((min(mujoco.mj_geomDistance(model, scratch, g1, g2, 0.30, None) for g1 in foot_geoms), n)
                for g2, n in ((g, gname[g]) for g in rung_geoms)))
    return best


def displaced(state, delta):
    q = state.copy()
    setup(state)
    target = scratch.xpos[foot].copy() + delta
    for _ in range(8):
        error = target - scratch.xpos[foot]
        if np.linalg.norm(error) < 2e-5:
            break
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, scratch, jacp, None, foot)
        jac = jacp[:, leg_vadr]
        dq = jac.T @ np.linalg.inv(jac @ jac.T + 1e-6 * np.eye(3)) @ error
        q[leg_qadr] = np.clip(q[leg_qadr] + dq.ravel(), lo, hi)
        setup(q)
    return q


print('%-8s %10s %10s %10s %10s   %s' % ('t', 'base mm', 'dx+ mm', 'dy+ mm', 'dz+ mm', 'nearest rung'))
for probe in PROBE_TIMES:
    i = int(np.argmin(abs(times - probe)))
    state = qpos[i]
    setup(state)
    base = clearance()
    nearest = nearest_rung()[1]
    deltas = {}
    for axis, name in ((0, 'dx'), (1, 'dy'), (2, 'dz')):
        delta = np.zeros(3)
        delta[axis] = STEP_MM / 1000.
        q = displaced(state, delta)
        setup(q)
        deltas[name] = clearance()
    print('%-8.1f %10.2f %10.2f %10.2f %10.2f   %s' % (
        probe, 1000 * base, 1000 * deltas['dx'], 1000 * deltas['dy'], 1000 * deltas['dz'], nearest))
