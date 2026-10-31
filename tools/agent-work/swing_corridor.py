"""2-D feasibility sweep of the swing-foot corridor.

Offsets the left foot by (dx away from the wall, dz up) inside the swing window using damped
least squares, and reports the minimum foot-to-rung clearance for each combination. Answers which
direction (and how far) the reference can be opened up without losing contact feasibility.
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
CASE = 'training_13'
WINDOW = (75.0, 89.0)          # swing only: after lift-off, before touch-down
DX = [-4., 0., 4., 8.]          # away from the wall
DZ = [0., 4., 8.]

manifest = json.loads((BASE / 'manifest.json').read_text())
plan_path = Path(manifest['reference'])
with np.load(plan_path) as data:
    times = data['time'].copy()
    qpos = data['qpos'].copy()

env = StrictEnv(plan=str(plan_path), cases={CASE: manifest['evaluation_cases'][CASE]},
                normal_bias_y_deg=manifest['normal_bias_y_deg'],
                memory_enabled=manifest['force_history_memory'])
physical = env.env
model = physical.model
qadr, vadr = np.asarray(physical.qadr), np.asarray(physical.vadr)
joint_ids = np.asarray(physical.joints)
names = [model.joint(int(j)).name for j in joint_ids]
leg = [i for i, n in enumerate(names) if n.startswith('left_')
       and any(k in n for k in ('hip', 'knee', 'ankle'))]
leg_qadr, leg_vadr = qadr[leg], vadr[leg]
lo, hi = model.jnt_range[joint_ids, 0], model.jnt_range[joint_ids, 1]
leg_lo, leg_hi = lo[leg], hi[leg]

foot = model.body('left_ankle_roll_link').id
geom_name = [model.geom(g).name or '' for g in range(model.ngeom)]
foot_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == foot
              and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]
rungs = {}
for g in range(model.ngeom):
    n = geom_name[g]
    if n.startswith('rung_') and 'arm' not in n and 'brace' not in n:
        rungs[g] = n

scratch = mujoco.MjData(model)
window = np.flatnonzero((times >= WINDOW[0]) & (times <= WINDOW[1]))


def per_rung_clearance(state):
    scratch.qpos[:] = state
    mujoco.mj_forward(model, scratch)
    return {n: min(mujoco.mj_geomDistance(model, scratch, g1, g2, 0.25, None) for g1 in foot_geoms)
            for g2, n in rungs.items()}


def foot_pos(state):
    scratch.qpos[:] = state
    mujoco.mj_forward(model, scratch)
    return scratch.xpos[foot].copy()


def offset(state, dx_mm, dz_mm):
    q = state.copy()
    start = foot_pos(state)
    target = start + np.array([dx_mm / 1000., 0., dz_mm / 1000.])
    for _ in range(6):
        error = target - foot_pos(q)
        if np.linalg.norm(error) < 5e-5:
            break
        scratch.qpos[:] = q
        mujoco.mj_forward(model, scratch)
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, scratch, jacp, None, foot)
        jac = jacp[:, leg_vadr]
        dq = jac.T @ np.linalg.inv(jac @ jac.T + 1e-6 * np.eye(3)) @ error
        q[leg_qadr] = np.clip(q[leg_qadr] + dq.ravel(), leg_lo, leg_hi)
    return q


baseline = None
print('swing window %.1f-%.1f s, %d nodes' % (WINDOW[0], WINDOW[1], len(window)))
print('\n%-8s %-8s %10s %10s %14s' % ('dx mm', 'dz mm', 'min clr', 'rung', 'limit hits'))
summary = []
for dx in DX:
    for dz in DZ:
        worst, worst_rung, hits = 1e9, '', 0
        for i in window:
            state = qpos[i]
            q = offset(state, dx, dz)
            clearance = per_rung_clearance(q)
            name, value = min(clearance.items(), key=lambda kv: kv[1])
            if value < worst:
                worst, worst_rung = value, name
            if np.any(q[leg_qadr] >= leg_hi - 1e-6) or np.any(q[leg_qadr] <= leg_lo + 1e-6):
                hits += 1
        summary.append(dict(dx=dx, dz=dz, min_clearance_mm=1000 * worst, rung=worst_rung, limit_hits=hits))
        print('%-8.0f %-8.0f %10.2f %10s %14d' % (dx, dz, 1000 * worst, worst_rung, hits))

Path(r'D:\G1Climbing\.dsh-watch\swing_corridor.json').write_text(
    json.dumps(dict(reference=str(plan_path), window_s=WINDOW, grid=summary), indent=2))
print('\nwrote .dsh-watch/swing_corridor.json')
