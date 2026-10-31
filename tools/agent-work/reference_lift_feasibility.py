"""Feasibility check: can the swing foot simply be lifted in the reference?

Lifts the left foot by dz inside the swing window with a damped least-squares joint correction,
then reports per level: achieved lift, max joint change, limit hits, and foot-to-rung clearance
before/after (collision geoms only). Nothing is written outside .dsh-watch.
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
WINDOW = (68.0, 84.0)
LIFTS = [4.0, 8.0, 12.0]

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
qadr = np.asarray(physical.qadr)
vadr = np.asarray(physical.vadr)
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
rung_geoms = [g for g in range(model.ngeom) if geom_name[g].startswith('rung_')
              and 'arm' not in geom_name[g] and 'brace' not in geom_name[g]]

scratch = mujoco.MjData(model)
window = np.flatnonzero((times >= WINDOW[0]) & (times <= WINDOW[1]))
print('reference:', plan_path)
print('window nodes:', len(window), ' leg joints:', [names[i] for i in leg])


def clearance(state):
    scratch.qpos[:] = state
    mujoco.mj_forward(model, scratch)
    mujoco.mj_collision(model, scratch)
    return min(mujoco.mj_geomDistance(model, scratch, g1, g2, 0.20, None)
               for g1 in foot_geoms for g2 in rung_geoms)


def foot_z(state):
    scratch.qpos[:] = state
    mujoco.mj_forward(model, scratch)
    return float(scratch.xpos[foot][2])


def lift(state, dz_m):
    q = state.copy()
    z0 = foot_z(state)
    for _ in range(4):
        error = dz_m - (foot_z(q) - z0)
        if abs(error) < 5e-5:
            break
        scratch.qpos[:] = q
        mujoco.mj_forward(model, scratch)
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, scratch, jacp, None, foot)
        row = jacp[2, leg_vadr][None, :]
        dq = row.T @ np.linalg.inv(row @ row.T + 1e-6) * error
        q[leg_qadr] = np.clip(q[leg_qadr] + dq.ravel(), leg_lo, leg_hi)
    return q


print('\n%-8s %12s %12s %12s %10s %8s' % ('lift mm', 'clr before', 'clr after', 'achieved mm', 'max dq deg', 'ok'))
summary = {}
for dz in LIFTS:
    before, after, achieved, dqmax, hits = [], [], [], [], 0
    for i in window:
        state = qpos[i]
        q = lift(state, dz / 1000.)
        before.append(clearance(state))
        after.append(clearance(q))
        achieved.append(1000 * (foot_z(q) - foot_z(state)))
        dqmax.append(np.degrees(np.max(np.abs(q[leg_qadr] - state[leg_qadr]))))
        if np.any(q[leg_qadr] >= leg_hi - 1e-6) or np.any(q[leg_qadr] <= leg_lo + 1e-6):
            hits += 1
    ok = min(after) > 0 and hits == 0
    summary[dz] = dict(clearance_before_mm=1000 * min(before), clearance_after_mm=1000 * min(after),
                       achieved_min_mm=min(achieved), max_dq_deg=max(dqmax), limit_hits=hits,
                       feasible=bool(ok))
    print('%-8.0f %12.3f %12.3f %12.3f %10.3f %8s' % (
        dz, 1000 * min(before), 1000 * min(after), min(achieved), max(dqmax), ok))

Path(r'D:\G1Climbing\.dsh-watch\reference_lift_feasibility.json').write_text(
    json.dumps(dict(reference=str(plan_path), window_s=WINDOW, case=CASE, results=summary,
                    leg_joints=[names[i] for i in leg]), indent=2))
print('\nwrote .dsh-watch/reference_lift_feasibility.json')
