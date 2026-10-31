"""Sweep the vertical swing lift: clearance at the critical instant vs plan-audit margins."""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(r'D:\G1Climbing\external\RoboMimic_Deploy')
sys.path.insert(0, str(REPO))

from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv  # noqa: E402
from tools.plan_clearance_cycle import audit_plan  # noqa: E402

BASE = REPO / 'outputs/calibrated_geometry_training_baseline_20261027_v1'
CASE = 'training_13'
LIFTS = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]
MARGIN = 0.018

manifest = json.loads((BASE / 'manifest.json').read_text())
plan_path = Path(manifest['reference'])
with np.load(plan_path) as data:
    plan = {k: data[k].copy() for k in data.files}

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
lo = model.jnt_range[joint_ids, 0][leg] + MARGIN
hi = model.jnt_range[joint_ids, 1][leg] - MARGIN
foot = model.body('left_ankle_roll_link').id
gname = [model.geom(g).name or '' for g in range(model.ngeom)]
foot_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == foot
              and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]
rung_geoms = [g for g in range(model.ngeom) if gname[g].startswith('rung_')
              and 'arm' not in gname[g] and 'brace' not in gname[g]]
scratch = mujoco.MjData(model)
times = plan['time']


def clearance(state):
    scratch.qpos[:] = state
    mujoco.mj_forward(model, scratch)
    mujoco.mj_collision(model, scratch)
    return min(mujoco.mj_geomDistance(model, scratch, g1, g2, 0.30, None)
               for g1 in foot_geoms for g2 in rung_geoms)


series = np.array([clearance(plan['qpos'][i]) for i in range(len(times))])
contact = series < 0.0005
target = int(np.argmin(abs(times - 77.5)))
left, right = target, target
while left > 0 and not contact[left - 1]:
    left -= 1
while right < len(times) - 1 and not contact[right + 1]:
    right += 1
t_lift, t_touch = float(times[left]), float(times[right])
print('case %s  swing %.2f-%.2f s  base clearance at 77.5 s: %.2f mm'
      % (CASE, t_lift, t_touch, 1000 * series[target]))

print('\n%-8s %12s %12s %12s %12s' % ('lift mm', 'clr@77.5', 'swing min', 'joint margin', 'audit'))
for dz in LIFTS:
    new = {k: v.copy() for k, v in plan.items()}
    if dz > 0:
        for i in range(left, right + 1):
            weight = float(np.sin(np.pi * (times[i] - t_lift) / (t_touch - t_lift)))
            if weight <= 1e-9:
                continue
            q = plan['qpos'][i].copy()
            scratch.qpos[:] = q
            mujoco.mj_forward(model, scratch)
            pos = scratch.xpos[foot].copy()
            tgt = pos + np.array([0., 0., dz * weight / 1000.])
            for _ in range(8):
                err = tgt - scratch.xpos[foot]
                if np.linalg.norm(err) < 2e-5:
                    break
                jacp = np.zeros((3, model.nv))
                mujoco.mj_jacBody(model, scratch, jacp, None, foot)
                jac = jacp[:, leg_vadr]
                q[leg_qadr] = np.clip(q[leg_qadr] + jac.T @ np.linalg.inv(jac @ jac.T + 1e-6 * np.eye(3)) @ err,
                                      lo, hi)
                scratch.qpos[:] = q
                mujoco.mj_forward(model, scratch)
            new['qpos'][i] = q
    audit = audit_plan(physical, new, dt=.005)
    cl = clearance(new['qpos'][target])
    span = [clearance(new['qpos'][i]) for i in range(left, right + 1)]
    print('%-8.0f %12.2f %12.2f %12.4f %12s' % (
        dz, 1000 * cl, 1000 * min(span), audit['joint_margin_rad'], audit['passed']))
