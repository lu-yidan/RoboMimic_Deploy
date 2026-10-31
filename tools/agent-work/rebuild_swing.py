"""Open the swing corridor in place, with a sine envelope that preserves both contacts and the endpoint.

  * the swing is the airborne gap that contains 77.5 s (the moment the rollouts die);
  * the offset is zero at lift-off and touch-down, maximal mid-swing;
  * joints are kept at the original 18 mrad native margin;
  * the result is audited with the project's own plan auditor.
"""
import json
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(r'D:\G1Climbing\external\RoboMimic_Deploy')
sys.path.insert(0, str(REPO))

from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv  # noqa: E402
from tools.plan_clearance_cycle import audit_plan  # noqa: E402
from tools.continue_vector_full_cycle_training import sha  # noqa: E402

BASE = REPO / 'outputs/calibrated_geometry_training_baseline_20261027_v1'
OUT = Path(r'D:\G1Climbing\.dsh-watch\rebuild/plan_swing_open_v4')
CASE = 'training_13'
DX, DZ = 0.0, 16.0
MARGIN = 0.018

manifest = json.loads((BASE / 'manifest.json').read_text())
plan_path = Path(manifest['reference'])
with np.load(plan_path) as data:
    plan = {k: data[k].copy() for k in data.files}

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
leg_lo, leg_hi = lo[leg] + MARGIN, hi[leg] - MARGIN
foot = model.body('left_ankle_roll_link').id
geom_name = [model.geom(g).name or '' for g in range(model.ngeom)]
foot_geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == foot
              and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]
rung_geoms = [g for g in range(model.ngeom) if geom_name[g].startswith('rung_')
              and 'arm' not in geom_name[g] and 'brace' not in geom_name[g]]
scratch = mujoco.MjData(model)
times = plan['time']


def clearance(state):
    scratch.qpos[:] = state
    mujoco.mj_forward(model, scratch)
    mujoco.mj_collision(model, scratch)
    return min(mujoco.mj_geomDistance(model, scratch, g1, g2, 0.20, None)
               for g1 in foot_geoms for g2 in rung_geoms)


def foot_pos(state):
    scratch.qpos[:] = state
    mujoco.mj_forward(model, scratch)
    return scratch.xpos[foot].copy()


series = np.array([clearance(plan['qpos'][i]) for i in range(len(times))])
contact = series < 0.0005
target_index = int(np.argmin(abs(times - 77.5)))
if contact[target_index]:
    raise SystemExit('77.5 s is not airborne for the left foot')
left = target_index
while left > 0 and not contact[left - 1]:
    left -= 1
right = target_index
while right < len(times) - 1 and not contact[right + 1]:
    right += 1
t_lift, t_touch = float(times[left]), float(times[right])
print('swing gap: %.2f - %.2f s (%d nodes)' % (t_lift, t_touch, right - left + 1))


def envelope(t):
    if t <= t_lift or t >= t_touch:
        return 0.
    return float(np.sin(np.pi * (t - t_lift) / (t_touch - t_lift)))


def solve(state, dx, dz):
    q = state.copy()
    start = foot_pos(state)
    target = start + np.array([dx / 1000., 0., dz / 1000.])
    for _ in range(10):
        error = target - foot_pos(q)
        if np.linalg.norm(error) < 2e-5:
            break
        scratch.qpos[:] = q
        mujoco.mj_forward(model, scratch)
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, scratch, jacp, None, foot)
        jac = jacp[:, leg_vadr]
        dq = jac.T @ np.linalg.inv(jac @ jac.T + 1e-6 * np.eye(3)) @ error
        q[leg_qadr] = np.clip(q[leg_qadr] + dq.ravel(), leg_lo, leg_hi)
    return q, float(np.linalg.norm(target - foot_pos(q)))


new = {k: v.copy() for k, v in plan.items()}
mid = float(times[target_index])
probe_before = float(clearance(plan['qpos'][target_index]))
residuals = []
for i in range(left, right + 1):
    weight = envelope(float(times[i]))
    if weight <= 1e-9:
        continue
    q, residual = solve(plan['qpos'][i], DX * weight, DZ * weight)
    residuals.append(residual)
    new['qpos'][i] = q
    new['errors'][i] = max(float(new['errors'][i]), residual)

probe_after = float(clearance(new['qpos'][target_index]))
span = series[left:right + 1]
span_new = np.array([clearance(new['qpos'][i]) for i in range(left, right + 1)])
print('clearance at %.2f s: %.2f -> %.2f mm' % (mid, 1000 * probe_before, 1000 * probe_after))
print('clearance over the swing: min %.2f -> %.2f mm (excluding the last 3 nodes before touch-down)'
      % (1000 * span.min(), 1000 * span_new[:-3].min()))
print('max IK residual: %.1e m' % max(residuals))

audit = audit_plan(physical, new, dt=.005)
print('\nplan audit:')
for key in ('passed', 'maximum_penetration_m', 'joint_margin_rad',
            'peak_reference_joint_speed_rad_s', 'ik_residual'):
    print('  %-34s %s' % (key, audit[key]))
print('  failures:', audit['failures'][:3])

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
np.savez_compressed(OUT / 'plan.npz', **new)
(OUT / 'audit.json').write_text(json.dumps(audit, indent=2))
(OUT / 'manifest.json').write_text(json.dumps(dict(
    parent=str(plan_path), parent_sha256=sha(plan_path), plan_sha256=sha(OUT / 'plan.npz'),
    source_sha256=sha(__file__), swing_gap_s=[t_lift, t_touch], offset_mm=dict(away=DX, up=DZ),
    clearance_at_77p5_before_mm=1000 * probe_before, clearance_at_77p5_after_mm=1000 * probe_after,
    scope='Diagnostic candidate: mid-swing foot offset with a sine envelope; contacts and endpoint '
          'unchanged. Not an audited production reference.'), indent=2))
print('\nwrote', OUT)
