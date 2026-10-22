"""Independent geometry and endpoint audit for a zero-residual saved episode."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from training.full_cycle_residual_env import FullCycleResidualEnv
from tools.audit_multirung_cycle import audit_endpoints


def audit(directory, case, offsets):
    trajectory = directory/f'zero_{case}.npz'
    records = json.loads((directory/'zero.json').read_text())
    runtime = next(row for row in records if row['case'] == case)
    env = FullCycleResidualEnv(cases={case: offsets})
    env.reset(options={'case': case})
    e = env.env
    with np.load(trajectory) as data:
        z = {key: data[key].copy() for key in data.files}
    endpoints = audit_endpoints(e, z, env.plan)
    penetration = 0.
    for q in z['qpos']:
        e.data.qpos[:] = q
        mujoco.mj_kinematics(e.model, e.data)
        mujoco.mj_collision(e.model, e.data)
        for contact in e.data.contact:
            a, b = (int(e.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            ina, inb = a in e.robot_bodies, b in e.robot_bodies
            if not (ina or inb):
                continue
            expected = ina != inb and (a if ina else b) in e.contact_groups
            if not expected:
                penetration = max(penetration, float(-contact.dist))
    q = z['qpos'][:, e.qadr]
    margin = float(np.minimum(q-e.model.jnt_range[e.joints, 0], e.model.jnt_range[e.joints, 1]-q).min())
    checksum = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    checks = dict(hash_match=checksum == runtime['trajectory_sha256'],
        finite=bool(all(np.isfinite(z[key]).all() for key in z)),
        zero_actions=bool(np.all(z['actions'] == 0.)),
        initial_state=bool(np.array_equal(z['qpos'][0], env.plan['qpos'][0]) and np.all(z['qvel'][0] == 0.)),
        recorded_torque_bounds=bool(np.all(z['torques'] <= e.limits[:, 1]+1e-8) and np.all(z['torques'] >= e.limits[:, 0]-1e-8)),
        recorded_native_bounds_or_explicit_failure=margin >= -1e-7 or runtime['reason'] == 'native_joint_position_violation')
    accepted = bool(runtime['success'] and all(x['passed'] for x in endpoints) and penetration < 1e-5 and all(checks.values()))
    result = dict(case=case, offsets=offsets, integrity_passed=all(checks.values()), checks=checks,
        runtime_success=runtime['success'], independently_accepted=accepted, endpoints=endpoints,
        unexpected_penetration_m=penetration, sampled_minimum_margin_rad=margin,
        trajectory_sha256=checksum, audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Saved 50 Hz geometry, endpoints and motor commands; 2000 Hz collision peaks remain runtime evidence')
    (directory/f'zero_{case}_independent.json').write_text(json.dumps(result, indent=2))
    env.close()
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--case', required=True)
    parser.add_argument('--offsets', default='{}')
    args = parser.parse_args()
    audit(args.directory, args.case, json.loads(args.offsets))
