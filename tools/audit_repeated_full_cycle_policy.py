"""Independently verify saved repeated-cycle policy inputs, actions and geometry."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from deploy_mujoco.run_trained_full_cycle import load_verified, sha
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.audit_multirung_cycle import audit_endpoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    provenance = json.loads((args.run/'provenance.json').read_text())
    summary = json.loads((args.run/'summary.json').read_text())
    parent = Path(provenance['experiment'])
    original, policy, manifest, parent_summary = load_verified(parent, 'nominal')
    original.close()
    env = FullCycleResidualEnv(plan=provenance['plan'], cases={'nominal': {}},
        normal_bias_y_deg=manifest['normal_bias_y_deg'], memory_enabled=manifest['force_history_memory'])
    env.reset(options={'case': 'nominal'})
    e = env.env
    with np.load(args.run/'trajectory.npz') as saved:
        z = {key: saved[key].copy() for key in saved.files}
    n = len(z['actions'])
    q, v, ts, obs, raw = (z[key] for key in ('qpos', 'qvel', 'time_s', 'observations', 'raw_observations'))
    checks = dict(trajectory_hash=sha(args.run/'trajectory.npz') == summary['trajectory_sha256'],
        checkpoint_hash=sha(parent/'policy.zip') == provenance['checkpoint_sha256'],
        parent_audit_hash=sha(parent/'independent_training_audit.json') == provenance['parent_audit_sha256'],
        reference_hash=sha(provenance['plan']) == provenance['plan_sha256'],
        runner_hash=sha(args.run/'runner_source.py') == provenance['runner_sha256'],
        finite=all(np.isfinite(value).all() for value in z.values()),
        dimensions=(q.shape == (n+1, e.model.nq) and v.shape == (n+1, e.model.nv)
            and obs.shape == raw.shape == (n, 136) and z['actions'].shape == (n, 26)
            and z['torques'].shape == (n, 29) and z['rewards'].shape == (n,) and ts.shape == (n+1,)),
        initial_state=np.array_equal(q[0], e.data.qpos) and np.array_equal(v[0], e.data.qvel),
        unchanged_other_observations=np.array_equal(np.delete(obs, 105, axis=1), np.delete(raw, 105, axis=1)),
        raw_remaining_time=np.max(abs(raw[:, 105]-np.maximum(0., 1.-ts[:-1]/env.horizon))) < 1e-7,
        timing=abs(ts[0]) < 1e-12 and np.all(abs(np.diff(ts)[:-1]-.02) < 1e-8),
        action_bounds=np.max(abs(z['actions'])) <= 1.,
        torque_bounds=np.all(z['torques'] <= e.limits[:, 1]+1e-8) and np.all(z['torques'] >= e.limits[:, 0]-1e-8))
    expected_remaining = []
    tracking_error, reward_error = 0., 0.
    previous_action = np.zeros(26)
    for i, t in enumerate(ts[:-1]):
        # Independent interval calculation; no call to the input adapter under test.
        index = min(sum(float(t) >= end for end in env.ends), len(env.ends)-1)
        start = 0. if index == 0 else env.ends[index-1]
        expected_remaining.append(max(0., 1.-(t-start)/(env.ends[index]-start)))
        reference = env.reference(float(t))[0]
        tracking_error = max(tracking_error, float(abs(raw[i, :29]-(q[i, e.qadr]-reference[e.qadr])).max()))
        target = env.reference(float(ts[i+1]))[0]
        action = z['actions'][i].astype(float)
        reward = .02-2*np.mean((q[i+1, e.qadr]-target[e.qadr])**2)-.001*np.mean(action**2)-.001*np.mean((action-previous_action)**2)
        if i == n-1:
            reward += 20. if summary['success'] else -20.
        reward_error = max(reward_error, abs(float(z['rewards'][i])-reward))
        previous_action = action
    zero_policy = provenance.get('policy_mode', 'learned') == 'zero'
    replay = np.zeros_like(z['actions']) if zero_policy else np.asarray([policy.predict(row, deterministic=True)[0] for row in obs])
    margin = float(np.minimum(q[:, e.qadr]-e.model.jnt_range[e.joints, 0], e.model.jnt_range[e.joints, 1]-q[:, e.qadr]).min())
    checks.update(cycle_local_time=np.max(abs(obs[:, 105]-expected_remaining)) < 1e-7,
        action_replay=np.array_equal(replay, z['actions']), tracking_observation=tracking_error < 1e-7,
        velocity_observation=np.max(abs(raw[:, 29:58]-v[:-1, e.vadr])) < 1e-6,
        reward_reconstruction=reward_error < 1e-7,
        native_bounds_or_declared_failure=margin >= -1e-7 or summary['reason'] == 'native_joint_position_violation',
        successful_horizon=not summary['success'] or abs(ts[-1]-env.horizon) < 1e-7)
    parent_label = 'zero' if zero_policy else 'after'
    original_path = parent/f'{parent_label}_nominal.npz'
    original_record = next(row for row in parent_summary[parent_label] if row['case'] == 'nominal')
    checks['original_trajectory_hash'] = sha(original_path) == original_record['trajectory_sha256']
    with np.load(original_path) as original:
        first_n = len(original['actions'])
        prefix = {key: bool(np.array_equal(z[key][:first_n+(key in ('qpos', 'qvel', 'time_s'))], original[key]))
                  for key in ('qpos', 'qvel', 'time_s', 'observations', 'actions', 'torques')}
    checks['first_cycle_exact'] = all(prefix.values())
    endpoints = audit_endpoints(e, z, env.plan)
    data = mujoco.MjData(e.model)
    penetration = 0.
    for pose in q:
        data.qpos[:] = pose
        mujoco.mj_kinematics(e.model, data)
        mujoco.mj_collision(e.model, data)
        for contact in data.contact:
            a, b = (int(e.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            ina, inb = a in e.robot_bodies, b in e.robot_bodies
            if (ina or inb) and not (ina != inb and (a if ina else b) in e.contact_groups):
                penetration = max(penetration, float(-contact.dist))
    checks = {key: bool(value) for key, value in checks.items()}
    result = dict(integrity_passed=all(checks.values()), checks=checks, prefix_checks=prefix,
        independently_accepted=bool(all(checks.values()) and summary['success'] and all(row['passed'] for row in endpoints) and penetration < 1e-5),
        endpoints=endpoints, sampled_unexpected_penetration_m=penetration, sampled_native_margin_rad=margin,
        action_max_difference=float(abs(replay-z['actions']).max()), reward_max_difference=reward_error,
        summary_sha256=sha(args.run/'summary.json'), provenance_sha256=sha(args.run/'provenance.json'),
        trajectory_sha256=sha(args.run/'trajectory.npz'), source_sha256=sha(__file__),
        scope='Offline saved-state checks at 50 Hz. Runtime 2000 Hz force/limit checks remain separate evidence; not hardware validation.')
    (args.run/'independent_policy_audit.json').write_text(json.dumps(result, indent=2))
    (args.run/'independent_policy_audit_source.py').write_bytes(Path(__file__).read_bytes())
    env.close()
    print(json.dumps(result), flush=True)
    if not result['integrity_passed']:
        raise AssertionError('Repeated policy integrity audit failed')


if __name__ == '__main__':
    main()
