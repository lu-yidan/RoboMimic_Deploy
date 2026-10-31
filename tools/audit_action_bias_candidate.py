"""Independent saved-trajectory and checkpoint checks; not a physics-step audit.

Native per-physics-step limits remain runtime evidence. This script separately
checks saved states, actor inputs/actions, source hashes and checkpoint reload.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco
import torch
from stable_baselines3 import PPO
from training.closure_composite_policy import ClosureCompositePolicy
from training.full_cycle_residual_env import FullCycleResidualEnv, ROOT
from training.bar_geometry_observation import ROWS, OBSERVATION_CONTRACT
from tools.audit_multirung_cycle import audit_endpoints


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_evaluation_sets(manifest, summary, results):
    """Reject partial, duplicate, or metadata-divergent evaluation suites."""
    if summary.get('evaluation_pending') or summary.get('after_evaluation_pending') or summary.get('stopped_by_request'):
        raise ValueError('Experiment evaluation is not complete')
    expected = set(manifest['evaluation_cases'])
    if not expected:
        raise ValueError('No evaluation cases declared')
    for label in ('after',):
        rows = results[label]
        names = [row['case'] for row in rows]
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError('Incomplete or duplicate evaluation cases: '+label)
        if rows != summary.get(label):
            raise ValueError('Evaluation metadata differs from summary: '+label)


def audit(out):
    torch.set_num_threads(1)
    manifest = json.loads((out/'manifest.json').read_text())
    if manifest['observation_contract'] != OBSERVATION_CONTRACT:
        raise ValueError('Expected geometry observation contract')
    summary = json.loads((out/'summary.json').read_text())
    results_by_label = {label: json.loads((out/f'{label}.json').read_text()) for label in ('after',)}
    validate_evaluation_sets(manifest, summary, results_by_label)
    env = FullCycleResidualEnv(plan=manifest['reference'], cases=manifest['evaluation_cases'])
    checks = dict(frozen_source_hashes=all(digest(out/'source_snapshot'/rel) == value for rel, value in manifest['sources'].items()),
                  current_environment_matches_frozen=all(digest(ROOT/rel) == value for rel, value in manifest['sources'].items()
                      if rel.replace('\\', '/').startswith('training/')),
                  reference_hash=digest(manifest['reference']) == manifest['reference_sha256'],
                  checkpoint_hash=digest(out/'policy.zip') == summary['checkpoint_sha256'],
                  scene_bundle_hash=digest(env.env.bundle/'manifest.json') == manifest['bundle_manifest_sha256'])
    model = ClosureCompositePolicy(PPO.load(out/'policy.zip', device='cpu'), manifest)
    original_predict=model.predict
    spec=manifest['closure_action_bias']
    assert manifest['policy_adapter']=='closure-action-bias-v1' and spec['geometry_gate']=='existing_high_rod'
    bias=np.asarray(spec['normalized_bias'],dtype=float)
    assert bias.shape==(26,) and np.isfinite(bias).all() and abs(bias).max()<=.2
    assert all(bias[i]==0 for i in range(26) if i not in (3,4))
    assert spec['start_seconds'] in (70.2,99.66) and spec['ramp_seconds']==3.
    assert manifest['horizon_seconds']==113.66
    def independent_prediction(observation,deterministic=True):
        value=np.asarray(observation)
        if value.ndim==2:
            return np.stack([independent_prediction(row,deterministic)[0] for row in value]),None
        action,state=original_predict(value,deterministic=deterministic)
        elapsed=113.66*(1.-float(value[105]))
        fraction=max(0.,min(1.,(elapsed-spec['start_seconds'])/3.))
        gain=max(0.,min(1.,(float(value[138])-.625)/.125))
        if not np.any(bias) or fraction==0. or gain==0.:return action,state
        blend=fraction*fraction*(3.-2.*fraction)*gain
        return np.minimum(1.,np.maximum(-1.,np.asarray(action,dtype=float)+blend*bias)).astype(np.float32),state
    model.predict=independent_prediction

    rows = []
    for label in ('after',):
        results = results_by_label[label]
        for result in results:
            path = out/f"{label}_{result['case']}.npz"
            with np.load(path) as z:
                q, v, ts, actions, observations, torques, rewards = (z[k] for k in
                    ('qpos', 'qvel', 'time_s', 'actions', 'observations', 'torques', 'rewards'))
                n = len(actions)
                env.reset(options={'case': result['case']})
                e = env.env
                finite = all(np.isfinite(x).all() for x in (q, v, ts, actions, observations, torques, rewards))
                dimensions = (q.shape == (n+1, e.model.nq) and v.shape == (n+1, e.model.nv)
                              and observations.shape == (n, 136 if label == 'zero' else 148) and torques.shape == (n, 29)
                              and actions.shape == (n, 26) and rewards.shape == (n,) and ts.shape == (n+1,))
                geometry_error = 0.
                if label == 'after':
                    expected_geometry = np.array([manifest['evaluation_cases'][result['case']].get(r,[0.,0.,0.]) for r in ROWS]).reshape(-1)/.02
                    if manifest['geometry_mode'] == 'blind': expected_geometry[:] = 0.
                    geometry_error = float(abs(observations[:,136:]-expected_geometry).max())
                margin = float(np.min(np.minimum(q[:, e.qadr]-e.model.jnt_range[e.joints, 0],
                                                e.model.jnt_range[e.joints, 1]-q[:, e.qadr])))
                replayed = (np.zeros_like(actions) if label == 'zero' else
                            np.asarray([model.predict(row, deterministic=True)[0] for row in observations]))
                action_difference = float(abs(actions-replayed).max())
                tracking_error = 0.
                velocity_error = float(abs(observations[:, 29:58]-v[:-1, e.vadr]).max())
                reward_error = 0.
                previous_action = np.zeros(26)
                for i, t in enumerate(ts[:-1]):
                    reference, _, _, _ = env.reference(float(t))
                    tracking_error = max(tracking_error, float(abs(observations[i, :29]-(q[i, e.qadr]-reference[e.qadr])).max()))
                    end_reference, _, _, _ = env.reference(float(ts[i+1]))
                    expected_reward = (.02-2*np.mean((q[i+1, e.qadr]-end_reference[e.qadr])**2)
                                       -.001*np.mean(actions[i].astype(float)**2)
                                       -.001*np.mean((actions[i].astype(float)-previous_action)**2))
                    if i == n-1:
                        expected_reward += 20. if result['success'] else -20.
                    reward_error = max(reward_error, abs(float(rewards[i])-expected_reward))
                    previous_action = actions[i].astype(float)
                timing = bool(abs(ts[0]) < 1e-12 and np.all(abs(np.diff(ts)[:-1]-.02) < 1e-8)
                              and -1e-10 <= ts[-1]-ts[-2] <= .02000001)
                row_checks = dict(file_hash=digest(path) == result['trajectory_sha256'], finite=bool(finite), dimensions=dimensions,
                    initial_state=bool(np.array_equal(q[0], e.data.qpos) and np.array_equal(v[0], e.data.qvel)),
                    action_bounds=bool(np.max(abs(actions)) <= 1.), action_replay=action_difference == 0.,
                    observation_geometry=geometry_error < 1e-7, observation_tracking=tracking_error < 1e-7, observation_velocity=velocity_error < 1e-6,
                    reward_reconstruction=reward_error < 1e-7, timing=timing,
                    sampled_torque_bounds=bool(np.all(torques <= e.limits[:, 1]+1e-8) and np.all(torques >= e.limits[:, 0]-1e-8)),
                    success_requires_complete_horizon=bool(not result['success'] or abs(ts[-1]-env.horizon) < 1e-7),
                    sampled_native_bounds_or_declared_failure=margin >= -1e-7 or result['reason'] == 'native_joint_position_violation')
                endpoints = audit_endpoints(e, dict(qpos=q, qvel=v, time_s=ts), env.plan)
                scratch = mujoco.MjData(e.model)
                penetration = 0.
                for pose in q:
                    scratch.qpos[:] = pose
                    mujoco.mj_kinematics(e.model, scratch)
                    mujoco.mj_collision(e.model, scratch)
                    for contact in scratch.contact:
                        a, b = (int(e.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
                        ina, inb = a in e.robot_bodies, b in e.robot_bodies
                        if not (ina or inb):
                            continue
                        expected = ina != inb and (a if ina else b) in e.contact_groups
                        if not expected:
                            penetration = max(penetration, float(-contact.dist))
                rows.append(dict(label=label, case=result['case'], experiment_success=result['success'],
                    trajectory_sha256=digest(path),
                    independently_accepted=bool(result['success'] and all(row_checks.values()) and all(x['passed'] for x in endpoints) and penetration < 1e-5),
                    sampled_unexpected_penetration_m=penetration,
                    independent_endpoints=endpoints,
                    passed=all(row_checks.values()), checks=row_checks, sampled_minimum_margin_rad=margin,
                    action_max_difference=action_difference, tracking_max_difference=tracking_error,
                    velocity_max_difference=velocity_error, reward_max_difference=reward_error))
    result = dict(passed=all(checks.values()) and all(r['passed'] for r in rows), checks=checks, trajectories=rows,
        experiment_manifest_sha256=digest(out/'manifest.json'), experiment_summary_sha256=digest(out/'summary.json'),
        checkpoint_sha256=digest(out/'policy.zip'), audit_source_sha256=digest(__file__),
        action_replay_mode='Single observation at a time, one Torch thread, matching composite prefix/suffix inference; exact equality required',
        scope='Saved states and independently reconstructed bounded knee/ankle action-bias policy. No PPO training; original task gates. Failed task trajectories remain failures.')
    (out/'independent_candidate_audit.json').write_text(json.dumps(result, indent=2))
    (out/'independent_candidate_audit_source.py').write_bytes(Path(__file__).read_bytes())
    env.close()
    print(json.dumps(result), flush=True)
    if not result['passed']:
        raise AssertionError('Independent training integrity checks failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    audit(parser.parse_args().output)
