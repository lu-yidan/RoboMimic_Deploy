"""Declared fixed closure residual, chosen only on training cases.

Same actuator/filter limits and strict environment as PPO. The intervention
depends on the reference phase, never on a case name or geometry offset.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
import mujoco
import numpy as np
from training.full_cycle_residual_env import ROOT
from training.strict_full_cycle_residual_env import FullCycleResidualEnv, CONTRACT
from tools.audit_multirung_cycle import audit_endpoints


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_action(phase, residual_rad, index):
    action = np.zeros(26, dtype=np.float32)
    if phase in ('posture_closure', 'closed_hold'):
        action[index] = residual_rad/.015
    return action


def run_case(payload):
    manifest, case, residual, output = payload[:4]
    joint_names = payload[4] if len(payload) > 4 else ['right_ankle_pitch_joint']
    output = Path(output)
    output.mkdir()
    env = FullCycleResidualEnv(plan=manifest['reference'], cases={case: manifest['training_cases'][case]},
        normal_bias_y_deg=manifest['normal_bias_y_deg'], memory_enabled=manifest['force_history_memory'])
    if sha(env.env.bundle/'manifest.json') != manifest['bundle_manifest_sha256']:
        raise ValueError('Scene bundle changed')
    names = [env.env.model.joint(env.env.joints[index]).name for index in env.indices]
    index = [names.index(name) for name in joint_names]
    obs, _ = env.reset(seed=20261018, options={'case': case})
    q, v, ts = [env.env.data.qpos.copy()], [env.env.data.qvel.copy()], [0.]
    actions, observations, torques, rewards = [], [], [], []
    started = time.perf_counter()
    while True:
        _, _, _, phase = env.reference(float(env.env.data.time))
        action = fixed_action(phase, residual, index)
        observations.append(obs.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        torques.append(env.last_torque.copy())
        q.append(env.env.data.qpos.copy())
        v.append(env.env.data.qvel.copy())
        ts.append(float(env.env.data.time))
        if terminated or truncated:
            break
    wall = time.perf_counter()-started
    arrays = dict(qpos=np.array(q), qvel=np.array(v), time_s=np.array(ts), actions=np.array(actions),
                  observations=np.array(observations), torques=np.array(torques), rewards=np.array(rewards))
    np.savez_compressed(output/'trajectory.npz', **arrays)
    endpoint = audit_endpoints(env.env, arrays, env.plan)
    scratch = mujoco.MjData(env.env.model)
    penetration = 0.
    for pose in q:
        scratch.qpos[:] = pose
        mujoco.mj_kinematics(env.env.model, scratch)
        mujoco.mj_collision(env.env.model, scratch)
        for contact in scratch.contact:
            a, b = (int(env.env.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
            ina, inb = a in env.env.robot_bodies, b in env.env.robot_bodies
            if not (ina or inb):
                continue
            expected = ina != inb and (a if ina else b) in env.env.contact_groups
            if not expected:
                penetration = max(penetration, float(-contact.dist))
    replay = np.array([fixed_action(env.reference(t)[3], residual, index) for t in ts[:-1]])
    checks = dict(finite=all(np.isfinite(value).all() for value in arrays.values()),
                  action_replay=np.array_equal(replay, arrays['actions']),
                  action_bounds=np.max(abs(arrays['actions'])) <= 1.,
                  torque_bounds=np.all(arrays['torques'] <= env.env.limits[:, 1]+1e-8)
                      and np.all(arrays['torques'] >= env.env.limits[:, 0]-1e-8),
                  no_unexpected_sampled_penetration=penetration < 1e-5)
    checks = {key: bool(value) for key, value in checks.items()}
    result = dict(**info, residual_rad=residual, joint_names=joint_names, wall_seconds=wall, return_sum=float(sum(rewards)),
                  endpoint_checks=endpoint, checks=checks,
                  independently_accepted=bool(info['success'] and all(checks.values()) and all(row['passed'] for row in endpoint)),
                  sampled_unexpected_penetration_m=penetration, trajectory_sha256=sha(output/'trajectory.npz'))
    (output/'result.json').write_text(json.dumps(result, indent=2))
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3, choices=(1, 2, 3))
    parser.add_argument('--joints', nargs='+', default=['right_ankle_pitch_joint'])
    parser.add_argument('--positive-only', action='store_true')
    parser.add_argument('--selection-protocol', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.experiment/'manifest.json').read_text())
    if manifest['contract'] != CONTRACT or sha(manifest['reference']) != manifest['reference_sha256']:
        raise ValueError('Matching strict contract and reference required')
    for relative, expected in manifest['sources'].items():
        if relative.replace('\\', '/').startswith('training/') and sha(ROOT/relative) != expected:
            raise ValueError('Frozen environment changed: '+relative)
    args.output.mkdir(parents=True, exist_ok=False)
    for relative, expected in manifest['sources'].items():
        source = args.experiment/'source_snapshot'/relative
        if sha(source) != expected:
            raise ValueError('Source snapshot changed')
        destination = args.output/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copy2(__file__, args.output/'baseline_source.py')
    candidates = [.003] if args.positive_only else [-.003, .003]
    protocol = dict(parent_manifest_sha256=sha(args.experiment/'manifest.json'), manifest=manifest,
        candidates_rad=candidates, joints=args.joints, phases=['posture_closure', 'closed_hold'],
        source_sha256=sha(__file__), selection_cases=list(manifest['training_cases']),
        scope='Deterministic fixed baseline candidate search, not reinforcement learning; training cases only. No deployment or generalization claim.')
    if args.selection_protocol:
        selection = json.loads(args.selection_protocol.read_text())
        if (selection['candidates_rad'] != candidates or selection['selection_cases'] != list(manifest['training_cases'])
                or selection['joints'] != args.joints):
            raise ValueError('Selection protocol does not match the declared candidates')
        shutil.copy2(args.selection_protocol, args.output/'selection_protocol.json')
        protocol['selection_protocol_sha256'] = sha(args.output/'selection_protocol.json')
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    payloads = [(manifest, case, residual, str((args.output/f"{'minus' if residual < 0 else 'plus'}3mrad_{case}").resolve()), args.joints)
                for residual in candidates for case in manifest['training_cases']]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_case, payload) for payload in payloads]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            (args.output/'results.json').write_text(json.dumps(results, indent=2))
            print(json.dumps({key: row[key] for key in ('case', 'residual_rad', 'independently_accepted', 'reason')}), flush=True)
    (args.output/'summary.json').write_text(json.dumps(dict(completed_cases=len(results),
        candidates=[dict(residual_rad=residual, accepted=sum(row['independently_accepted'] for row in results if row['residual_rad'] == residual),
                         total=sum(row['residual_rad'] == residual for row in results)) for residual in candidates],
        results=results, protocol_sha256=sha(args.output/'protocol.json')), indent=2))


if __name__ == '__main__':
    main()
