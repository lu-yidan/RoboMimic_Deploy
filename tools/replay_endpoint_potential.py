"""Check a prospective reward potential on recorded trajectories; no training."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from training.full_cycle_residual_env import FullCycleResidualEnv, ROOT
from training.endpoint_potential import endpoint_potential
from deploy_mujoco.run_trained_full_cycle import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.experiment/'manifest.json').read_text())
    audit = json.loads((args.experiment/'independent_training_audit.json').read_text())
    summary = json.loads((args.experiment/'summary.json').read_text())
    if not audit['passed'] or sha(args.experiment/'summary.json') != audit['experiment_summary_sha256']:
        raise ValueError('Completed audited trajectories required')
    env = FullCycleResidualEnv(plan=manifest['reference'], cases=manifest['evaluation_cases'])
    gamma = manifest['config']['gamma']
    body_ids = [env.env.model.body(name).id for name in ('hook', 'right_hook', 'left_ankle_roll_link', 'right_ankle_roll_link')]
    data = mujoco.MjData(env.env.model)
    starts, desired = [], []
    for k, end in enumerate(env.ends):
        start = 0. if k == 0 else env.ends[k-1]
        indices = np.flatnonzero((env.plan['phase'] == 'posture_closure') & (env.plan['time'] > start) & (env.plan['time'] <= end))
        if not len(indices) or indices[0] == 0:
            raise ValueError('Reference lacks a defined closure phase')
        starts.append(float(env.plan['time'][indices[0]-1]))
        j = int(np.argmin(abs(env.plan['time']-end)))
        data.qpos[:] = env.plan['qpos'][j]
        mujoco.mj_kinematics(env.env.model, data)
        desired.append(data.xpos[body_ids].copy())
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for label in ('zero', 'after'):
        for row in summary[label]:
            path = args.experiment/f"{label}_{row['case']}.npz"
            if sha(path) != row['trajectory_sha256']:
                raise ValueError('Trajectory changed')
            with np.load(path) as z:
                n = len(z['rewards'])
                phi = np.zeros(n+1)
                for i in range(1, n):
                    t = float(z['time_s'][i])
                    cycle = min(int(np.searchsorted(env.ends, t, side='right')), len(env.ends)-1)
                    u = float(np.clip((t-starts[cycle])/3., 0., 1.))
                    if not u:
                        continue
                    data.qpos[:] = z['qpos'][i]
                    mujoco.mj_kinematics(env.env.model, data)
                    error = float(np.linalg.norm(data.xpos[body_ids]-desired[cycle], axis=1).max())
                    phi[i] = endpoint_potential(error, u*u*(3-2*u))
                # Both successful and failed terminal states have potential zero.
                shaping = gamma*phi[1:]-phi[:-1]
                weights = gamma**np.arange(n)
                shift = float(weights@shaping)
                return_difference = float(weights@(z['rewards']+shaping)-weights@z['rewards'])
                np.savez_compressed(args.output/f"{label}_{row['case']}_rewards.npz", potentials=phi,
                    original_rewards=z['rewards'], shaping_rewards=shaping)
            results.append(dict(label=label, case=row['case'], original_success=row['success'],
                trajectory_sha256=row['trajectory_sha256'], discounted_shaping_sum=shift,
                discounted_return_difference=return_difference, passed=abs(shift) < 1e-10 and abs(return_difference) < 1e-10,
                nonzero_potential_samples=int(np.count_nonzero(phi)), minimum_potential=float(phi.min()),
                max_abs_shaping_reward=float(abs(shaping).max())))
    result = dict(passed=all(row['passed'] for row in results), gamma=gamma, weight=4., error_scale_m=.01,
        squared_error_cap=9., ramp_seconds=3., closure_starts_s=starts, results=results,
        parent_audit_sha256=sha(args.experiment/'independent_training_audit.json'), source_sha256=sha(__file__),
        potential_source_sha256=sha(ROOT/'training/endpoint_potential.py'),
        scope='Offline prospective reward replay only. Start/terminal potential zero makes each complete discounted return unchanged within floating-point tolerance. This is not evidence of faster PPO learning or repaired task failures.')
    (args.output/'summary.json').write_text(json.dumps(result, indent=2))
    (args.output/'replay_source.py').write_bytes(Path(__file__).read_bytes())
    (args.output/'potential_source.py').write_bytes((ROOT/'training/endpoint_potential.py').read_bytes())
    env.close()
    print(json.dumps(dict(passed=result['passed'], trajectories=len(results), max_discounted_return_difference=max(abs(row['discounted_return_difference']) for row in results))), flush=True)
    if not result['passed']:
        raise AssertionError('Potential reward changes complete discounted returns')


if __name__ == '__main__':
    main()
