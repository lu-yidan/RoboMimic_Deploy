"""Inspect reward signals on audited parent traces, without changing policy."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.full_cycle_residual_env import FullCycleResidualEnv
from training.boundary_landing_env import boundary_bonus
from tools.compare_audited_closure_candidates import load_run, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verified = load_run(args.experiment)
    manifest = json.loads((args.experiment/'manifest.json').read_text())
    env = FullCycleResidualEnv(plan=manifest['reference'], cases=manifest['evaluation_cases'])
    rows = []
    try:
        initial = env.plan['qpos'][0]
        indices = np.flatnonzero(env.plan['phase'] == 'posture_closure')
        start = float(env.plan['time'][indices[0]-1])
        for case in verified['cases']:
            with np.load(args.experiment/f'after_{case}.npz') as trace:
                t = trace['time_s'][1:]
                q = trace['qpos'][1:]
            active = t > start
            closure = abs(q[:, env.env.qadr]-initial[env.env.qadr]).max(axis=1)
            rise = q[:, 2]-initial[2]
            joint = np.maximum(0, closure-.032)/.008
            height = np.maximum(0, abs(rise-.25)-.006)/.004
            u = np.clip((t-start)/3, 0, 1)
            ramp = u*u*(3-2*u)
            bonuses = {mode: np.array([boundary_bonus(c, h, r, mode=mode)
                       for c, h, r in zip(closure, rise, ramp)])
                       for mode in ['average', 'bottleneck']}
            delta = bonuses['bottleneck']-bonuses['average']
            assert np.all(delta <= 1e-14)
            rows.append(dict(case=case, parent_success=verified['rows'][case]['success'],
                closure_samples=int(active.sum()),
                joint_limiting_fraction=float(np.mean(joint[active] > height[active])),
                height_limiting_fraction=float(np.mean(height[active] > joint[active])),
                undiscounted_bonus_sum={mode: float(values.sum()) for mode, values in bonuses.items()},
                mean_bonus_change_during_closure=float(delta[active].mean())))
    finally:
        env.close()
    args.output.mkdir(exist_ok=False)
    report = dict(rows=rows, parent_audit_sha256=verified['audit_sha256'],
                  source_sha256=sha(__file__),
                  scope='Counterfactual reward values on identical audited parent trajectories. This is not training performance, improvement, or an on-policy value estimate.')
    (args.output/'summary.json').write_text(json.dumps(report, indent=2))
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(rows[-3:], indent=2))


if __name__ == '__main__':
    main()
