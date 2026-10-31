"""Describe sampled reward components and the declared GAE weighting kernel."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.compare_audited_closure_candidates import sha


def main():
    p = argparse.ArgumentParser()
    for name in ('dataset', 'experiment', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    summary = json.loads((a.dataset / 'summary.json').read_text())
    m = json.loads((a.experiment / 'manifest.json').read_text())
    assert summary['passed']
    rows = []
    for row in summary['rows']:
        if row['split'] != 'train':
            continue
        file = a.dataset / row['file']
        assert sha(file) == row['trajectory_sha256']
        with np.load(file) as z:
            t, base = z['time_s'][1:], z['base_rewards']
            bonus = z['rewards'] - base
            rows.append(dict(case=row['case'], success=row['success'], samples=len(base),
                             base_before_closure=float(base[t < 99.66].sum()),
                             base_closure_excluding_final_step=float(base[(t >= 99.66) & (t < 113.65)].sum()),
                             final_base_reward=float(base[-1]), quality_bonus_sum=float(bonus.sum()),
                             total_return=float(z['rewards'].sum()), trajectory_sha256=sha(file)))
    config = m['config']
    gamma, lam = config['gamma'], config['gae_lambda']
    result = dict(rows=rows, gamma=gamma, gae_lambda=lam,
                  rollout_steps_per_worker=config['n_steps'],
                  rollout_seconds_per_worker=config['n_steps']*.02,
                  gae_single_td_weight={str(seconds): float((gamma*lam)**round(seconds/.02))
                                        for seconds in (1, 5, 10, 20)},
                  dataset_summary_sha256=sha(a.dataset/'summary.json'),
                  experiment_manifest_sha256=sha(a.experiment/'manifest.json'),
                  source_sha256=sha(__file__),
                  scope='Descriptive reward accounting on eight existing fit episodes. The GAE kernel is a single future TD residual coefficient within one rollout, not the total policy-gradient influence. Bootstrap values can propagate information across rollouts. This diagnostic neither proves why PPO succeeds/fails nor changes reward, discount or task gates.')
    a.output.mkdir(exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    (a.output/'diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
