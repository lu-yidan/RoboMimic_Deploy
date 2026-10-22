"""Compare audited nominal long runs without crediting unexecuted cycles."""
import argparse
import json
from pathlib import Path
import numpy as np
from deploy_mujoco.run_trained_full_cycle import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zero', type=Path, required=True)
    parser.add_argument('--learned', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records, provenance = [], []
    for mode in ('zero', 'learned'):
        directory = getattr(args, mode)
        p = json.loads((directory/'provenance.json').read_text())
        s = json.loads((directory/'summary.json').read_text())
        a = json.loads((directory/'independent_policy_audit.json').read_text())
        if p['policy_mode'] != mode or not a['integrity_passed']:
            raise ValueError('Matching independently audited modes required')
        for filename, key in [('summary.json', 'summary_sha256'),
                              ('provenance.json', 'provenance_sha256'),
                              ('trajectory.npz', 'trajectory_sha256')]:
            if sha(directory/filename) != a[key]:
                raise ValueError('Evidence changed after independent audit: '+filename)
        if sha(directory/'trajectory.npz') != s['trajectory_sha256']:
            raise ValueError('Trajectory changed')
        with np.load(directory/'trajectory.npz') as z:
            end = float(z['time_s'][-1])
            rise = float(z['qpos'][-1, 2]-z['qpos'][0, 2])
        records.append(dict(mode=mode, independently_accepted=a['independently_accepted'],
            time_s=end, measured_root_rise_m=rise, reason=s['reason'],
            executed_cycles=[e for e in a['endpoints'] if e['sample_count'] > 0],
            unexecuted_cycles=[e['cycle'] for e in a['endpoints'] if e['sample_count'] == 0],
            sources={name: sha(directory/name) for name in
                ['summary.json', 'provenance.json', 'independent_policy_audit.json', 'trajectory.npz']}))
        provenance.append(p)
    matching = ('experiment', 'checkpoint_sha256', 'parent_audit_sha256', 'plan_sha256', 'runner_sha256', 'cycle_count', 'adaptation')
    if not all(provenance[0][k] == provenance[1][k] for k in matching):
        raise ValueError('Long-run setups differ beyond policy mode')
    result = dict(records=records, matched_provenance_fields=list(matching),
        interpretation='Single nominal deterministic pair: learned residual completes eight cycles; zero mode may fail earlier. Unequal-duration peak metrics are not compared. This is not multi-seed or perturbation robustness evidence.',
        source_sha256=sha(__file__))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'comparison.json').write_text(json.dumps(result, indent=2))
    (args.output/'comparison_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps([{k:r[k] for k in ('mode','independently_accepted','time_s','measured_root_rise_m')} for r in records]))


if __name__ == '__main__':
    main()
