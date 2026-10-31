"""Verify full-cycle composite evaluations exactly preserve the frozen prefix."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.continue_vector_full_cycle_training import sha


def audit(p):
    p = Path(p)
    m = json.loads((p / 'manifest.json').read_text())
    audit = json.loads((p / 'independent_training_audit.json').read_text())
    parent = Path(m['parent_experiment'])
    old = json.loads((parent / 'independent_training_audit.json').read_text())
    assert audit['passed'] and old['passed']
    assert sha(p / 'summary.json') == audit['experiment_summary_sha256']
    count = round(m['suffix_start_seconds'] / .02)
    rows = []
    for case in m['evaluation_cases']:
        current_row = next(r for r in audit['trajectories'] if r['label'] == 'after' and r['case'] == case)
        old_row = next(r for r in old['trajectories'] if r['label'] == 'after' and r['case'] == case)
        a, b = parent / f'after_{case}.npz', p / f'after_{case}.npz'
        assert sha(a) == old_row['trajectory_sha256'] and sha(b) == current_row['trajectory_sha256']
        with np.load(a) as x, np.load(b) as y:
            exact = {}
            for key in ('qpos', 'qvel', 'time_s', 'observations', 'actions', 'torques', 'rewards'):
                n = count + 1 if key in ('qpos', 'qvel', 'time_s') else count
                exact[key] = bool(len(y[key]) >= n and np.array_equal(x[key][:n], y[key][:n]))
        rows.append(dict(case=case, passed=all(exact.values()), exact=exact))
    result = dict(passed=all(r['passed'] for r in rows), rows=rows, source_sha256=sha(__file__),
        scope='Every evaluated composite starts at time zero; original prefix arrays match through suffix entry.')
    (p / 'composite_prefix_audit.json').write_text(json.dumps(result, indent=2))
    (p / 'composite_prefix_audit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))
    assert result['passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=Path)
    audit(parser.parse_args().experiment)
