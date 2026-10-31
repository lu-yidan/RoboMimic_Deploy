"""Pair independent and episode-persistent exploration; no policy selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.audit_multirung_cycle import sha


def main():
    p = argparse.ArgumentParser()
    for name in ('independent', 'persistent', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    paths = (a.independent, a.persistent)
    protocols = [json.loads((path/'protocol.json').read_text()) for path in paths]
    results = [json.loads((path/'summary.json').read_text()) for path in paths]
    shared = ('active_cases', 'sigmas', 'trials', 'sampling_seed_base', 'initial_checkpoint_sha256',
              'seed_command', 'delta_scale', 'hold_steps', 'physical_gamma', 'teacher_sha256', 'sources')
    assert all(protocols[0][key] == protocols[1][key] for key in shared)
    assert all(result['completed'] and len(result['rows']) == 36 for result in results)
    assert all(result['protocol_sha256'] == sha(path/'protocol.json') for path, result in zip(paths, results))
    assert sha(paths[0]/'probe_source.py') == protocols[0]['source_sha256']
    assert sha(paths[1]/'persistent_probe_source.py') == protocols[1]['source_sha256']
    def identity(row):
        return row['case'], row['sigma'], row['trial']
    lookup = [{identity(row): row for row in result['rows']} for result in results]
    assert lookup[0].keys() == lookup[1].keys() and len(lookup[0]) == 36
    paired = []
    for key in sorted(lookup[0]):
        rows = [table[key] for table in lookup]
        assert rows[0]['seed'] == rows[1]['seed']
        arrays = []
        for path, row in zip(paths, rows):
            assert sha(path/row['file']) == row['trajectory_sha256']
            with np.load(path/row['file']) as z:
                data = {name: z[name].copy() for name in z.files}
            assert all(np.isfinite(value).all() for value in data.values())
            assert abs(data['actions']).max() <= 1.
            assert len(data['actions']) == row['steps'] == 140
            assert row['teacher_steps'] == 700
            arrays.append(data)
        assert np.array_equal(arrays[0]['actions'][0], arrays[1]['actions'][0])
        assert np.array_equal(arrays[0]['observations'][0], arrays[1]['observations'][0])
        assert np.array_equal(arrays[1]['actions'], np.broadcast_to(arrays[1]['actions'][0], arrays[1]['actions'].shape))
        assert not np.array_equal(arrays[0]['actions'], np.broadcast_to(arrays[0]['actions'][0], arrays[0]['actions'].shape))
        paired.append(dict(case=key[0], sigma=key[1], trial=key[2], seed=rows[0]['seed'],
            first_action=arrays[0]['actions'][0].tolist(), independent_success=rows[0]['success'],
            persistent_success=rows[1]['success'], independent_final_closure_rad=rows[0]['info']['current_closure_rad'],
            persistent_final_closure_rad=rows[1]['info']['current_closure_rad']))
    result = dict(integrity_passed=True, pairs=paired,
        independent_successes=sum(row['independent_success'] for row in paired),
        persistent_successes=sum(row['persistent_success'] for row in paired),
        summary_sha256=[sha(path/'summary.json') for path in paths], source_sha256=sha(__file__),
        scope='Known cached 14-second suffixes with fixed actor, matched initial observations and random first actions. '
              'Temporal action correlation is deliberately changed. Final-frame closure is not the worst endpoint error. '
              'No training, from-zero validation, new geometry or learned-policy improvement claim.')
    a.output.mkdir(exist_ok=False)
    (a.output/'comparison.json').write_text(json.dumps(result, indent=2))
    (a.output/'comparison_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key: value for key, value in result.items() if key != 'pairs'}))


if __name__ == '__main__':
    main()
