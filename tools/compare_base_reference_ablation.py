"""Compare the base tracking reference intervention against identical joint actions."""
import json
from pathlib import Path
import numpy as np
from tools.continue_vector_full_cycle_training import sha


def main():
    baseline = Path('outputs/load_margin_fixed_residual_v1')
    candidate = Path('outputs/load_margin_base_reference_v1')
    protocol = json.loads((candidate/'protocol.json').read_text())
    assert protocol['joint_baseline_selection_sha256'] == sha(baseline/'selection.json')
    rows = []
    for case in protocol['cases']:
        records = []
        for root in (baseline, candidate):
            p = root/case
            m = json.loads((p/'manifest.json').read_text())
            s = json.loads((p/'summary.json').read_text())
            a = json.loads((p/'independent_audit.json').read_text())
            assert a['integrity_passed']
            assert a['summary_sha256'] == sha(p/'summary.json')
            assert a['trajectory_sha256'] == sha(p/'trajectory.npz')
            assert s['manifest_sha256'] == sha(p/'manifest.json')
            records.append((m, s, a))
        bm, bs, ba = records[0]
        cm, cs, ca = records[1]
        for key in ('offsets', 'reference_sha256', 'bundle_manifest_sha256',
                    'normal_bias_y_deg', 'memory_enabled', 'contract', 'horizon_s', 'action_rule'):
            assert bm[key] == cm[key], key
        assert 'reference_modifier' not in bm
        assert cm['reference_modifier'] == dict(type='bounded-base-reference-57to65s-cubic2s-v1',
            base_bias=protocol['base_reference_bias_m'], start_s=57., end_s=65., ramp_s=2.)
        with np.load(baseline/case/'trajectory.npz') as x, np.load(candidate/case/'trajectory.npz') as y:
            nx = int(np.searchsorted(x['time_s'], 57., side='left'))
            ny = int(np.searchsorted(y['time_s'], 57., side='left'))
            assert nx == ny
            exact = {k: bool(np.array_equal(x[k][:nx], y[k][:ny])) for k in
                     ('qpos', 'qvel', 'time_s', 'actions', 'observations', 'torques', 'rewards')}
            assert all(exact.values())
            n = min(len(x['actions']), len(y['actions']))
            assert np.array_equal(x['actions'][:n], y['actions'][:n])
            assert np.array_equal(x['time_s'][:n], y['time_s'][:n])
        before, after = ba['independently_accepted'], ca['independently_accepted']
        rows.append(dict(case=case, joint_only_passed=before, base_reference_passed=after,
            fixed=not before and after, regressed=before and not after,
            joint_only_time_s=bs['time_s'], base_reference_time_s=cs['time_s'], reason=cs['reason'],
            prefix_exact=exact, prefix_samples=nx, common_action_samples=n,
            audit_hashes=[sha(root/case/'independent_audit.json') for root in (baseline,candidate)]))
    fixed = [r['case'] for r in rows if r['fixed']]
    regressed = [r['case'] for r in rows if r['regressed']]
    result = dict(cases=rows, fixed_cases=fixed, regressed_cases=regressed,
        selected=bool(fixed and not regressed), protocol_sha256=sha(candidate/'protocol.json'),
        source_sha256=sha(__file__), scope='Known-case reference ablation. Identical joint actions; bounded base tracking reference differs. Not a learned policy or held-out evaluation.')
    (candidate/'selection.json').write_text(json.dumps(result, indent=2))
    (candidate/'selection_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
