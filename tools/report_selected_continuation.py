"""Compare audited full-cycle endpoints against the selected parent."""
import json
from pathlib import Path
import numpy as np
from tools.continue_vector_full_cycle_training import sha


def read_run(path):
    audit = json.loads((path / 'independent_training_audit.json').read_text())
    manifest = json.loads((path / 'manifest.json').read_text())
    summary = json.loads((path / 'summary.json').read_text())
    assert audit['passed'] and not summary['evaluation_pending']
    for filename, key in [('manifest.json', 'experiment_manifest_sha256'),
                          ('summary.json', 'experiment_summary_sha256'),
                          ('policy.zip', 'checkpoint_sha256')]:
        assert sha(path / filename) == audit[key]
    assert sha(manifest['reference']) == manifest['reference_sha256']
    with np.load(manifest['reference']) as plan:
        end = float(plan['time'][-1])
        initial_z = float(plan['qpos'][0, 2])
    rows = {}
    for record in audit['trajectories']:
        if record['label'] != 'after':
            continue
        trajectory = path / f"after_{record['case']}.npz"
        assert record['passed'] and sha(trajectory) == record['trajectory_sha256']
        endpoint = record['independent_endpoints'][0]
        with np.load(trajectory) as z:
            mask = (z['time_s'] > end - 2.) & (z['time_s'] <= end + 1e-7)
            rise = z['qpos'][mask, 2] - initial_z
        rows[record['case']] = dict(success=record['independently_accepted'],
            joint_closure_mrad=1000 * endpoint['joint_closure_rad'],
            root_rise_max_mm=float(rise.max() * 1000) if len(rise) else None,
            root_rise_min_mm=float(rise.min() * 1000) if len(rise) else None,
            joint_speed_rad_s=endpoint['joint_speed_rad_s'],
            effector_error_mm=1000 * endpoint['effector_error_m'],
            endpoint=endpoint)
    assert set(rows) == set(manifest['evaluation_cases'])
    return rows, manifest, summary


def main():
    parent = Path('outputs/closure_margin_control98304_v1')
    candidate = Path('outputs/selected_continuation98304_20261023_v1')
    before, pm, _ = read_run(parent)
    after, cm, summary = read_run(candidate)
    assert pm['reference_sha256'] == cm['reference_sha256']
    assert pm['evaluation_cases'] == cm['evaluation_cases']
    assert cm['parent_checkpoint_sha256'] == sha(parent / 'policy.zip')
    rows = [dict(case=name, before=before[name], after=after[name]) for name in before]
    gained = [name for name in before if after[name]['success'] and not before[name]['success']]
    lost = [name for name in before if before[name]['success'] and not after[name]['success']]
    result = dict(rows=rows, gained=gained, lost=lost,
        additional_steps=summary['additional_steps'],
        audit_hashes={str(p): sha(p / 'independent_training_audit.json') for p in (parent, candidate)},
        scope='Known five-case deterministic evaluation. No automatic promotion; grid validation remains required.')
    out = Path('outputs/selected_continuation_comparison_20261023_v1')
    out.mkdir(exist_ok=False)
    (out / 'comparison.json').write_text(json.dumps(result, indent=2))
    (out / 'report_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
