"""Reject omitted active traces and fabricated traces before controller entry."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import numpy as np
from tools.audit_trained_closure_controller_v2 import audit
from tools.compare_audited_closure_candidates import sha


def main():
    parser = argparse.ArgumentParser()
    for key in ('baseline', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    original = json.loads((args.baseline/'summary.json').read_text())
    manifest = json.loads((args.baseline/'manifest.json').read_text())
    args.output.mkdir(exist_ok=False)
    rows = []
    variants = [('early_empty_valid', 'fresh_boundary_03', None, True),
                ('early_fabricated_trace', 'fresh_boundary_03', 'fabricate', False),
                ('active_missing_trace', 'fresh_boundary_02', 'remove', False)]
    for name, case, mutation, expected in variants:
        path = args.output/name
        path.mkdir()
        result = dict(next(row for row in original['after'] if row['case'] == case))
        with np.load(args.baseline/f'after_{case}.npz') as z:
            arrays = {key: z[key].copy() for key in z.files}
        if mutation == 'fabricate':
            arrays['waist_reference_trace'] = np.zeros((1, 10))
        elif mutation == 'remove':
            assert len(arrays['waist_reference_trace']) > 0
            arrays['waist_reference_trace'] = np.zeros((0, 10))
        trajectory = path/f'after_{case}.npz'
        np.savez_compressed(trajectory, **arrays)
        result['trajectory_sha256'] = sha(trajectory)
        (path/'manifest.json').write_text(json.dumps(manifest))
        (path/'summary.json').write_text(json.dumps(dict(after=[result])))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                audit(path)
        except AssertionError:
            assert not expected
        report = json.loads((path/'controller_modifier_audit_v2.json').read_text())
        assert report['passed'] == expected
        rows.append(dict(name=name, expected=expected, actual=report['passed'],
                         passed=True, audit_sha256=sha(path/'controller_modifier_audit_v2.json')))
    result = dict(passed=True, rows=rows, source_sha256=sha(__file__),
                  baseline_summary_sha256=sha(args.baseline/'summary.json'),
                  scope='Deliberately modified local audit fixtures only. No simulation, policy updates or task-success changes.')
    (args.output/'summary.json').write_text(json.dumps(result, indent=2))
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
