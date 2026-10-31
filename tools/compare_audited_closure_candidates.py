"""Compare complete, independently audited candidates without changing selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_run(path):
    path = path.resolve()
    manifest = json.loads((path/'manifest.json').read_text())
    summary = json.loads((path/'summary.json').read_text())
    files = [p for p in (path/'independent_training_audit.json', path/'independent_candidate_audit.json') if p.exists()]
    assert len(files) == 1, 'Exactly one independent audit expected'
    audit = json.loads(files[0].read_text())
    assert audit['passed'] and not summary.get('evaluation_pending', False)
    for name, key in [('manifest.json', 'experiment_manifest_sha256'), ('summary.json', 'experiment_summary_sha256'), ('policy.zip', 'checkpoint_sha256')]:
        assert sha(path/name) == audit[key]
    assert sha(manifest['reference']) == manifest['reference_sha256']
    with np.load(manifest['reference']) as reference:
        initial_z = float(reference['qpos'][0, 2])
        end = float(reference['time'][-1])
    rows = {}
    for row in audit['trajectories']:
        if row['label'] != 'after':
            continue
        assert row['passed'] and row['case'] not in rows
        trajectory = path/f"after_{row['case']}.npz"
        assert sha(trajectory) == row['trajectory_sha256']
        endpoints = row['independent_endpoints']
        endpoint = endpoints[0] if endpoints else None
        with np.load(trajectory) as trace:
            mask = (trace['time_s'] > end-2) & (trace['time_s'] <= end+1e-7)
            rise = trace['qpos'][mask, 2]-initial_z
            complete = abs(float(trace['time_s'][-1])-end) < 1e-6
        closure = endpoint['joint_closure_rad'] if endpoint else None
        height_error = float(abs(rise-.25).max()) if len(rise) and complete else None
        rows[row['case']] = dict(success=row['independently_accepted'], complete=complete,
            closure_mrad=1000*closure if closure is not None else None,
            rise_max_mm=float(rise.max()*1000) if len(rise) else None,
            scaled_margin=min((.04-closure)/.001, (.01-height_error)/.0001) if closure is not None and height_error is not None else None)
    assert set(rows) == set(manifest['evaluation_cases'])
    return dict(directory=str(path), audit_sha256=sha(files[0]), checkpoint_sha256=sha(path/'policy.zip'),
        reference_sha256=manifest['reference_sha256'], cases=manifest['evaluation_cases'],
        controller_modifier=manifest.get('controller_modifier'), adaptive_reference_command=manifest.get('adaptive_reference_command'), rows=rows,
        additional_steps=summary.get('additional_steps', 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    before, after = load_run(args.before), load_run(args.after)
    assert before['reference_sha256'] == after['reference_sha256']
    assert before['cases'] == after['cases'], 'Compare the same complete case suite'
    assert before['controller_modifier'] == after['controller_modifier'], 'Controller authority differs'
    assert before['adaptive_reference_command'] == after['adaptive_reference_command'], 'Additional reference commands differ'
    rows = []
    for case in before['cases']:
        a, b = before['rows'][case], after['rows'][case]
        row = dict(case=case)
        for prefix, source in [('before', a), ('after', b)]:
            row.update({prefix+'_'+key: value for key, value in source.items()})
        row['closure_change_mrad'] = b['closure_mrad']-a['closure_mrad'] if a['closure_mrad'] is not None and b['closure_mrad'] is not None else None
        rows.append(row)
    result = dict(before=before, after=after, rows=rows,
        gained=[r['case'] for r in rows if r['after_success'] and not r['before_success']],
        lost=[r['case'] for r in rows if r['before_success'] and not r['after_success']],
        source_sha256=sha(__file__), scope='Same-reference complete single-cycle development evaluation, fixed controller authority. No independent generalization or automatic promotion.')
    args.output.mkdir(exist_ok=False)
    (args.output/'comparison.json').write_text(json.dumps(result, indent=2))
    with (args.output/'comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.output/'report_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(gained=result['gained'], lost=result['lost'], rows=rows), indent=2))


if __name__ == '__main__':
    main()
