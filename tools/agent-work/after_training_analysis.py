"""Post-training verdict for the geometry PPO run: baseline vs final, with closure/rise metrics.

Reuses the audited formulas (same as tools/compare_audited_closure_candidates.py):
  closure = independent_endpoints[0].joint_closure_rad  (max joint deviation)
  rise    = max(root z - reference initial z) over the last 2 s
  margin  = min((0.040 - closure)/0.001, (0.010 - max|rise-0.250|)/0.0001)
"""
import argparse
import json
from pathlib import Path

import numpy as np

SEVEN = ['training_00', 'training_01', 'training_02', 'training_09',
         'training_13', 'training_14', 'training_15']


def load_run(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    with np.load(manifest['reference']) as reference:
        initial_z = float(reference['qpos'][0, 2])
        end = float(reference['time'][-1])
    reasons = {}
    after_path = directory / 'after.json'
    if after_path.exists():
        for row in json.loads(after_path.read_text()):
            reasons[row['case']] = row.get('reason')
    audit_path = directory / 'independent_candidate_audit.json'
    audit_rows = {}
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        for row in audit['trajectories']:
            if row['label'] == 'after':
                audit_rows[row['case']] = row
    elif after_path.exists():
        # audit missing (evaluation crashed after writing after.json): still report pass/fail
        for row in json.loads(after_path.read_text()):
            audit_rows[row['case']] = dict(case=row['case'], experiment_success=row['success'],
                                           independently_accepted=row['success'],
                                           independent_endpoints=[])
    rows = {}
    for case, row in audit_rows.items():
        endpoints = row['independent_endpoints']
        closure = endpoints[0]['joint_closure_rad'] if endpoints else None
        trajectory = directory / f'after_{case}.npz'
        rise_max = complete = None
        if trajectory.exists():
            with np.load(trajectory) as trace:
                mask = (trace['time_s'] > end - 2) & (trace['time_s'] <= end + 1e-7)
                rise = trace['qpos'][mask, 2] - initial_z
                rise_max = float(rise.max() * 1000) if len(rise) else None
                complete = abs(float(trace['time_s'][-1]) - end) < 1e-6
        margin = None
        if closure is not None and complete and rise_max is not None:
            height_error_mm = abs(rise_max / 1000 - .25) * 1000
            margin = min((.040 - closure) / .001, (.010 - height_error_mm / 1000) / .0001)
        rows[case] = dict(success=bool(row['experiment_success']),
                          independently_accepted=bool(row['independently_accepted']),
                          reason=reasons.get(case), complete=complete,
                          closure_mrad=1000 * closure if closure is not None else None,
                          rise_max_mm=rise_max, scaled_margin=margin)
    return rows


def fmt(value, spec):
    return 'None' if value is None else format(value, spec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = load_run(args.baseline)
    final = load_run(args.run)
    cases = [c for c in base if c in final]
    table = []
    for case in cases:
        a, b = base[case], final[case]
        table.append(dict(
            case=case,
            baseline_success=a['success'], final_success=b['success'],
            baseline_reason=a['reason'], final_reason=b['reason'],
            baseline_closure_mrad=a['closure_mrad'], final_closure_mrad=b['closure_mrad'],
            delta_closure_mrad=(b['closure_mrad'] - a['closure_mrad']
                                if a['closure_mrad'] is not None and b['closure_mrad'] is not None else None),
            baseline_rise_max_mm=a['rise_max_mm'], final_rise_max_mm=b['rise_max_mm'],
            delta_rise_mm=(b['rise_max_mm'] - a['rise_max_mm']
                           if a['rise_max_mm'] is not None and b['rise_max_mm'] is not None else None),
            baseline_margin=a['scaled_margin'], final_margin=b['scaled_margin']))
    result = dict(
        run=str(args.run), baseline=str(args.baseline),
        baseline_successes=sum(r['baseline_success'] for r in table),
        final_successes=sum(r['final_success'] for r in table),
        cases=len(table),
        gained=[r['case'] for r in table if r['final_success'] and not r['baseline_success']],
        lost=[r['case'] for r in table if r['baseline_success'] and not r['final_success']],
        seven=[r for r in table if r['case'] in SEVEN],
        rows=table)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))

    lines = [f"# geometry PPO verdict  baseline {result['baseline_successes']}/{result['cases']}"
             f" -> final {result['final_successes']}/{result['cases']}",
             f"gained={result['gained']}  lost={result['lost']}", '',
             '| case | base | final | closure mrad (base -> final) | rise max mm (base -> final) |',
             '|---|---|---|---|---|']
    for r in table:
        if r['case'] not in SEVEN:
            continue
        lines.append('| {} | {} | {} | {} -> {} | {} -> {} |'.format(
            r['case'], 'PASS' if r['baseline_success'] else 'fail',
            'PASS' if r['final_success'] else 'fail',
            fmt(r['baseline_closure_mrad'], '.3f'), fmt(r['final_closure_mrad'], '.3f'),
            fmt(r['baseline_rise_max_mm'], '.3f'), fmt(r['final_rise_max_mm'], '.3f')))
    lines += ['', '| all cases | base | final |', '|---|---|---|']
    for r in table:
        if r['case'] in SEVEN:
            continue
        lines.append(f"| {r['case']} | {'PASS' if r['baseline_success'] else 'fail'} | "
                     f"{'PASS' if r['final_success'] else 'fail'} |")
    markdown = '\n'.join(lines) + '\n'
    (args.output.parent / 'geometry_verdict.md').write_text(markdown, encoding='utf-8')
    print(markdown)


if __name__ == '__main__':
    main()
