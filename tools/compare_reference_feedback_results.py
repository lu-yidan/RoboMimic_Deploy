"""Compare task outcomes while explicitly retaining different control contracts."""
import argparse
import csv
import json
from pathlib import Path
from tools.compare_audited_closure_candidates import load_run, sha
from tools.evaluate_seeded_reference_feedback import FEEDBACK_CONTRACT


def main():
    p = argparse.ArgumentParser()
    for name in ('before', 'after', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    before, after = load_run(a.before), load_run(a.after)
    execution = json.loads((a.after/'summary.json').read_text())
    after.pop('additional_steps', None)  # Legacy loader assumes 26D teacher-step units.
    after['additional_macro_steps'] = execution.get('additional_macro_steps', 0)
    after['teacher_transitions'] = execution.get('teacher_steps', 0)
    m = json.loads((a.after/'manifest.json').read_text())
    bm = json.loads((a.before/'manifest.json').read_text())
    assert m['observation_contract'] == FEEDBACK_CONTRACT
    assert before['cases'] == after['cases'] and before['reference_sha256'] == after['reference_sha256']
    assert before['checkpoint_sha256'] == m['teacher_checkpoint_sha256']
    assert before['controller_modifier'] == after['controller_modifier']
    assert before['adaptive_reference_command'] == after['adaptive_reference_command']
    assert m['prefix_checkpoint_sha256'] == bm['prefix_checkpoint_sha256']
    rows = []
    for case in before['cases']:
        row = dict(case=case)
        for label, run in [('before', before), ('after', after)]:
            row.update({label+'_'+key: value for key, value in run['rows'][case].items()})
        old, new = row['before_closure_mrad'], row['after_closure_mrad']
        row['closure_change_mrad'] = new-old if old is not None and new is not None else None
        rows.append(row)
    a.output.mkdir(exist_ok=False)
    result = dict(before=before, after=after, rows=rows,
        gained=[r['case'] for r in rows if r['after_success'] and not r['before_success']],
        lost=[r['case'] for r in rows if r['before_success'] and not r['after_success']],
        control_contracts=dict(before=bm['observation_contract'], after=m['observation_contract']),
        source_sha256=sha(__file__),
        scope='Fixed-reference baseline versus learned bounded reference feedback over the same frozen teacher. '
              'Single-cycle known development cases, different control decisions; not an equal-authority pure 26D PPO ablation. '
              'No independent generalization or automatic promotion.')
    (a.output/'comparison.json').write_text(json.dumps(result, indent=2))
    with (a.output/'comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(gained=result['gained'], lost=result['lost'], rows=rows)), flush=True)


if __name__ == '__main__':
    main()
