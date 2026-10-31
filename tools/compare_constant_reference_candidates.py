"""Compare constant controller parameters under the same frozen actor and limits."""
import argparse
import csv
import json
from pathlib import Path
from tools.compare_audited_closure_candidates import load_run, sha


def main():
    p = argparse.ArgumentParser()
    for name in ('before', 'after', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    before, after = load_run(a.before), load_run(a.after)
    for key in ('cases', 'checkpoint_sha256', 'reference_sha256', 'controller_modifier'):
        assert before[key] == after[key], key
    manifests = [json.loads((path/'manifest.json').read_text()) for path in (a.before, a.after)]
    assert manifests[0]['adaptive_reference_scales'] == manifests[1]['adaptive_reference_scales']
    assert manifests[0]['prefix_checkpoint_sha256'] == manifests[1]['prefix_checkpoint_sha256']
    rows = []
    for case in before['cases']:
        row = dict(case=case)
        for label, run in (('before', before), ('after', after)):
            row.update({label+'_'+key: value for key, value in run['rows'][case].items()})
        old, new = row['before_closure_mrad'], row['after_closure_mrad']
        row['closure_change_mrad'] = new-old if old is not None and new is not None else None
        rows.append(row)
    result = dict(before=before, after=after, rows=rows,
        gained=[row['case'] for row in rows if row['after_success'] and not row['before_success']],
        lost=[row['case'] for row in rows if row['before_success'] and not row['after_success']],
        commands=[manifest['adaptive_reference_command'] for manifest in manifests],
        source_sha256=sha(__file__),
        scope='Same frozen actor and absolute reference bounds; constant controller parameters differ. '
              'Already-used development cases. Not new policy learning, independent geometry evidence or automatic promotion.')
    a.output.mkdir(exist_ok=False)
    (a.output/'comparison.json').write_text(json.dumps(result, indent=2))
    with (a.output/'comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(gained=result['gained'], lost=result['lost'], rows=rows)), flush=True)


if __name__ == '__main__':
    main()
