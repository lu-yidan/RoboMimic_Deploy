"""Recover interrupted evaluation without changing completed trajectories or sources."""
import argparse
import json
from pathlib import Path
import shutil
from tools.evaluate_closure_foreaft_reference import evaluate_parallel, sha
from tools.compare_audited_closure_candidates import load_run
from tools.audit_fullcycle_candidate_only import audit
from tools.audit_closure_flexible_prefix_v2 import audit as audit_prefix
from tools.audit_trained_closure_controller_v2 import audit as audit_fixed
from tools.audit_closure_foreaft_reference import audit as audit_added


def main():
    p = argparse.ArgumentParser()
    p.add_argument('experiment', type=Path)
    a = p.parse_args(); root = a.experiment
    m = json.loads((root/'manifest.json').read_text())
    s = json.loads((root/'summary.json').read_text())
    assert s['evaluation_pending']
    assert not (root/'regression.json').exists()
    parent = Path(m['parent_experiment'])
    assert sha(parent/'manifest.json') == m['parent_manifest_sha256']
    grid = load_run(parent)
    found = {}; provenance = {}
    for case in m['evaluation_cases']:
        directory = root/'parallel_after'/case
        if (directory/'result.json').exists():
            row = json.loads((directory/'result.json').read_text())
            assert row['case'] == case
            assert sha(directory/f'after_{case}.npz') == row['trajectory_sha256']
            found[case] = row; provenance[case] = directory
    missing = [c for c in m['evaluation_cases'] if c not in found]
    if missing:
        recovered = evaluate_parallel(root, root/'recovery_after', missing, workers=1)
        for row in recovered['trajectories']:
            found[row['case']] = row
            provenance[row['case']] = root/'recovery_after'/row['case']
    rows = [found[c] for c in m['evaluation_cases']]
    for row in rows:
        name = f"after_{row['case']}.npz"
        assert not (root/name).exists()
        shutil.copy2(provenance[row['case']]/name, root/name)
    (root/'recovery_provenance.json').write_text(json.dumps(dict(
        retried_cases=missing, sources={k:str(v) for k,v in provenance.items()},
        original_summary_sha256=sha(root/'summary.json'), recovery_source_sha256=sha(__file__)),indent=2))
    s.update(after=rows, evaluation_pending=False)
    (root/'after.json').write_text(json.dumps(rows,indent=2))
    (root/'summary.json').write_text(json.dumps(s,indent=2))
    audit(root); audit_prefix(root); audit_fixed(root); audit_added(root)
    result = dict(successes=sum(r['success'] for r in rows), cases=len(rows),
        lost=[r['case'] for r in rows if grid['rows'][r['case']]['success'] and not r['success']],
        gained=[r['case'] for r in rows if not grid['rows'][r['case']]['success'] and r['success']],
        automatic_promotion=False)
    (root/'regression.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
