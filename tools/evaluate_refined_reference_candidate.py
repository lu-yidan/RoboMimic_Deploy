"""Validate a successful cached constant-reference proposal from time zero."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from tools.compare_audited_closure_candidates import load_run, sha
from tools.evaluate_closure_foreaft_reference import evaluate_parallel
from tools.audit_fullcycle_candidate_only import audit
from tools.audit_closure_flexible_prefix import audit as audit_prefix
from tools.audit_trained_closure_controller import audit as audit_fixed
from tools.audit_closure_foreaft_reference import audit as audit_reference


def main():
    p = argparse.ArgumentParser()
    for name in ('parent', 'search', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    parent = load_run(a.parent)
    protocol = json.loads((a.search/'protocol.json').read_text())
    search = json.loads((a.search/'summary.json').read_text())
    assert search['completed'] and search['best']['successes'] == 4
    assert search['protocol_sha256'] == sha(a.search/'protocol.json')
    assert sha(a.search/'search_source.py') == protocol['source_sha256']
    assert Path(protocol['parent']).resolve() == a.parent.resolve()
    assert protocol['parent_audit_sha256'] == parent['audit_sha256']
    assert protocol['teacher_sha256'] == parent['checkpoint_sha256']
    for path, digest in protocol['sources'].items():
        assert sha(path) == digest
    best = search['best']
    selected = [row for row in search['rows'] if row['label'] == best['label']]
    assert len(selected) == 4 and all(row['success'] and row['complete'] for row in selected)
    assert {row['case'] for row in selected} == set(protocol['active_cases'])
    for row in selected:
        assert sha(a.search/row['file']) == row['trajectory_sha256']
        assert row['delta'] == best['delta']
    command = np.clip(np.asarray(protocol['seed_command'])+protocol['delta_scale']*np.asarray(best['delta']), -1., 1.)
    assert command.shape == (5,) and np.isfinite(command).all()
    m = json.loads((a.parent/'manifest.json').read_text())
    old = json.loads((a.parent/'summary.json').read_text())
    assert m['adaptive_reference_command'] == protocol['seed_command']
    assert m['controller_modifier'] == protocol['controller_modifier']
    a.output.mkdir(exist_ok=False)
    shutil.copytree(a.parent/'source_snapshot', a.output/'source_snapshot')
    shutil.copy2(a.parent/'policy.zip', a.output/'policy.zip')
    shutil.copy2(m['prefix_checkpoint'], a.output/'prefix_policy.zip')
    sources = dict(m['sources'])
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if not file:
            continue
        path = Path(file).resolve()
        if path.suffix == '.py' and path.is_relative_to(root):
            relative = path.relative_to(root)
            if relative.parts[0] in ('tools', 'training'):
                destination = a.output/'source_snapshot'/relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(path.read_bytes())
                sources[str(relative)] = sha(path)
    m.update(sources=sources, parent_experiment=str(a.parent.resolve()), parent_manifest_sha256=sha(a.parent/'manifest.json'),
        prefix_checkpoint=str((a.output/'prefix_policy.zip').resolve()), adaptive_reference_command=command.tolist(),
        refinement_search=str(a.search.resolve()), refinement_summary_sha256=sha(a.search/'summary.json'),
        refinement_protocol_sha256=sha(a.search/'protocol.json'), refinement_label=best['label'],
        training_cases={}, evaluation_only=True, new_training_steps=0,
        scope='New five-dimensional constant reference parameters from a bounded local search. Same frozen actor, '
              'physical limits, reference trajectory and original acceptance. Eight complete time-zero development cases; '
              'not PPO improvement, independent geometry validation or automatic promotion.')
    # Preserve actor-training provenance, explicitly mark this stage as evaluation only.
    (a.output/'manifest.json').write_text(json.dumps(m, indent=2))
    summary = dict(checkpoint_sha256=sha(a.output/'policy.zip'), evaluation_pending=True,
        evaluation_only=True, new_training_steps=0, before=old['after'])
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2))
    results = evaluate_parallel(a.output, a.output/'parallel_after', None, workers=2)['trajectories']
    for row in results:
        name = f"after_{row['case']}.npz"
        shutil.copy2(a.output/'parallel_after'/name, a.output/name)
    summary.update(after=results, evaluation_pending=False)
    (a.output/'after.json').write_text(json.dumps(results, indent=2))
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2))
    audit(a.output); audit_prefix(a.output); audit_fixed(a.output); audit_reference(a.output)
    matches = []
    for row in selected:
        with np.load(a.search/row['file']) as cached, np.load(a.output/f"after_{row['case']}.npz") as full:
            errors = {key: float(abs(full[key][4983:]-cached[key]).max()) for key in ('qpos', 'qvel', 'time_s')}
        matches.append(dict(case=row['case'], errors=errors,
            passed=errors['qpos'] < 1e-10 and errors['qvel'] < 1e-8 and errors['time_s'] < 1e-8))
    inactive = []
    for row in results:
        if row['case'] in protocol['active_cases']:
            continue
        with np.load(a.parent/f"after_{row['case']}.npz") as before, np.load(a.output/f"after_{row['case']}.npz") as after:
            checks = {key: bool(np.array_equal(before[key], after[key])) for key in
                ('qpos', 'qvel', 'time_s', 'actions', 'observations', 'torques', 'rewards', 'waist_reference_trace', 'body_reference_trace')}
        inactive.append(dict(case=row['case'], checks=checks, passed=all(checks.values())))
    transfer = dict(passed=all(row['passed'] for row in matches+inactive), cached_to_full=matches,
        inactive_cases=inactive, search_summary_sha256=sha(a.search/'summary.json'),
        full_summary_sha256=sha(a.output/'summary.json'), source_sha256=sha(__file__))
    (a.output/'refinement_transfer_audit.json').write_text(json.dumps(transfer, indent=2))
    assert transfer['passed']
    print(json.dumps(dict(successes=sum(row['success'] for row in results), cases=len(results),
        transfer_audit_passed=True, command=command.tolist(), automatic_promotion=False)), flush=True)


if __name__ == '__main__':
    main()
