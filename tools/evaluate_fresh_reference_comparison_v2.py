"""Evaluate one of three frozen controllers on eight freshly declared geometry points."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from tools.compare_audited_closure_candidates import load_run, sha
from tools.evaluate_closure_foreaft_reference import evaluate_parallel
from tools.audit_fullcycle_candidate_only import audit
from tools.audit_closure_flexible_prefix_v2 import audit as audit_prefix
from tools.audit_trained_closure_controller_v2 import audit as audit_fixed
from tools.audit_closure_foreaft_reference import audit as audit_added


def main():
    p = argparse.ArgumentParser()
    for name in ('protocol', 'candidate', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--role', choices=('baseline', 'trained', 'refined'), required=True)
    p.add_argument('--baseline-result', type=Path)
    p.add_argument('--workers', type=int, default=2, choices=(1, 2))
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(a.protocol.read_text())
    run = load_run(a.candidate)
    m = json.loads((a.candidate / 'manifest.json').read_text())
    summary = json.loads((a.candidate / 'summary.json').read_text())
    selection = protocol['candidates'][a.role]
    assert a.candidate.resolve() == Path(selection['directory']).resolve()
    assert sha(a.candidate/'manifest.json') == selection['manifest_sha256']
    assert run['checkpoint_sha256'] == selection['checkpoint_sha256']
    assert run['adaptive_reference_command'] == selection['command']
    assert run['reference_sha256'] == selection['reference_sha256']
    assert m['controller_modifier'] == protocol['controller_modifier']
    assert m['adaptive_reference_scales'] == protocol['reference_scales']
    assert sum(row['success'] for row in run['rows'].values()) >= selection['minimum_development_successes']
    if selection['refinement_transfer_gate_required']:
        transfer = json.loads((a.candidate/'refinement_transfer_audit.json').read_text())
        assert transfer['passed'] and transfer['full_summary_sha256'] == sha(a.candidate/'summary.json')
    if a.role == 'baseline':
        assert a.baseline_result is None
        before = None
    else:
        assert a.baseline_result is not None
        before = load_run(a.baseline_result)
        bm = json.loads((a.baseline_result/'manifest.json').read_text())
        assert bm['holdout_protocol_sha256'] == sha(a.protocol) and bm['fresh_role'] == 'baseline'
        assert before['cases'] == protocol['cases']
        assert before['reference_sha256'] == run['reference_sha256']
        assert before['controller_modifier'] == run['controller_modifier']
        assert before['checkpoint_sha256'] == protocol['candidates']['baseline']['checkpoint_sha256']
        assert bm['prefix_checkpoint_sha256'] == m['prefix_checkpoint_sha256']
    a.output.mkdir(exist_ok=False)
    shutil.copytree(a.candidate / 'source_snapshot', a.output / 'source_snapshot')
    shutil.copy2(a.candidate / 'policy.zip', a.output / 'policy.zip')
    shutil.copy2(m['prefix_checkpoint'], a.output / 'prefix_policy.zip')
    shutil.copy2(a.protocol, a.output / 'holdout_protocol.json')
    sources = dict(m['sources'])
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if not name:
            continue
        source = Path(name).resolve()
        if source.suffix != '.py' or not source.is_relative_to(root):
            continue
        relative = source.relative_to(root)
        if relative.parts[0] not in ('tools', 'training'):
            continue
        target = a.output / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        sources[str(relative)] = sha(source)
    parent = a.baseline_result if before else a.candidate
    m.update(sources=sources, evaluation_cases=protocol['cases'], training_cases={},
             original_training_cases=m.get('training_cases', {}),
             parent_experiment=str(parent.resolve()),
             parent_manifest_sha256=sha(parent / 'manifest.json'),
             candidate_experiment=str(a.candidate.resolve()),
             candidate_audit_sha256=run['audit_sha256'],
             prefix_checkpoint=str((a.output / 'prefix_policy.zip').resolve()),
             evaluation_only=True, new_training_steps=0,
             require_inactive_case_exact_parent=False,
             holdout_protocol_sha256=sha(a.protocol), fresh_role=a.role,
             selection_manifest_sha256=selection['manifest_sha256'],
             scope='Predeclared eight fresh geometries; controller identity and constant commands frozen before rollout, no fitting or checkpoint selection. Inactive-case equality with the original eight-case parent is inapplicable because these are new geometries. Paired trained run verifies the frozen prefix against the baseline on these exact geometries.')
    (a.output / 'manifest.json').write_text(json.dumps(m, indent=2))
    s = dict(checkpoint_sha256=sha(a.output / 'policy.zip'), evaluation_pending=True,
             evaluation_only=True, new_training_steps=0)
    (a.output / 'summary.json').write_text(json.dumps(s, indent=2))
    rows = evaluate_parallel(a.output, a.output / 'parallel_after', None,
                             workers=a.workers)['trajectories']
    for row in rows:
        name = f"after_{row['case']}.npz"
        shutil.copy2(a.output / 'parallel_after' / name, a.output / name)
    s.update(after=rows, evaluation_pending=False)
    (a.output / 'summary.json').write_text(json.dumps(s, indent=2))
    (a.output / 'after.json').write_text(json.dumps(rows, indent=2))
    audit(a.output)
    if before:
        audit_prefix(a.output)
    audit_fixed(a.output)
    audit_added(a.output)
    accepted = load_run(a.output)
    result = dict(successes=sum(r['success'] for r in accepted['rows'].values()),
                  cases=len(rows), protocol_sha256=sha(a.protocol),
                  strata={}, automatic_promotion=False)
    for group in sorted(set(protocol['strata'].values())):
        keys = [k for k, value in protocol['strata'].items() if value == group]
        result['strata'][group] = dict(cases=len(keys), successes=sum(accepted['rows'][k]['success'] for k in keys))
    if before:
        result['gained'] = [k for k in accepted['rows'] if accepted['rows'][k]['success'] and not before['rows'][k]['success']]
        result['lost'] = [k for k in accepted['rows'] if not accepted['rows'][k]['success'] and before['rows'][k]['success']]
    (a.output / 'geometry_result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
