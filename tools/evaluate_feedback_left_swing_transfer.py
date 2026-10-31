"""Transfer frozen five-delta feedback to the separately audited left-foot lift pilot."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from tools.compare_audited_closure_candidates import load_run, sha
from tools.evaluate_seeded_reference_feedback import evaluate_parallel, FEEDBACK_CONTRACT
from tools.audit_seeded_reference_feedback import audit
from tools.audit_closure_flexible_prefix import audit as audit_prefix
from tools.audit_trained_closure_controller import audit as audit_fixed


def main():
    p = argparse.ArgumentParser()
    for name in ('candidate', 'grid-parent', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    candidate, grid = load_run(a.candidate), load_run(a.grid_parent)
    m = json.loads((a.candidate/'manifest.json').read_text())
    gm = json.loads((a.grid_parent/'manifest.json').read_text())
    assert m['observation_contract'] == FEEDBACK_CONTRACT
    plan_directory=Path(gm['reference']).parent
    plan=json.loads((plan_directory/'manifest.json').read_text())
    pilot=json.loads((a.grid_parent/'pilot_audit.json').read_text())
    geometry=json.loads((plan_directory/'interpolation_audit.json').read_text())
    assert pilot['passed'] and pilot['manifest_sha256']==sha(a.grid_parent/'manifest.json')
    assert pilot['summary_sha256']==sha(a.grid_parent/'summary.json')
    assert geometry['passed'] and geometry['plan_sha256']==grid['reference_sha256']
    assert plan['parent_sha256']==candidate['reference_sha256']
    assert plan['plan_sha256']==grid['reference_sha256']
    assert m['teacher_checkpoint_sha256'] == grid['checkpoint_sha256']
    assert sha(m['teacher_checkpoint']) == m['teacher_checkpoint_sha256']
    assert m['prefix_checkpoint_sha256'] == gm['prefix_checkpoint_sha256']
    assert m['suffix_start_seconds'] == gm['suffix_start_seconds'] == 99.66
    assert m['controller_modifier'] == gm['controller_modifier']
    assert m['adaptive_reference_command'] == gm['adaptive_reference_command']
    assert set(grid['cases']) == {'foot_z_plus15','fresh_boundary_03'} and grid['rows']['foot_z_plus15']['success']
    for name in ('composite_prefix_audit.json', 'controller_modifier_audit.json'):
        assert json.loads((a.candidate/name).read_text())['passed']
    sources = dict(m['sources'])
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if not name:
            continue
        file = Path(name).resolve()
        if file.suffix == '.py' and file.is_relative_to(root):
            relative = file.relative_to(root)
            if relative.parts[0] in ('tools', 'training'):
                sources[str(relative)] = sha(file)
    a.output.mkdir(exist_ok=False)
    for source, digest in sources.items():
        assert sha(root/source) == digest
        target = a.output/'source_snapshot'/source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root/source).read_bytes())
    shutil.copy2(a.candidate/'policy.zip', a.output/'policy.zip')
    shutil.copy2(m['teacher_checkpoint'], a.output/'teacher.zip')
    shutil.copy2(m['prefix_checkpoint'], a.output/'prefix_policy.zip')
    m.update(sources=sources, reference=gm['reference'], reference_sha256=gm['reference_sha256'], evaluation_cases=grid['cases'], training_cases={},
        parent_experiment=str(a.grid_parent.resolve()), parent_manifest_sha256=sha(a.grid_parent/'manifest.json'),
        candidate_experiment=str(a.candidate.resolve()), candidate_manifest_sha256=sha(a.candidate/'manifest.json'),
        teacher_checkpoint=str((a.output/'teacher.zip').resolve()),
        prefix_checkpoint=str((a.output/'prefix_policy.zip').resolve()),
        evaluation_only=True, new_training_steps=0,
        scope='Frozen 5D reference-feedback actor transferred to a new 5 mm foot-lift reference on two known development cases. '
              'Full time-zero evaluation with the original prefix. Different control policy from fixed-reference baseline; '
              'no independent generalization claim, no training, no automatic promotion.')
    (a.output/'manifest.json').write_text(json.dumps(m, indent=2))
    result = dict(checkpoint_sha256=sha(a.output/'policy.zip'), evaluation_pending=True,
                  evaluation_only=True, new_training_steps=0)
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    rows = evaluate_parallel(a.output, a.output/'parallel_after', workers=2)['trajectories']
    for row in rows:
        name = f"after_{row['case']}.npz"
        shutil.copy2(a.output/'parallel_after'/name, a.output/name)
    result.update(after=rows, evaluation_pending=False)
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    (a.output/'after.json').write_text(json.dumps(rows, indent=2))
    audit(a.output)
    audit_prefix(a.output)
    audit_fixed(a.output)
    checked = load_run(a.output)
    comparison = dict(successes=sum(r['success'] for r in checked['rows'].values()), cases=len(rows),
        lost=[case for case, row in checked['rows'].items() if grid['rows'][case]['success'] and not row['success']],
        gained=[case for case,row in checked['rows'].items() if not grid['rows'][case]['success'] and row['success']],
        audit_sha256=checked['audit_sha256'], baseline_audit_sha256=grid['audit_sha256'], automatic_promotion=False)
    (a.output/'regression.json').write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison), flush=True)


if __name__ == '__main__':
    main()
