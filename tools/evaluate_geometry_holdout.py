"""Evaluate a final geometry-curriculum checkpoint on the frozen local holdout."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from tools.compare_audited_closure_candidates import load_run, sha
from tools.evaluate_closure_action_bias_v2 import evaluate_parallel
from tools.audit_action_bias_candidate_v2 import audit
from tools.audit_trained_closure_controller_v2 import audit as audit_controller
from tools.audit_closure_foreaft_reference import audit as audit_reference


def main():
    parser = argparse.ArgumentParser()
    for name in ('candidate', 'geometry-protocol', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 4), default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    candidate = load_run(args.candidate)
    protocol = json.loads((args.geometry_protocol/'protocol.json').read_text())
    manifest = json.loads((args.candidate/'manifest.json').read_text())
    summary = json.loads((args.candidate/'summary.json').read_text())
    update = json.loads((args.candidate/'parameter_update_audit.json').read_text())
    matched = json.loads((args.candidate/'matched_protocol.json').read_text())
    assert not summary['evaluation_pending']
    assert summary['additional_steps'] == protocol['planned_additional_steps'] == 131072
    assert update['passed'] and update['after_sha256'] == candidate['checkpoint_sha256']
    assert matched['geometry_protocol_sha256'] == sha(args.geometry_protocol/'protocol.json')
    assert candidate['cases'] == protocol['training_cases']
    assert manifest['training_cases'] == protocol['training_cases']
    assert manifest['seed'] == protocol['planned_training_seed']
    assert manifest['boundary_reward_mode'] == protocol['reward_mode'] == 'average'
    assert manifest['policy_adapter'] == 'closure-action-bias-v2'
    assert manifest['require_inactive_case_exact_parent'] is False
    cases = protocol['validation_cases']
    assert len(cases) == 16 and len(candidate['cases']) == 24
    serialize = lambda value: json.dumps(value, sort_keys=True)
    assert not {serialize(v) for v in cases.values()} & {serialize(v) for v in candidate['cases'].values()}
    for name in ('controller_modifier_audit_v2.json', 'adaptive_reference_audit.json', 'composite_prefix_audit_v2.json'):
        report = json.loads((args.candidate/name).read_text())
        assert report['passed']
    args.output.mkdir(exist_ok=False)
    shutil.copytree(args.candidate/'source_snapshot', args.output/'source_snapshot')
    shutil.copy2(args.candidate/'policy.zip', args.output/'policy.zip')
    shutil.copy2(manifest['prefix_checkpoint'], args.output/'prefix_policy.zip')
    sources = dict(manifest['sources'])
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if not file:
            continue
        file = Path(file).resolve()
        if file.suffix != '.py' or not file.is_relative_to(root):
            continue
        relative = file.relative_to(root)
        if relative.parts[0] not in ('training', 'tools'):
            continue
        # Runtime must match the completed training snapshot.
        if relative.parts[0] == 'training':
            assert sources[str(relative)] == sha(file)
        target = args.output/'source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.read_bytes())
        sources[str(relative)] = sha(file)
    manifest.update(sources=sources, evaluation_cases=cases, original_training_cases=manifest['training_cases'],
        training_cases={}, parent_experiment=str(args.candidate.resolve()),
        parent_manifest_sha256=sha(args.candidate/'manifest.json'),
        candidate_experiment=str(args.candidate.resolve()),
        candidate_checkpoint_sha256=candidate['checkpoint_sha256'],
        prefix_checkpoint=str((args.output/'prefix_policy.zip').resolve()),
        geometry_protocol_sha256=sha(args.geometry_protocol/'protocol.json'),
        evaluation_only=True, new_training_steps=0,
        scope='Final-checkpoint evaluation on sixteen prospectively reserved local geometries. Frozen checkpoint hash is recorded before simulation. Full time-zero rollouts retain original failure termination. Independent state/action/controller/reference audits, but no same-geometry parent-prefix equality test because no parent holdout trajectories exist. Not broad scene generalization. Do not select intermediate checkpoints or tune on these outcomes; no automatic promotion.')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    result = dict(checkpoint_sha256=candidate['checkpoint_sha256'], evaluation_only=True,
        new_training_steps=0, evaluation_pending=True)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2))
    rows = evaluate_parallel(args.output, args.output/'parallel_after', None, workers=args.workers)['trajectories']
    for row in rows:
        name = f"after_{row['case']}.npz"
        shutil.copy2(args.output/'parallel_after'/name, args.output/name)
    result.update(after=rows, evaluation_pending=False)
    (args.output/'after.json').write_text(json.dumps(rows, indent=2))
    (args.output/'summary.json').write_text(json.dumps(result, indent=2))
    audit(args.output)
    audit_controller(args.output)
    audit_reference(args.output)
    verified = load_run(args.output)
    final = dict(completed=True, successes=sum(r['success'] for r in verified['rows'].values()),
        cases=len(rows), checkpoint_sha256=candidate['checkpoint_sha256'],
        audit_sha256=verified['audit_sha256'], automatic_promotion=False)
    (args.output/'holdout_result.json').write_text(json.dumps(final, indent=2))
    print(json.dumps(final), flush=True)


if __name__ == '__main__':
    main()
