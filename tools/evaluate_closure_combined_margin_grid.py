"""Transfer an audited148-input policy to an audited standard reference suite."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import torch
from stable_baselines3 import PPO
from tools.continue_vector_full_cycle_training import sha
from tools.evaluate_closure_action_bias import evaluate_parallel as evaluate_bias
from tools.evaluate_closure_geometry_bias import evaluate_parallel as evaluate_geometry_bias
from tools.evaluate_closure_curriculum_policy import evaluate_parallel as evaluate_composite
from tools.evaluate_closure_waist_body import evaluate_parallel as evaluate_modified


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--zero-suite',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    parent=json.loads((args.parent/'manifest.json').read_text());audit=json.loads((args.parent/'independent_training_audit.json').read_text())
    assert audit['passed'] and audit['checkpoint_sha256']==sha(args.parent/'policy.zip') and audit['experiment_manifest_sha256']==sha(args.parent/'manifest.json') and audit['experiment_summary_sha256']==sha(args.parent/'summary.json')
    for path,digest in parent['sources'].items():
        if path.replace('\\','/').startswith('training/'):assert sha(path)==digest and sha(args.parent/'source_snapshot'/path)==digest
    successes={r['case'] for r in audit['trajectories'] if r['label']=='after' and r['independently_accepted']}
    if successes!=set(parent['evaluation_cases']) or len(successes)!=5:raise ValueError('Require all five known cases before this grid stage')
    assert 'controller_modifier' in parent
    modifier_audit=json.loads((args.parent/'controller_modifier_audit.json').read_text())
    assert modifier_audit['passed'] and modifier_audit['manifest_sha256']==sha(args.parent/'manifest.json') and modifier_audit['summary_sha256']==sha(args.parent/'summary.json')
    adapter='waist_body' if 'controller_modifier' in parent else 'geometry_bias' if 'geometry_bias_activation' in parent else 'bias' if 'closure_action_bias' in parent else 'composite'
    evaluate_parallel=evaluate_modified if adapter=='waist_body' else evaluate_geometry_bias if adapter=='geometry_bias' else evaluate_bias if adapter=='bias' else evaluate_composite
    comparison=json.loads((args.zero_suite/'independent_comparison.json').read_text())
    protocol=json.loads((args.zero_suite/'protocol.json').read_text())
    assert len(comparison['results'])==len(protocol['jobs']) and {r['label'] for r in comparison['results']}=={j[1] for j in protocol['jobs']}
    cases={};zero=[];provenance=[];references=set()
    args.output.mkdir(exist_ok=False)
    for row in comparison['results']:
        p=args.zero_suite/row['label'];m=json.loads((p/'manifest.json').read_text());s=json.loads((p/'summary.json').read_text());a=json.loads((p/'independent_audit.json').read_text())
        assert row['audit_sha256']==sha(p/'independent_audit.json') and a['integrity_passed'] and a['summary_sha256']==sha(p/'summary.json') and a['trajectory_sha256']==sha(p/'trajectory.npz')
        assert s['manifest_sha256']==sha(p/'manifest.json') and sha(m['reference'])==m['reference_sha256']
        assert not any(key in m for key in ('controller_modifier','reference_modifier','action_rule'))
        assert m['normal_bias_y_deg']==parent['normal_bias_y_deg'] and m['memory_enabled']==parent['force_history_memory'] and m['bundle_manifest_sha256']==parent['bundle_manifest_sha256']
        assert m['case'] not in cases
        cases[m['case']]=m['offsets'];references.add((m['reference'],m['reference_sha256']));zero.append(s)
        shutil.copy2(p/'trajectory.npz',args.output/f"zero_{m['case']}.npz")
        provenance.append(dict(case=m['case'],source=str(p),audit_sha256=sha(p/'independent_audit.json')))
    assert len(references)==1
    reference,reference_hash=references.pop()
    torch.set_num_threads(1);policy=PPO.load(args.parent/'policy.zip',device='cpu');parent_counter=policy.num_timesteps
    assert policy.observation_space.shape==(148,) and policy.action_space.shape==(26,)
    sources={}
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if not path:continue
        path=Path(path).resolve()
        if path.suffix!='.py' or not path.is_relative_to(root):continue
        relative=path.relative_to(root)
        if relative.parts[0] not in ('training','tools'):continue
        dest=args.output/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());sources[str(relative)]=sha(dest)
    manifest={k:parent[k] for k in ('contract','observation_contract','geometry_mode','bundle_manifest_sha256','normal_bias_y_deg','force_history_memory')}
    manifest.update(reference=reference,reference_sha256=reference_hash,evaluation_cases=cases,training_cases={},sources=sources,
        evaluation_only=True,policy_source=str(args.parent.resolve()),source_policy_sha256=sha(args.parent/'policy.zip'),
        parent_checkpoint_counter=parent_counter,new_training_steps=0,zero_provenance=provenance,
        zero_comparison_sha256=sha(args.zero_suite/'independent_comparison.json'),
        scope='No additional training. Existing148-input actor on a new audited reference, retaining exact task gates. Known-case transfer only.')
    for key in ('suffix_start_seconds','horizon_seconds','closure_action_bias','geometry_bias_activation','prefix_checkpoint','prefix_checkpoint_sha256','controller_modifier','zero_baseline_scope'):
        if key in parent:manifest[key]=parent[key]
    manifest['parent_experiment']=str(Path('outputs/closure_control_policy_validation_v1').resolve())
    manifest['policy_adapter']=adapter
    manifest['scope']='Existing twelve-case validation/selection grid, no tuning or training on this grid run. Five known cases already passed. Original task gates; not an independent test of the selected candidate.'
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2));(args.output/'zero.json').write_text(json.dumps(zero,indent=2))
    shutil.copy2(args.parent/'policy.zip',args.output/'policy.zip')
    summary=dict(evaluation_only=True,new_training_steps=0,checkpoint_sha256=sha(args.output/'policy.zip'),zero=zero,evaluation_pending=True)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    results=evaluate_parallel(args.output,args.output/'parallel_after',None,workers=2)['trajectories']
    for r in results:
        name=f"after_{r['case']}.npz";shutil.copy2(args.output/'parallel_after'/name,args.output/name)
    summary.update(after=results,evaluation_pending=False)
    (args.output/'after.json').write_text(json.dumps(results,indent=2));(args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    if adapter=='geometry_bias':
        from tools.audit_closure_geometry_bias import audit as audit_result
    elif adapter=='bias':
        from tools.audit_closure_action_bias import audit as audit_result
    else:
        from tools.audit_closure_curriculum_training import audit as audit_result
    audit_result(args.output)
    from tools.audit_closure_composite_prefix import audit as audit_prefix
    audit_prefix(args.output)
    from tools.audit_closure_waist_body import audit as audit_modifier
    audit_modifier(args.output)
    old=json.loads(Path('outputs/closure_waist_body_grid_20261024_v1/independent_training_audit.json').read_text())
    assert old['passed'] and old['experiment_manifest_sha256']==sha('outputs/closure_waist_body_grid_20261024_v1/manifest.json') and old['experiment_summary_sha256']==sha('outputs/closure_waist_body_grid_20261024_v1/summary.json')
    old_success={r['case'] for r in old['trajectories'] if r['label']=='after' and r['independently_accepted']}
    new_success={r['case'] for r in results if r['success']}
    decision=dict(comparison_audit_sha256=sha('outputs/closure_waist_body_grid_20261024_v1/independent_training_audit.json'),comparison='First waist/body candidate, twelve prior successes',eligible_for_release_review=not bool(old_success-new_success),lost_grid_successes=sorted(old_success-new_success),new_grid_successes=sorted(new_success-old_success),grid_successes=len(new_success),known_successes=5,scope='Eligibility only; portable export, replay and release bookkeeping still required. No automatic replacement.')
    (args.output/'selection.json').write_text(json.dumps(decision,indent=2))
    print(json.dumps(decision))


if __name__=='__main__':main()
