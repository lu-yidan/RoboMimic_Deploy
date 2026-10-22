"""Evaluate an existing audited policy on the fixed high-rod reference, no learning."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from stable_baselines3 import PPO
from tools.continue_vector_full_cycle_training import sha
from tools.evaluate_bar_geometry_policy import evaluate_parallel


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    parent=args.parent;root=Path(__file__).resolve().parents[1]
    m=json.loads((parent/'manifest.json').read_text());a=json.loads((parent/'independent_training_audit.json').read_text())
    if not a['passed'] or a['checkpoint_sha256']!=sha(parent/'policy.zip') or a['experiment_manifest_sha256']!=sha(parent/'manifest.json'):raise ValueError('Audited policy required')
    for path,digest in m['sources'].items():
        if path.replace('\\','/').startswith('training/') and sha(path)!=digest:raise ValueError('Environment changed')
    zero_root=Path('outputs/highrod_height_mismatch_v1')
    comparison=json.loads((zero_root/'height_comparison.json').read_text())
    if comparison['comparison_sha256']!=sha(zero_root/'independent_comparison.json'):raise ValueError('Zero evaluation evidence changed')
    cases={};zero=[];provenance=[]
    args.output.mkdir(exist_ok=False)
    for r in json.loads((zero_root/'results.json').read_text()):
        p=zero_root/r['label'];zm=json.loads((p/'manifest.json').read_text());zs=json.loads((p/'summary.json').read_text());za=json.loads((p/'independent_audit.json').read_text())
        assert za['integrity_passed'] and za['summary_sha256']==sha(p/'summary.json') and za['trajectory_sha256']==sha(p/'trajectory.npz')
        assert zm['normal_bias_y_deg']==m['normal_bias_y_deg'] and zm['memory_enabled']==m['force_history_memory'] and zm['bundle_manifest_sha256']==m['bundle_manifest_sha256']
        cases[zs['case']]=zm['offsets'];zero.append(zs)
        shutil.copy2(p/'trajectory.npz',args.output/f"zero_{zs['case']}.npz")
        provenance.append(dict(case=zs['case'],source=str(p),audit_sha256=sha(p/'independent_audit.json')))
    plan=Path('outputs/highrod_endpoint_adapt_plan_v1/plan.npz')
    # Load before snapshotting: custom feature classes are imported by loading.
    policy_probe=PPO.load(parent/'policy.zip',device='cpu')
    del policy_probe
    sources={}
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if not file:continue
        file=Path(file).resolve()
        if file.suffix!='.py' or not file.is_relative_to(root):continue
        relative=file.relative_to(root)
        if relative.parts[0] not in ('training','tools'):continue
        target=args.output/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(file.read_bytes());sources[str(relative)]=sha(target)
    m.update(reference=str(plan.resolve()),reference_sha256=sha(plan),evaluation_cases=cases,training_cases={},sources=sources,
        evaluation_only=True,policy_source=str(parent.resolve()),source_policy_sha256=sha(parent/'policy.zip'),
        zero_provenance=provenance,scope='No additional training. Existing policy transfer to a different fixed high-rod reference and five actual heights. Separate evaluation, not original-task improvement.')
    (args.output/'manifest.json').write_text(json.dumps(m,indent=2));(args.output/'zero.json').write_text(json.dumps(zero,indent=2))
    shutil.copy2(parent/'policy.zip',args.output/'policy.zip')
    summary=dict(evaluation_only=True,new_training_steps=0,checkpoint_sha256=sha(args.output/'policy.zip'),zero=zero,evaluation_pending=True)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    results=evaluate_parallel(args.output,args.output/'parallel_after',None,workers=3)['trajectories']
    for r in results:
        filename=f"after_{r['case']}.npz";shutil.copy2(args.output/'parallel_after'/filename,args.output/filename)
    summary.update(after=results,evaluation_pending=False)
    (args.output/'after.json').write_text(json.dumps(results,indent=2));(args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps([dict(case=r['case'],success=r['success'],time_s=r['time_s'],reason=r['reason']) for r in results]),flush=True)


if __name__=='__main__':main()
