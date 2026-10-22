"""Transfer an audited148-input policy to an audited standard reference suite."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import torch
from stable_baselines3 import PPO
from tools.continue_vector_full_cycle_training import sha
from tools.evaluate_bar_geometry_policy import evaluate_parallel


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--zero-suite',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    parent=json.loads((args.parent/'manifest.json').read_text());audit=json.loads((args.parent/'independent_training_audit.json').read_text())
    assert audit['passed'] and audit['checkpoint_sha256']==sha(args.parent/'policy.zip') and audit['experiment_manifest_sha256']==sha(args.parent/'manifest.json') and audit['experiment_summary_sha256']==sha(args.parent/'summary.json')
    for path,digest in parent['sources'].items():
        if path.replace('\\','/').startswith('training/'):assert sha(path)==digest and sha(args.parent/'source_snapshot'/path)==digest
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
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2));(args.output/'zero.json').write_text(json.dumps(zero,indent=2))
    shutil.copy2(args.parent/'policy.zip',args.output/'policy.zip')
    summary=dict(evaluation_only=True,new_training_steps=0,checkpoint_sha256=sha(args.output/'policy.zip'),zero=zero,evaluation_pending=True)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    results=evaluate_parallel(args.output,args.output/'parallel_after',None,workers=2)['trajectories']
    for r in results:
        name=f"after_{r['case']}.npz";shutil.copy2(args.output/'parallel_after'/name,args.output/name)
    summary.update(after=results,evaluation_pending=False)
    (args.output/'after.json').write_text(json.dumps(results,indent=2));(args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps([dict(case=r['case'],success=r['success'],reason=r['reason']) for r in results]))


if __name__=='__main__':main()
