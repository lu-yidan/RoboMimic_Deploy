"""From-zero five-case validation of one locally predicted action bias."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from tools.continue_vector_full_cycle_training import sha
from tools.evaluate_closure_waist_body import evaluate_parallel
from tools.audit_closure_waist_body import audit as audit_modifier
from tools.audit_closure_curriculum_training import audit
from tools.audit_closure_composite_prefix import audit as audit_prefix


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=Path,required=True)
    parser.add_argument('--search',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    search=json.loads((args.search/'summary.json').read_text())
    protocol=json.loads((args.search/'protocol.json').read_text())
    assert sha(args.search/'protocol.json')==search['protocol_sha256']
    for name,digest in protocol['source_sha256'].items():assert sha(name)==digest
    candidate=search['best']
    assert search['completed'] and candidate['success'], 'Suffix candidate must pass original gates first'
    trajectory=args.search/f"trial_{candidate['trial']:03d}.npz"
    assert sha(trajectory)==candidate['trajectory_sha256']
    root=Path.cwd()
    parent=Path('outputs/closure_margin_control98304_v1')
    seed=args.seed
    gate=json.loads((seed/'summary.json').read_text())
    assert gate['passed'] and sha(seed/'policy_initial.zip')==gate['checkpoint_sha256']
    assert gate['search_summary_sha256']==sha(args.search/'summary.json')
    for name,digest in gate['sources'].items():assert sha(name)==digest
    out=args.output;out.mkdir(exist_ok=False)
    manifest=json.loads((parent/'manifest.json').read_text())
    parent_audit=json.loads((parent/'independent_training_audit.json').read_text())
    assert parent_audit['passed'] and sha(parent/'policy.zip')==parent_audit['checkpoint_sha256']
    for rel,digest in manifest['sources'].items():
        assert sha(parent/'source_snapshot'/rel)==digest
        if rel.replace('\\','/').startswith('training/'):assert sha(root/rel)==digest
    shutil.copytree(parent/'source_snapshot',out/'source_snapshot')
    for name in ('training/closure_seeded_policy.py','training/closure_exploration_policy.py','training/closure_composite_policy.py','tools/evaluate_closure_curriculum_policy.py','tools/audit_closure_curriculum_training.py','tools/evaluate_closure_waist_body.py','tools/run_closure_combined_margin.py','tools/audit_closure_waist_body.py','training/closure_waist_target.py','training/closure_body_target.py'):
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(name).read_bytes());manifest['sources'][name]=sha(name)
    manifest.update(parent_experiment=str(parent.resolve()),parent_checkpoint_sha256=sha(parent/'policy.zip'),
        suffix_start_seconds=99.66,horizon_seconds=113.66,
        prefix_checkpoint=str((out/'prefix_policy.zip').resolve()),prefix_checkpoint_sha256=sha(parent/'policy.zip'),
        evaluation_only=True,training_additional_steps=0,seed_gate_sha256=sha(seed/'summary.json'),
        seed_bias=gate['seed_bias'],seed_ramp_seconds=gate['seed_ramp_seconds'],
        controller_modifier=dict(waist_bias=candidate['waist_bias'],body_drop_m=candidate['body_drop_m'],waist_ramp_seconds=1.,geometry_gain='clip((raw_obs[138]-.625)/.125,0,1)'),
        selection_summary_sha256=sha(args.search/'summary.json'),
        zero_baseline_scope='Original unmodified controller with zero actor. Historical comparison only; NOT zero actor under the additional waist/body controller.',
        scope='Search-initialized actor plus explicit waist and body reference authority. Full time-zero five known cases, original gates. No PPO training gain and no held-out validation.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    shutil.copy2(parent/'policy.zip',out/'prefix_policy.zip')
    shutil.copy2(seed/'policy_initial.zip',out/'policy.zip')
    zero=json.loads((parent/'zero.json').read_text())
    for row in zero:
        name=f"zero_{row['case']}.npz"
        assert sha(parent/name)==row['trajectory_sha256']
        shutil.copy2(parent/name,out/name)
    shutil.copy2(parent/'zero.json',out/'zero.json')
    (out/'zero_reuse_provenance.json').write_text(json.dumps(dict(source=str(parent.resolve()),
        source_manifest_sha256=sha(parent/'manifest.json'),scope=manifest['zero_baseline_scope'],
        files={f"zero_{r['case']}.npz":r['trajectory_sha256'] for r in zero}),indent=2))
    summary=dict(zero=zero,checkpoint_sha256=sha(out/'policy.zip'),evaluation_pending=True,additional_steps=0)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    after=evaluate_parallel(out,out/'parallel_after',None,workers=3)['trajectories']
    for row in after:
        name=f"after_{row['case']}.npz";shutil.copy2(out/'parallel_after'/name,out/name)
    (out/'after.json').write_text(json.dumps(after,indent=2))
    summary.update(after=after,evaluation_pending=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    audit(out);audit_prefix(out);audit_modifier(out)
    print(json.dumps(dict(success=sum(r['success'] for r in after),cases=len(after))),flush=True)


if __name__=='__main__':main()
