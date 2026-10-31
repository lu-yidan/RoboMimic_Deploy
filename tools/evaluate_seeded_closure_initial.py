"""From-zero five-case validation of one locally predicted action bias."""
import json
from pathlib import Path
import shutil
import numpy as np
from tools.continue_vector_full_cycle_training import sha
from tools.full_cycle_experiment_io import reuse_zero_evaluations
from tools.evaluate_closure_curriculum_policy import evaluate_parallel
from tools.audit_closure_curriculum_training import audit
from tools.audit_closure_composite_prefix import audit as audit_prefix


def main():
    root=Path.cwd()
    parent=Path('outputs/closure_margin_control98304_v1')
    seed=Path('outputs/closure_seed_policy_20261024_v1')
    gate=json.loads((seed/'summary.json').read_text())
    assert gate['passed'] and sha(seed/'policy_initial.zip')==gate['checkpoint_sha256']
    for name,digest in gate['sources'].items():assert sha(name)==digest
    out=Path('outputs/seeded_closure_initial_fullcycle_20261024_v1');out.mkdir(exist_ok=False)
    manifest=json.loads((parent/'manifest.json').read_text())
    parent_audit=json.loads((parent/'independent_training_audit.json').read_text())
    assert parent_audit['passed'] and sha(parent/'policy.zip')==parent_audit['checkpoint_sha256']
    for rel,digest in manifest['sources'].items():
        assert sha(parent/'source_snapshot'/rel)==digest
        if rel.replace('\\','/').startswith('training/'):assert sha(root/rel)==digest
    shutil.copytree(parent/'source_snapshot',out/'source_snapshot')
    for name in ('training/closure_seeded_policy.py','training/closure_exploration_policy.py','training/closure_composite_policy.py','tools/evaluate_closure_curriculum_policy.py','tools/audit_closure_curriculum_training.py','tools/evaluate_seeded_closure_initial.py'):
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(name).read_bytes());manifest['sources'][name]=sha(name)
    manifest.update(parent_experiment=str(parent.resolve()),parent_checkpoint_sha256=sha(parent/'policy.zip'),
        suffix_start_seconds=99.66,horizon_seconds=113.66,
        prefix_checkpoint=str((out/'prefix_policy.zip').resolve()),prefix_checkpoint_sha256=sha(parent/'policy.zip'),
        evaluation_only=True,training_additional_steps=0,seed_gate_sha256=sha(seed/'summary.json'),
        seed_bias=gate['seed_bias'],seed_ramp_seconds=gate['seed_ramp_seconds'],
        scope='Search-initialized mean before any PPO; existing geometry-gated action bias. Five known cases, full time-zero physics and original success gates. Not a learned improvement or held-out validation.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    shutil.copy2(parent/'policy.zip',out/'prefix_policy.zip')
    shutil.copy2(seed/'policy_initial.zip',out/'policy.zip')
    zero=reuse_zero_evaluations(parent,out,manifest,root)
    summary=dict(zero=zero,checkpoint_sha256=sha(out/'policy.zip'),evaluation_pending=True,additional_steps=0)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    after=evaluate_parallel(out,out/'parallel_after',None,workers=3)['trajectories']
    for row in after:
        name=f"after_{row['case']}.npz";shutil.copy2(out/'parallel_after'/name,out/name)
    (out/'after.json').write_text(json.dumps(after,indent=2))
    summary.update(after=after,evaluation_pending=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    audit(out);audit_prefix(out)
    print(json.dumps(dict(success=sum(r['success'] for r in after),cases=len(after))),flush=True)


if __name__=='__main__':main()
