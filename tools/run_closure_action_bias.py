"""From-zero five-case validation of one locally predicted action bias."""
import json
from pathlib import Path
import shutil
import numpy as np
from tools.continue_vector_full_cycle_training import sha
from tools.full_cycle_experiment_io import reuse_zero_evaluations
from tools.evaluate_closure_action_bias import evaluate_parallel
from tools.audit_closure_action_bias import audit
from tools.audit_closure_composite_prefix import audit as audit_prefix


def main():
    root=Path.cwd()
    parent=Path('outputs/closure_margin_control98304_v1')
    fit_path=Path('outputs/closure_action_response_fit_20261024_v1/summary.json')
    fit=json.loads(fit_path.read_text())
    assert fit['optimizer_success'] and fit['predicted_cost'] < fit['zero_cost']
    bias=np.asarray(fit['basis'])@np.asarray(fit['parameters'])
    assert abs(bias).max()<=1.000000001
    out=Path('outputs/closure_action_bias_fullcycle_20261024_v1');out.mkdir(exist_ok=False)
    manifest=json.loads((parent/'manifest.json').read_text())
    parent_audit=json.loads((parent/'independent_training_audit.json').read_text())
    assert parent_audit['passed'] and sha(parent/'policy.zip')==parent_audit['checkpoint_sha256']
    for rel,digest in manifest['sources'].items():
        assert sha(parent/'source_snapshot'/rel)==digest
        if rel.replace('\\','/').startswith('training/'):assert sha(root/rel)==digest
    shutil.copytree(parent/'source_snapshot',out/'source_snapshot')
    for name in ('training/closure_action_bias_policy.py','tools/evaluate_closure_action_bias.py','tools/audit_closure_action_bias.py','tools/run_closure_action_bias.py'):
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(name).read_bytes());manifest['sources'][name]=sha(name)
    manifest.update(parent_experiment=str(parent.resolve()),parent_checkpoint_sha256=sha(parent/'policy.zip'),
        suffix_start_seconds=99.66,horizon_seconds=113.66,closure_action_bias=bias.tolist(),
        evaluation_only=True,training_additional_steps=0,action_response_fit_sha256=sha(fit_path),
        scope='Known-case bounded action bias chosen by local finite differences; not PPO improvement or held-out validation. Full time-zero physics and original success gates.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    shutil.copy2(parent/'policy.zip',out/'policy.zip')
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
