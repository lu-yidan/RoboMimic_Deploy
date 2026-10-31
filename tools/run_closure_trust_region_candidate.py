"""From-zero five-case validation of one locally predicted action bias."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from tools.continue_vector_full_cycle_training import sha
from tools.full_cycle_experiment_io import reuse_zero_evaluations
from tools.evaluate_closure_geometry_bias import evaluate_parallel
from tools.audit_closure_geometry_bias import audit
from tools.audit_closure_composite_prefix import audit as audit_prefix


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--search',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    gate_path=Path('outputs/closure_geometry_bias_gate_20261024_v1/summary.json')
    gate=json.loads(gate_path.read_text())
    assert gate['passed'] and gate['policy_source_sha256']==sha('training/closure_geometry_bias_policy.py')
    root=Path.cwd()
    parent=Path('outputs/closure_margin_control98304_v1')
    fit_path=args.search/'summary.json'
    fit=json.loads(fit_path.read_text())
    assert fit['completed'] and fit['best']['complete'] and fit['best']['success'], 'Passing physical suffix required before full regression'
    bias=np.asarray(fit['basis'])@np.asarray(fit['best']['parameters'])
    assert abs(bias).max()<=1.000000001
    bias=np.clip(bias,-1.,1.)  # Canonicalize solver roundoff at declared bounds.
    out=args.output;out.mkdir(exist_ok=False)
    manifest=json.loads((parent/'manifest.json').read_text())
    parent_audit=json.loads((parent/'independent_training_audit.json').read_text())
    assert parent_audit['passed'] and sha(parent/'policy.zip')==parent_audit['checkpoint_sha256']
    for rel,digest in manifest['sources'].items():
        assert sha(parent/'source_snapshot'/rel)==digest
        if rel.replace('\\','/').startswith('training/'):assert sha(root/rel)==digest
    shutil.copytree(parent/'source_snapshot',out/'source_snapshot')
    for name in ('training/closure_geometry_bias_policy.py','tools/evaluate_closure_geometry_bias.py','tools/audit_closure_geometry_bias.py','tools/run_closure_trust_region_candidate.py'):
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(name).read_bytes());manifest['sources'][name]=sha(name)
    manifest.update(parent_experiment=str(parent.resolve()),parent_checkpoint_sha256=sha(parent/'policy.zip'),
        suffix_start_seconds=99.66,horizon_seconds=113.66,closure_action_bias=bias.tolist(),
        geometry_bias_activation=dict(lower_m=.0125,upper_m=.015,feature_index=138,scale_m=.02),
        evaluation_only=True,training_additional_steps=0,trust_region_search_sha256=sha(fit_path),action_interface_gate_sha256=sha(gate_path),selected_search_trial=fit['best']['trial'],
        scope='Known-case bounded action bias chosen by local search with actual physics acceptance; linear activation between known +12.5mm last passing height and +15mm failure, zero below and saturated above; existing oracle geometry only; not PPO improvement or held-out validation. Full time-zero physics and original success gates.')
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
