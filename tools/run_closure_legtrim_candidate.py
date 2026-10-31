"""Run a searched constant-reference candidate from time zero on all eight cases."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from tools.compare_audited_closure_candidates import load_run,sha
from tools.evaluate_closure_legtrim_reference import evaluate_parallel
from tools.audit_fullcycle_candidate_only import audit
from tools.audit_closure_candidate_prefix import audit as audit_prefix
from tools.audit_trained_closure_controller import audit as audit_fixed
from tools.audit_closure_legtrim_reference import audit as audit_added


def main():
    parser=argparse.ArgumentParser()
    for name in ('parent','search','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    verified=load_run(args.parent)
    protocol=json.loads((args.search/'protocol.json').read_text());result=json.loads((args.search/'summary.json').read_text())
    assert result['completed'] and result['protocol_sha256']==sha(args.search/'protocol.json')
    assert sha(args.search/'search_source.py')==protocol['source_sha256']
    assert Path(protocol['parent']).resolve()==args.parent.resolve() and protocol['parent_audit_sha256']==verified['audit_sha256']
    best=result['best'];assert best['highest_preserved'] and best['successes']>result['rows'][0]['successes']
    command=np.asarray(best['command'],dtype=float);assert command.shape==(2,) and np.isfinite(command).all() and abs(command).max()<=1.+1e-12
    manifest=json.loads((args.parent/'manifest.json').read_text());before=json.loads((args.parent/'summary.json').read_text())
    args.output.mkdir(exist_ok=False);shutil.copytree(args.parent/'source_snapshot',args.output/'source_snapshot')
    for name,source in [('policy.zip',args.parent/'policy.zip'),('prefix_policy.zip',Path(manifest['prefix_checkpoint']))]:shutil.copy2(source,args.output/name)
    sources=dict(manifest['sources'])
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if not file:continue
        file=Path(file).resolve()
        if file.suffix!='.py' or not file.is_relative_to(root):continue
        relative=file.relative_to(root)
        if relative.parts[0] not in ('training','tools'):continue
        target=args.output/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(file.read_bytes());sources[str(relative)]=sha(file)
    manifest.update(sources=sources,parent_experiment=str(args.parent.resolve()),parent_manifest_sha256=sha(args.parent/'manifest.json'),parent_checkpoint_sha256=sha(args.parent/'policy.zip'),
        prefix_checkpoint=str((args.output/'prefix_policy.zip').resolve()),training_cases={},evaluation_only=True,new_training_steps=0,
        adaptive_reference_command=np.clip(command,-1,1).tolist(),requested_reference_command=command.tolist(),adaptive_reference_scales=[.003,.003],
        require_inactive_case_exact_parent=True,search_directory=str(args.search.resolve()),search_summary_sha256=sha(args.search/'summary.json'),
        scope='Unchanged composite actor plus two common constant knee/ankle reference commands selected on four difficult development cases. All eight evaluated from time zero. No new PPO learning or independent generalization claim.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    summary=dict(checkpoint_sha256=sha(args.output/'policy.zip'),evaluation_only=True,new_training_steps=0,evaluation_pending=True,before=before['after'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    rows=evaluate_parallel(args.output,args.output/'parallel_after',None,workers=2)['trajectories']
    for row in rows:
        name=f"after_{row['case']}.npz";shutil.copy2(args.output/'parallel_after'/name,args.output/name)
    summary.update(after=rows,evaluation_pending=False)
    (args.output/'after.json').write_text(json.dumps(rows,indent=2));(args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    audit(args.output);audit_prefix(args.output);audit_fixed(args.output);audit_added(args.output)
    print(json.dumps(dict(completed=True,successes=sum(r['success'] for r in rows),cases=len(rows),new_training_steps=0)),flush=True)


if __name__=='__main__':main()
