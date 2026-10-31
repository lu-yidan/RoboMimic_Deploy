"""Validate a fixed actor on eight cases, reusing its already audited five runs."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from tools.continue_vector_full_cycle_training import sha
from tools.evaluate_closure_waist_body_recycled import evaluate_parallel
from tools.audit_fullcycle_candidate_only import audit
from tools.audit_trained_closure_controller import audit as audit_controller


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=json.loads((a.parent/'manifest.json').read_text());s=json.loads((a.parent/'summary.json').read_text());verified=json.loads((a.parent/'independent_training_audit.json').read_text())
    assert verified['passed'] and verified['experiment_manifest_sha256']==sha(a.parent/'manifest.json') and verified['experiment_summary_sha256']==sha(a.parent/'summary.json') and verified['checkpoint_sha256']==sha(a.parent/'policy.zip')
    assert len(s['after'])==5 and all(r['independently_accepted'] for r in verified['trajectories'] if r['label']=='after')
    cp=json.loads((a.cache/'protocol.json').read_text());cs=json.loads((a.cache/'summary.json').read_text());assert cs['passed'] and cs['protocol_sha256']==sha(a.cache/'protocol.json')
    assert cp['parent_checkpoint_sha256']==m['prefix_checkpoint_sha256'] and cp['reference_sha256']==m['reference_sha256']
    old_cases=m['evaluation_cases'];assert set(old_cases)<set(cp['cases']) and all(old_cases[k]==cp['cases'][k] for k in old_cases)
    a.output.mkdir(exist_ok=False);shutil.copytree(a.parent/'source_snapshot',a.output/'source_snapshot')
    m['sources']=dict(m['sources'])
    for name in ('tools/prepare_trained_closure_eightcase_evaluation.py','tools/evaluate_closure_waist_body_recycled.py','tools/audit_fullcycle_candidate_only.py','tools/audit_trained_closure_controller.py'):
        target=a.output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes());m['sources'][name]=sha(name)
    for name,source in [('policy.zip',a.parent/'policy.zip'),('prefix_policy.zip',Path(m['prefix_checkpoint']))]:shutil.copy2(source,a.output/name)
    m.update(evaluation_cases=cp['cases'],training_cases={},evaluation_only=True,parent_experiment=str(a.parent.resolve()),parent_checkpoint_sha256=sha(a.parent/'policy.zip'),parent_training_cases=m.get('training_cases',{}),parent_training_steps=s.get('additional_steps',0),parent_manifest_sha256=sha(a.parent/'manifest.json'),new_training_steps=0,prefix_checkpoint=str((a.output/'prefix_policy.zip').resolve()),
        prefix_cache=str(a.cache.resolve()),prefix_cache_summary_sha256=sha(a.cache/'summary.json'),
        scope='Five-case-trained actor evaluated without updates on eight development cases. Five audited complete runs reused with hashes; three extra X/Z cases run from zero. No learning or held-out claim.')
    (a.output/'manifest.json').write_text(json.dumps(m,indent=2))
    summary=dict(checkpoint_sha256=sha(a.output/'policy.zip'),evaluation_pending=True,evaluation_only=True,new_training_steps=0)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    reused={r['case']:r for r in s['after']}
    for case,row in reused.items():
        file=f'after_{case}.npz';assert sha(a.parent/file)==row['trajectory_sha256'];shutil.copy2(a.parent/file,a.output/file)
    extra=[case for case in cp['cases'] if case not in reused]
    results=evaluate_parallel(a.output,a.output/'parallel_extra',extra,workers=2)['trajectories']
    for row in results:
        file=f"after_{row['case']}.npz";shutil.copy2(a.output/'parallel_extra'/file,a.output/file);reused[row['case']]=row
    rows=[reused[case] for case in cp['cases']];summary.update(after=rows,evaluation_pending=False)
    (a.output/'after.json').write_text(json.dumps(rows,indent=2));(a.output/'summary.json').write_text(json.dumps(summary,indent=2));audit(a.output);audit_controller(a.output)
    prefix=[]
    for cache_row in cs['rows']:
        case=cache_row['case'];source_cache=Path(cache_row['source_cache_directory']);source_parent=Path(json.loads((source_cache/'protocol.json').read_text())['parent'])
        path=source_parent/f'after_{case}.npz';assert sha(path)==cache_row['source_trajectory_sha256']
        with np.load(path) as x,np.load(a.output/f'after_{case}.npz') as y:
            checks={}
            for k in ('qpos','qvel','time_s','observations','actions','torques','rewards'):
                count=4984 if k in ('qpos','qvel','time_s') else 4983
                checks[k]=bool(np.array_equal(x[k][:count],y[k][:count]))
        prefix.append(dict(case=case,passed=all(checks.values()),checks=checks,source_trajectory_sha256=sha(path)))
    report=dict(passed=all(r['passed'] for r in prefix),rows=prefix,manifest_sha256=sha(a.output/'manifest.json'),cache_summary_sha256=sha(a.cache/'summary.json'),source_sha256=sha(__file__))
    (a.output/'composite_prefix_audit.json').write_text(json.dumps(report,indent=2));assert report['passed']
    print(json.dumps(dict(completed=True,successes=sum(r['success'] for r in rows),cases=len(rows))),flush=True)

if __name__=='__main__':main()
