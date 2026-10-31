"""Audit a calibrated action-bias policy on a complete declared geometry suite."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from tools.compare_audited_closure_candidates import load_run, sha
from tools.evaluate_closure_action_bias_v2 import evaluate_parallel
from tools.audit_action_bias_candidate_v2 import audit
from tools.audit_closure_flexible_prefix_v2 import audit as audit_prefix
from tools.audit_trained_closure_controller_v2 import audit as audit_fixed
from tools.audit_closure_foreaft_reference import audit as audit_reference


def main():
    p=argparse.ArgumentParser()
    for name in ['parent','search','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--label',default=None)
    a=p.parse_args();parent=load_run(a.parent)
    protocol=json.loads((a.search/'protocol.json').read_text())
    record=a.search/'summary.json'
    search=json.loads(record.read_text())
    assert protocol['kind']=='combined_action_reference'
    assert (search['completed'] or search.get('selected_group_complete')) and search['protocol_sha256']==sha(a.search/'protocol.json')
    origin=load_run(Path(protocol['parent']))
    assert origin['audit_sha256']==protocol['parent_audit_sha256']
    assert origin['checkpoint_sha256']==parent['checkpoint_sha256']==protocol['checkpoint_sha256']
    for key in ['reference_sha256','controller_modifier','adaptive_reference_command']:
        assert parent[key]==origin[key]
    for f,digest in protocol['sources'].items():assert sha(f)==digest
    choice=search['best'] if a.label is None else next(r for r in search['groups'] if r['label']==a.label)
    assert choice['label']!='base' and choice['neighbor_preserved'] and choice['successes']>1
    command=np.asarray(choice['reference_command'],dtype=float)
    assert command.shape==(5,) and np.all(abs(command)<=1.)
    tied=float(choice['tied_action_bias']);assert -.5<=tied<=.5
    bias=np.zeros(26);bias[3:5]=tied
    bias_spec=dict(normalized_bias=bias.tolist(),start_seconds=protocol['start_seconds'],ramp_seconds=3.,geometry_gate='existing_high_rod')
    m=json.loads((a.parent/'manifest.json').read_text());before=json.loads((a.parent/'summary.json').read_text())
    root=Path(__file__).resolve().parents[1]
    a.output.mkdir(exist_ok=False)
    selection=dict(search=search,source_record=str(record.resolve()),source_record_sha256=sha(record))
    (a.output/'selection_snapshot.json').write_text(json.dumps(selection,indent=2))
    shutil.copytree(a.parent/'source_snapshot',a.output/'source_snapshot')
    shutil.copy2(a.parent/'policy.zip',a.output/'policy.zip')
    shutil.copy2(m['prefix_checkpoint'],a.output/'prefix_policy.zip')
    sources=dict(m['sources'])
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if not file:continue
        file=Path(file).resolve()
        if file.suffix!='.py' or not file.is_relative_to(root):continue
        relative=file.relative_to(root)
        if relative.parts[0] not in ['tools','training']:continue
        target=a.output/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(file.read_bytes());sources[str(relative)]=sha(file)
    m.update(sources=sources,parent_experiment=str(a.parent.resolve()),parent_manifest_sha256=sha(a.parent/'manifest.json'),
        prefix_checkpoint=str((a.output/'prefix_policy.zip').resolve()),training_cases={},evaluation_only=True,new_training_steps=0,
        policy_adapter='closure-action-bias-v2',closure_action_bias=bias_spec,adaptive_reference_command=command.tolist(),requested_reference_command=command.tolist(),require_inactive_case_exact_parent=True,
        search_directory=str(a.search.resolve()),search_completed=search['completed'],search_label=choice['label'],
        selection_snapshot_sha256=sha(a.output/'selection_snapshot.json'),
        scope='Same actor checkpoint, joint/reference calibration with knee/ankle offsets inside unchanged 26D action bounds. '
              'Complete time-zero development cases, including provenance-checked searched trajectories. Not PPO improvement or automatic promotion.')
    (a.output/'manifest.json').write_text(json.dumps(m,indent=2))
    s=dict(evaluation_only=True,new_training_steps=0,evaluation_pending=True,checkpoint_sha256=sha(a.output/'policy.zip'),before=before['after'])
    (a.output/'summary.json').write_text(json.dumps(s,indent=2))
    selected={row['case']:row for row in search['rows'] if row['label']==choice['label']}
    results={};reused=[]
    for case in m['evaluation_cases']:
        if case not in selected:continue
        assert m['evaluation_cases'][case]==origin['cases'][case]
        row=selected[case];path=Path(row['file'])
        assert sha(path)==row['trajectory_sha256']
        info=json.loads((path.parent/'result.json').read_text())
        assert info['trajectory_sha256']==row['trajectory_sha256'] and info['case']==case
        assert info['closure_action_bias']==bias_spec and info['adaptive_reference_command']==command.tolist()
        shutil.copy2(path,a.output/f'after_{case}.npz');results[case]=info
        reused.append(dict(case=case,source=str(path),trajectory_sha256=sha(path)))
    remaining=[case for case in m['evaluation_cases'] if case not in results]
    if remaining:
        new=evaluate_parallel(a.output,a.output/'parallel_after',remaining,workers=4)['trajectories']
        for row in new:
            case=row['case'];results[case]=row
            shutil.copy2(a.output/'parallel_after'/f'after_{case}.npz',a.output/f'after_{case}.npz')
    rows=[results[case] for case in m['evaluation_cases']]
    s.update(after=rows,evaluation_pending=False,reused_time_zero_trajectories=reused)
    (a.output/'after.json').write_text(json.dumps(rows,indent=2));(a.output/'summary.json').write_text(json.dumps(s,indent=2))
    audit(a.output);audit_prefix(a.output);audit_fixed(a.output);audit_reference(a.output)
    verified=load_run(a.output)
    prefix=[]
    for case in m['evaluation_cases']:
        checks={}
        with np.load(a.parent/f'after_{case}.npz') as x,np.load(a.output/f'after_{case}.npz') as y:
            for key in ['qpos','qvel','time_s','observations','actions','torques','rewards']:
                n=round(protocol['start_seconds']/.02)+(1 if key in ['qpos','qvel','time_s'] else 0)
                checks[key]=bool(np.array_equal(x[key][:n],y[key][:n]))
        prefix.append(dict(case=case,passed=all(checks.values()),checks=checks))
    assert all(r['passed'] for r in prefix)
    result=dict(successes=sum(r['success'] for r in verified['rows'].values()),cases=len(rows),
        gained=[c for c,r in verified['rows'].items() if r['success'] and not parent['rows'][c]['success']],
        lost=[c for c,r in verified['rows'].items() if not r['success'] and parent['rows'][c]['success']],
        frozen_prefix=prefix,prefix_end_seconds=protocol['start_seconds'],automatic_promotion=False,source_sha256=sha(__file__))
    (a.output/'regression.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='frozen_prefix'}),flush=True)


if __name__=='__main__':main()
