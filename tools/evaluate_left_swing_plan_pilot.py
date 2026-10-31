"""Strict time-zero pilot of a changed reference with unchanged frozen policies."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from tools.compare_audited_closure_candidates import load_run,sha
from tools.evaluate_closure_foreaft_reference import evaluate_parallel
from tools.audit_fullcycle_candidate_only import audit
from tools.audit_trained_closure_controller_v2 import audit as audit_fixed
from tools.audit_closure_foreaft_reference import audit as audit_added


def main():
    p=argparse.ArgumentParser()
    for key in ('parent','failure-parent','plan-run','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    parents={'foot_z_plus15':a.parent,'fresh_boundary_03':a.failure_parent}
    verified={k:load_run(v) for k,v in parents.items()}
    m=json.loads((a.parent/'manifest.json').read_text());fm=json.loads((a.failure_parent/'manifest.json').read_text())
    assert all(verified['foot_z_plus15'][k]==verified['fresh_boundary_03'][k] for k in ('checkpoint_sha256','reference_sha256','controller_modifier','adaptive_reference_command'))
    geometry=json.loads((a.plan_run/'differential_geometry_audit.json').read_text())
    assert geometry['passed'] and geometry['manifest_sha256']==sha(a.plan_run/'manifest.json')
    plan=json.loads((a.plan_run/'manifest.json').read_text())
    assert plan['parent_sha256']==m['reference_sha256'] and plan['plan_sha256']==sha(a.plan_run/'plan.npz')
    assert json.loads((a.plan_run/'audit.json').read_text())['passed']
    a.output.mkdir(exist_ok=False);shutil.copytree(a.parent/'source_snapshot',a.output/'source_snapshot')
    shutil.copy2(a.parent/'policy.zip',a.output/'policy.zip');shutil.copy2(m['prefix_checkpoint'],a.output/'prefix_policy.zip')
    sources=dict(m['sources'])
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if not file:continue
        file=Path(file).resolve()
        if file.suffix!='.py' or not file.is_relative_to(root):continue
        relative=file.relative_to(root)
        if relative.parts[0] not in ('tools','training'):continue
        dest=a.output/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(file.read_bytes());sources[str(relative)]=sha(file)
    m.update(sources=sources,reference=str((a.plan_run/'plan.npz').resolve()),reference_sha256=plan['plan_sha256'],
             evaluation_cases={'foot_z_plus15':m['evaluation_cases']['foot_z_plus15'],'fresh_boundary_03':fm['evaluation_cases']['fresh_boundary_03']},
             training_cases={},evaluation_only=True,new_training_steps=0,parent_experiment=str(a.parent.resolve()),
             prefix_checkpoint=str((a.output/'prefix_policy.zip').resolve()),require_inactive_case_exact_parent=False,
             geometry_change_gate_sha256=sha(a.plan_run/'differential_geometry_audit.json'),
             case_parents={k:dict(path=str(v.resolve()),audit_sha256=verified[k]['audit_sha256']) for k,v in parents.items()},
             scope='Two known development cases; new 5 mm left-foot reference lift, original frozen actor and prefix, unchanged strict dynamics thresholds. Differential offline gate excludes only exactly unchanged right-support reference overlap. Full success and prefix equivalence checked separately; not independent generalization or new policy training.')
    (a.output/'manifest.json').write_text(json.dumps(m,indent=2))
    summary=dict(checkpoint_sha256=sha(a.output/'policy.zip'),evaluation_pending=True,evaluation_only=True,new_training_steps=0)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    rows=evaluate_parallel(a.output,a.output/'parallel_after',None,workers=2)['trajectories']
    for row in rows:shutil.copy2(a.output/'parallel_after'/f"after_{row['case']}.npz",a.output/f"after_{row['case']}.npz")
    summary.update(evaluation_pending=False,after=rows)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));(a.output/'after.json').write_text(json.dumps(rows,indent=2))
    audit(a.output);audit_fixed(a.output);audit_added(a.output)
    checks=[]
    for row in rows:
        case=row['case'];before=parents[case]/f'after_{case}.npz'
        assert sha(before)==verified[case]['rows'][case]['trajectory_sha256']
        with np.load(before) as x,np.load(a.output/f'after_{case}.npz') as y:
            exact={}
            for key in ('qpos','qvel','time_s','actions','observations','torques','rewards'):
                count=3511 if key in ('qpos','qvel','time_s') else 3510
                exact[key]=bool(np.array_equal(x[key][:count],y[key][:count]))
        checks.append(dict(case=case,passed=all(exact.values()),exact_before_70_2_s=exact,success=row['success'],reason=row['reason']))
    result=dict(passed=all(r['passed'] for r in checks),successes=sum(r['success'] for r in rows),cases=len(rows),rows=checks,
                manifest_sha256=sha(a.output/'manifest.json'),summary_sha256=sha(a.output/'summary.json'),automatic_promotion=False)
    (a.output/'pilot_audit.json').write_text(json.dumps(result,indent=2));assert result['passed'];print(json.dumps(result))


if __name__=='__main__':main()
