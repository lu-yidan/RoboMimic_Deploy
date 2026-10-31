"""Isolate changed-leg geometry from inherited support-reference overlaps."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.compare_audited_closure_candidates import sha
from tools.plan_clearance_cycle import audit_plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan-run',type=Path,required=True)
    a=p.parse_args();m=json.loads((a.plan_run/'manifest.json').read_text())
    assert sha(a.plan_run/'plan.npz')==m['plan_sha256'] and sha(m['parent'])==m['parent_sha256']
    with np.load(m['parent']) as z:parent={k:v.copy() for k,v in z.items()}
    with np.load(a.plan_run/'plan.npz') as z:candidate={k:v.copy() for k,v in z.items()}
    env=FullCycleResidualEnv(plan=m['parent'],cases=m['cases']);env.reset(options={'case':'nominal'})
    e=env.env;model=e.model;data=mujoco.MjData(model)
    changed=set(e.qadr[:6]);other=[i for i in range(model.nq) if i not in changed]
    checks=dict(other_qpos_exact=np.array_equal(parent['qpos'][:,other],candidate['qpos'][:,other]),
                metadata_exact=all(np.array_equal(parent[k],candidate[k]) for k in parent if k!='qpos'))
    rows=[]
    try:
        nominal=audit_plan(e,candidate,dt=.005)
        nominal['scope']='Nominal geometry only; inherited IK residual field. Changed-node IK residuals are in manifest.json. Perturbed geometry is separately reported.'
        (a.plan_run/'audit.json').write_text(json.dumps(nominal,indent=2))
        checks['nominal_full_plan_geometry']=nominal['passed']
        for case in m['cases']:
            env.reset(options={'case':case});inherited=0;new=[];maximum_support_error=0.
            for time in np.arange(m['start_s'],m['end_s']+1e-8,.005):
                j=min(max(int(np.searchsorted(parent['time'],time,side='right')-1),0),len(parent['time'])-2)
                u=(time-parent['time'][j])/(parent['time'][j+1]-parent['time'][j]);contacts=[]
                for plan in (parent,candidate):
                    data.qpos[:]=(1-u)*plan['qpos'][j]+u*plan['qpos'][j+1]
                    mujoco.mj_kinematics(model,data);mujoco.mj_collision(model,data);pairs={}
                    for c in data.contact:
                        pair=tuple(sorted((int(c.geom1),int(c.geom2))))
                        pairs[pair]=min(float(c.dist),pairs.get(pair,float('inf')))
                    contacts.append(pairs)
                for pair,dist in contacts[1].items():
                    bodies=[int(model.geom_bodyid[g]) for g in pair]
                    if not any(b in e.robot_bodies for b in bodies):continue
                    expected=sum(b in e.robot_bodies for b in bodies)==1 and any(b in e.contact_groups for b in bodies)
                    if dist>=(-.0007 if expected else -.00001):continue
                    old=contacts[0].get(pair)
                    names=[model.geom(g).name for g in pair]
                    # Only exactly unchanged right-sole support overlap is inherited.
                    is_right_support=set(names)=={'right_sole_surface','rung_01'}
                    if is_right_support and old is not None and abs(dist-old)<=1e-12:
                        inherited+=1;maximum_support_error=max(maximum_support_error,abs(dist-old))
                    else:new.append(dict(time_s=float(time),geoms=names,distance_m=dist,parent_distance_m=old))
            row=next(r for r in m['checks'] if r['case']==case)
            rows.append(dict(case=case,passed=not new and row['minimum_sole_target_clearance_m']>=.003,
                             inherited_right_support_samples=inherited,maximum_support_distance_error_m=maximum_support_error,
                             changed_leg_target_clearance_m=row['minimum_sole_target_clearance_m'],new_or_changed_violations=new))
        result=dict(passed=bool(all(checks.values()) and all(r['passed'] for r in rows)),checks={k:bool(v) for k,v in checks.items()},rows=rows,
                    original_geometry_report_passed=m['passed'],manifest_sha256=sha(a.plan_run/'manifest.json'),source_sha256=sha(__file__),
                    scope='Differential offline gate for the changed left leg only. Original full-reference geometry report remains failed because the unchanged right-sole reference overlaps the perturbed rung; physical controller response is not represented by that scripted pose. No whole-body collision-free reference claim. A strict time-zero dynamic pilot is still required; collision and task thresholds are unchanged.')
        (a.plan_run/'differential_geometry_audit.json').write_text(json.dumps(result,indent=2))
        (a.plan_run/'differential_audit_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result))
    finally:env.close()


if __name__=='__main__':main()
