"""Check actual interpolation support, including the interval before first edit."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.compare_audited_closure_candidates import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    m=json.loads((a.directory/'manifest.json').read_text())
    with np.load(m['parent']) as z:old={k:v.copy() for k,v in z.items()}
    with np.load(a.directory/'plan.npz') as z:new={k:v.copy() for k,v in z.items()}
    changed=np.flatnonzero(np.any(old['qpos']!=new['qpos'],axis=1))
    start=float(new['time'][changed[0]-1]);end=float(new['time'][changed[-1]+1])
    env=FullCycleResidualEnv(plan=a.directory/'plan.npz',cases=m['cases']);rows=[]
    try:
        for case in m['cases']:
            env.reset(options={'case':case});e=env.env;d=mujoco.MjData(e.model)
            failures=[];minimum=float('inf')
            for t in np.arange(start,end+1e-8,.005):
                d.qpos[:]=env.reference(float(t))[0];mujoco.mj_kinematics(e.model,d);mujoco.mj_collision(e.model,d)
                minimum=min(minimum,float(mujoco.mj_geomDistance(e.model,d,e.model.geom('left_sole_surface').id,e.model.geom('rung_01').id,1.,None)))
                for c in d.contact:
                    ids=[int(c.geom1),int(c.geom2)];bodies=[int(e.model.geom_bodyid[g]) for g in ids]
                    # All geometry altered by the plan belongs to the six left-leg joints.
                    left=any(e.model.body(b).name.startswith('left_') and any(s in e.model.body(b).name for s in ('hip','knee','ankle')) for b in bodies)
                    if left and c.dist < -1e-5:failures.append(dict(time_s=float(t),geoms=[e.model.geom(g).name for g in ids],distance_m=float(c.dist)))
            rows.append(dict(case=case,passed=not failures and minimum>=.003,minimum_target_clearance_m=minimum,failures=failures))
        speeds=lambda plan:float(abs(np.diff(plan['qpos'][:,e.qadr],axis=0)/np.diff(plan['time'])[:,None]).max())
        result=dict(passed=all(r['passed'] for r in rows),rows=rows,effective_start_s=start,effective_end_s=end,
                    parent_peak_joint_speed_rad_s=speeds(old),candidate_peak_joint_speed_rad_s=speeds(new),
                    manifest_sha256=sha(a.directory/'manifest.json'),plan_sha256=sha(a.directory/'plan.npz'),source_sha256=sha(__file__),
                    scope='Supplement to differential geometry gate. Linear interpolation starts modifying the reference at 74.96 s although first edited node is 75.06 s. Changed-leg collision and target-clearance checks cover the complete actual interval at 5 ms. No dynamic claim.')
        (a.directory/'interpolation_audit.json').write_text(json.dumps(result,indent=2));(a.directory/'interpolation_audit_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result));assert result['passed']
    finally:env.close()


if __name__=='__main__':main()
