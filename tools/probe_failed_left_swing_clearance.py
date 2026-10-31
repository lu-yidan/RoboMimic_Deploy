"""Offline left-foot clearance sensitivity at a saved, audited failure."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.compare_audited_closure_candidates import load_run,sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',default='fresh_boundary_03')
    a=p.parse_args();verified=load_run(a.experiment);assert not verified['rows'][a.case]['success']
    m=json.loads((a.experiment/'manifest.json').read_text())
    env=FullCycleResidualEnv(plan=m['reference'],cases={a.case:m['evaluation_cases'][a.case]})
    env.reset(options={'case':a.case});e=env.env;model=e.model;data=mujoco.MjData(model)
    rod=model.geom('rung_01').id;sole=model.geom('left_sole_surface').id
    qadr=e.qadr[:6];joints=e.joints[:6]
    anchors=[model.body(name).id for name in ('right_ankle_roll_link','hook','right_hook')]
    def clearance(q):
        data.qpos[:]=q;mujoco.mj_kinematics(model,data)
        return float(mujoco.mj_geomDistance(model,data,rod,sole,1.,np.zeros(6)))
    rows=[]
    trajectory=a.experiment/f'after_{a.case}.npz'
    try:
        with np.load(trajectory) as z:
            for requested in (77.0,77.3,77.5,float(z['time_s'][-1])):
                index=int(np.argmin(abs(z['time_s']-requested)));q=z['qpos'][index].copy()
                before=clearance(q);anchor_before=data.xpos[anchors].copy();gradient=[]
                for address in qadr:
                    plus=q.copy();minus=q.copy();plus[address]+=1e-5;minus[address]-=1e-5
                    gradient.append((clearance(plus)-clearance(minus))/2e-5)
                gradient=np.asarray(gradient)
                lower=np.maximum(-.015,model.jnt_range[joints,0]-q[qadr]);upper=np.minimum(.015,model.jnt_range[joints,1]-q[qadr])
                assert np.all(lower<=upper)
                delta=np.where(gradient>=0,upper,lower);changed=q.copy();changed[qadr]+=delta
                after=clearance(changed)
                rows.append(dict(time_s=float(z['time_s'][index]),before_distance_m=before,
                    linearized_distance_m=float(before+gradient@delta),nonlinear_distance_m=after,
                    left_leg_delta_rad=delta.tolist(),other_support_body_position_difference_m=float(abs(data.xpos[anchors]-anchor_before).max())))
    finally:env.close()
    result=dict(rows=rows,perturbation_bound_rad=.015,parent_audit_sha256=verified['audit_sha256'],
        trajectory_sha256=sha(trajectory),source_sha256=sha(__file__),
        scope='Offline joint-position sensitivity only. The same numeric size as the actor reference bound does not establish tracking, dynamic reachability, all-body collision clearance or a feasible full cycle. Other support body positions stay fixed; no dynamics or task outcomes changed.')
    a.output.mkdir(exist_ok=False);(a.output/'summary.json').write_text(json.dumps(result,indent=2));(a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result))


if __name__=='__main__':main()
