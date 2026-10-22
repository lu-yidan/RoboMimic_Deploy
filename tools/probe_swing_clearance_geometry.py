"""Local kinematic clearance probe; not a feasible motor-control demonstration."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('experiment',type=Path);args=parser.parse_args()
    p=args.experiment;m=json.loads((p/'manifest.json').read_text());s=json.loads((p/'summary.json').read_text())
    audit=json.loads((p/'independent_training_audit.json').read_text())
    if not audit['passed'] or sha(p/'summary.json')!=audit['experiment_summary_sha256']:raise ValueError('Audited parent required')
    case='foot_z_plus10';path=p/f'after_{case}.npz';record=next(r for r in s['after'] if r['case']==case)
    if sha(path)!=record['trajectory_sha256']:raise ValueError('Trajectory changed')
    env=FullCycleResidualEnv(plan=m['reference'],cases={case:m['evaluation_cases'][case]});env.reset(options={'case':case})
    e=env.env;model=e.model;d=mujoco.MjData(model)
    rod=model.geom('rung_01').id;sole=model.geom('right_sole_surface').id
    qadr=e.qadr[6:12];joints=e.joints[6:12]
    anchors=[model.body(n).id for n in ('left_ankle_roll_link','hook','right_hook')]
    def distance(q):
        d.qpos[:]=q;mujoco.mj_kinematics(model,d)
        return float(mujoco.mj_geomDistance(model,d,rod,sole,1.,np.zeros(6)))
    results=[]
    with np.load(path) as z:
        for desired in [40.,41.,41.8,float(z['time_s'][-1])]:
            i=int(np.argmin(abs(z['time_s']-desired)));q=z['qpos'][i].copy();before=distance(q)
            anchor_before=d.xpos[anchors].copy();gradient=[]
            for address in qadr:
                plus=q.copy();minus=q.copy();plus[address]+=1e-5;minus[address]-=1e-5
                gradient.append((distance(plus)-distance(minus))/2e-5)
            gradient=np.asarray(gradient)
            lower=np.maximum(-.015,model.jnt_range[joints,0]-q[qadr]);upper=np.minimum(.015,model.jnt_range[joints,1]-q[qadr])
            if np.any(lower>upper):raise ValueError('Invalid native-bound interval')
            delta=np.where(gradient>=0,upper,lower);changed=q.copy();changed[qadr]+=delta
            after=distance(changed);anchor_difference=float(abs(d.xpos[anchors]-anchor_before).max())
            results.append(dict(time_s=float(z['time_s'][i]),phase=env.reference(float(z['time_s'][i]))[3],
                before_distance_m=before,linearized_candidate_distance_m=float(before+gradient@delta),
                nonlinear_candidate_distance_m=after,right_leg_delta_rad=delta.tolist(),
                fixed_support_body_position_difference_m=anchor_difference))
    sources={__file__:sha(__file__)}
    result=dict(results=results,perturbation_bound_rad=.015,trajectory_sha256=sha(path),sources=sources,
        scope='Offline right-leg joint-pose perturbations, fixed pelvis/other joints, native limits enforced. Linearized box maximizer is checked with nonlinear geometry, not a global optimum. Same numeric size as residual cap does NOT mean actual tracking, torque, timing or full-cycle feasibility. No dynamics stepped.')
    (p/'swing_clearance_geometry_probe.json').write_text(json.dumps(result,indent=2))
    (p/'swing_clearance_geometry_probe_source.py').write_bytes(Path(__file__).read_bytes());env.close()
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
