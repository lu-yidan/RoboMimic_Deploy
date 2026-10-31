"""Read-only T12 kinematic evidence; no assertion of a causal mechanism."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.validate_swing_reference import read,write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--candidate-plan',type=Path,required=True);p.add_argument('--candidate-trajectory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--case',default='training_12')
    a=p.parse_args();m=read(a.parent/'manifest.json')
    env=FullCycleResidualEnv(plan=m['reference'],cases={a.case:m['evaluation_cases'][a.case]},
        normal_bias_y_deg=m['normal_bias_y_deg'],memory_enabled=m['force_history_memory'])
    env.reset(options={'case':a.case});e=env.env;model=e.model;d=mujoco.MjData(model)
    foot=model.body('left_ankle_roll_link').id
    fg=[g for g in range(model.ngeom) if model.geom_bodyid[g]==foot and (model.geom_contype[g] or model.geom_conaffinity[g])]
    rung=model.geom('rung_01').id
    def metrics(q):
        d.qpos[:]=q;mujoco.mj_kinematics(model,d);mujoco.mj_collision(model,d)
        rotation=d.xmat[foot].reshape(3,3)
        return dict(position_m=d.xpos[foot].tolist(),pitch_deg=float(np.rad2deg(np.arctan2(-rotation[2,0],np.hypot(rotation[0,0],rotation[1,0])))),
            rung01_gap_mm=1000*min(mujoco.mj_geomDistance(model,d,g,rung,.2,None) for g in fg))
    refs=[np.load(m['reference']),np.load(a.candidate_plan)]
    traces=[np.load(a.parent/f'after_{a.case}.npz'),np.load(a.candidate_trajectory)]
    rows=[]
    for requested in (53.7,70.2,77.5,77.56,79.,80.66,81.46,82.26,82.66,90.,99.66,113.64):
        row=dict(requested_time_s=requested)
        for label,r,z in zip(('old','lift'),refs,traces):
            j=int(np.argmin(abs(r['time']-requested)));i=int(np.argmin(abs(z['time_s']-requested)))
            row[label]=dict(reference_time_s=float(r['time'][j]),phase=str(r['phase'][j]),reference=metrics(r['qpos'][j]),actual=metrics(z['qpos'][i]))
        rows.append(row)
    tail=[]
    for label,z in zip(('old','lift'),traces):
        selected=(z['time_s']>111.66)&(z['time_s']<=113.6600001)
        errors=abs(z['qpos'][selected][:,e.qadr]-refs[0]['qpos'][0,e.qadr]).max(axis=0)
        top=np.argsort(errors)[-6:][::-1]
        gaps=[(float(t),metrics(q)['rung01_gap_mm']) for t,q in zip(z['time_s'],z['qpos']) if 78<=t<=84]
        tail.append(dict(label=label,dominant_closure_joints=[dict(joint=model.joint(int(e.joints[i])).name,error_mrad=float(errors[i]*1000)) for i in top],
            first_sampled_touch_s=next((t for t,gap in gaps if gap<=0),None),
            minimum_sampled_gap_mm=min(g for _,g in gaps)))
    result=dict(case=a.case,rows=rows,terminal=tail,
        sources=dict(parent_manifest=sha(a.parent/'manifest.json'),old_reference=sha(m['reference']),new_reference=sha(a.candidate_plan),
            old_trajectory=sha(a.parent/f'after_{a.case}.npz'),new_trajectory=sha(a.candidate_trajectory),analysis_source=sha(__file__)),
        scope='Kinematics of saved 50 Hz states and reference knots in the declared T12 geometry. Contact onset is sampled geometric intersection, not force onset. Correlation alone cannot establish the cause of terminal closure regression.')
    a.output.parent.mkdir(parents=True,exist_ok=True);write(a.output,result);print(json.dumps(tail,indent=2));env.close()


if __name__=='__main__':main()
