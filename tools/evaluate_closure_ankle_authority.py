"""Test one bounded terminal ankle action on all five known heights."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import os
from pathlib import Path
import sys
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha

ROOT=Path(__file__).resolve().parents[1]


def envelope(t):
    enter=float(np.clip((t-99.66)/3.,0.,1.));leave=float(np.clip((116.66-t)/3.,0.,1.))
    return min(enter*enter*(3-2*enter),leave*leave*(3-2*leave))


def run(job):
    output,case,offsets,vector=job;p=Path(output)/case;p.mkdir(exist_ok=False)
    plan=Path('outputs/support_retarget_plan_v1/plan.npz')
    env=FullCycleResidualEnv(plan=plan,cases={case:offsets},normal_bias_y_deg=-3.)
    obs,_=env.reset(options={'case':case});e=env.env;sources={}
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if not path:continue
        path=Path(path).resolve()
        if path.suffix!='.py' or not path.is_relative_to(ROOT):continue
        rel=path.relative_to(ROOT)
        if rel.parts[0] not in ('training','tools'):continue
        dest=p/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());sources[str(rel)]=sha(dest)
    manifest=dict(label=case,case=case,offsets=offsets,reference=str(plan.resolve()),reference_sha256=sha(plan),bundle=str(e.bundle),
        bundle_manifest_sha256=sha(e.bundle/'manifest.json'),normal_bias_y_deg=-3.,memory_enabled=True,contract=env.contract,horizon_s=env.horizon,
        sources=sources,protocol_sha256=sha(Path(output)/'protocol.json'),action_rule=dict(type='fixed_vector_cubic_envelope',vector=vector,start_s=99.66,end_s=116.66,ramp_s=3.),
        scope='Fixed positive left-ankle-pitch residual during closure, using the existing maximum0.015rad policy authority. No base command, direct state modification or training. Same candidate for all five heights.')
    (p/'manifest.json').write_text(json.dumps(manifest,indent=2))
    q=[e.data.qpos.copy()];v=[e.data.qvel.copy()];t=[float(e.data.time)];observations=[];actions=[];torques=[];rewards=[]
    try:
        while True:
            action=envelope(float(e.data.time))*np.asarray(vector);observations.append(obs.copy());actions.append(action.copy())
            obs,reward,done,truncated,info=env.step(action)
            q.append(e.data.qpos.copy());v.append(e.data.qvel.copy());t.append(float(e.data.time));torques.append(env.last_torque.copy());rewards.append(reward)
            if done or truncated:break
        np.savez_compressed(p/'trajectory.npz',qpos=q,qvel=v,time_s=t,observations=observations,actions=actions,torques=torques,rewards=rewards)
        result=dict(info,label=case,trajectory_sha256=sha(p/'trajectory.npz'),manifest_sha256=sha(p/'manifest.json'))
        (p/'summary.json').write_text(json.dumps(result,indent=2));return {k:result[k] for k in ('label','success','reason','time_s')}
    finally:env.close()


def main():
    output=Path('outputs/closure_ankle_authority_v1');output.mkdir(exist_ok=False)
    env=FullCycleResidualEnv(plan='outputs/support_retarget_plan_v1/plan.npz');e=env.env
    joint=list(e.joints).index(e.model.joint('left_ankle_pitch_joint').id);index=list(env.indices).index(joint);env.close()
    vector=np.zeros(26);vector[index]=1.
    cases={f'foot_z_plus{h:g}':{'rung_01':[0.,0.,h/1000]} for h in (15.,12.5,10.,5.,7.5)}
    jobs=[(str(output),case,offset,vector.tolist()) for case,offset in cases.items()]
    protocol=dict(jobs=jobs,cases=cases,normalized_action_vector=vector.tolist(),source_sha256=sha(__file__),selection='Single maximum positive left-ankle reference bias; repair at least one baseline failure without success regression.',scope='Known-case action-authority calibration on the selected support reference. Original actor action bounds/filter/dynamics/strict acceptance preserved. No learned-policy or generalization claim.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2));(output/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run,j) for j in jobs]):
            r=future.result();results.append(r);(output/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(r),flush=True)


if __name__=='__main__':main()
