"""Dynamic calibration of retimed unload references; retain every declared case."""
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from training.retimed_unload_env import RetimedUnloadEnv
from tools.continue_vector_full_cycle_training import sha
ROOT=Path(__file__).resolve().parents[1]

def run_case(job):
    output,label,plan,bias,case,offsets=job
    p=Path(output)/label;p.mkdir(exist_ok=False)
    env=RetimedUnloadEnv(plan=plan,cases={case:offsets},normal_bias_y_deg=bias)
    obs,_=env.reset(options={'case':case});e=env.env
    sources={}
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if not file:continue
        source=Path(file).resolve()
        if source.suffix!='.py' or not source.is_relative_to(ROOT):continue
        relative=source.relative_to(ROOT)
        if relative.parts[0] not in ('training','tools'):continue
        target=p/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source.read_bytes());sources[str(relative)]=sha(target)
    manifest=dict(label=label,reference=str(Path(plan).resolve()),reference_sha256=sha(plan),
        bundle=str(e.bundle),bundle_manifest_sha256=sha(e.bundle/'manifest.json'),
        case=case,offsets=offsets,normal_bias_y_deg=bias,memory_enabled=True,
        contract=env.contract,horizon_s=env.horizon,sources=sources,
        bias_window_s=[49.,61.56],protocol_sha256=sha(Path(output)/'protocol.json'),
        scope='Unchanged strict environment; zero residual. Reference and offsets are explicitly frozen. Bias window remains absolute49..61.56s; no phase-alignment claim.')
    manifest['controller_modifier']=dict(type='reference-clock-unload-v1',factor=env.unload_factor)
    manifest['bias_window_s']=[49.,52.2+env.unload_factor*(61.56-52.2)]
    manifest['scope']='Same geometric path; slower unload, with contact-normal bias aligned to original reference time. Memory expiry remains in physical seconds. Zero actions and original strict thresholds.'
    (p/'manifest.json').write_text(json.dumps(manifest,indent=2))
    q=[e.data.qpos.copy()];v=[e.data.qvel.copy()];times=[float(e.data.time)]
    observations=[];actions=[];torques=[];rewards=[];wall=time.perf_counter()
    try:
        while True:
            action=np.zeros(26);observations.append(obs.copy());actions.append(action)
            obs,reward,done,truncated,info=env.step(action)
            q.append(e.data.qpos.copy());v.append(e.data.qvel.copy());times.append(float(e.data.time))
            torques.append(env.last_torque.copy());rewards.append(reward)
            if done or truncated:break
        np.savez_compressed(p/'trajectory.npz',qpos=q,qvel=v,time_s=times,observations=observations,actions=actions,torques=torques,rewards=rewards)
        result=dict(info,label=label,wall_seconds=time.perf_counter()-wall,trajectory_sha256=sha(p/'trajectory.npz'),manifest_sha256=sha(p/'manifest.json'))
        (p/'summary.json').write_text(json.dumps(result,indent=2))
        return {k:result[k] for k in ('label','success','reason','time_s','wall_seconds')}
    finally:env.close()


def main():
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    output=Path('outputs/retimed_unload_calibration_v1');output.mkdir(exist_ok=False)
    jobs=[(str(output),f'factor{factor:g}_height{h:g}',f'outputs/retimed_unload_factor{factor:g}_plan_v1/plan.npz',-3.,f'foot_z_plus{h:g}',{'rung_01':[0.,0.,h/1000]})
        for factor in (1.5,2.) for h in (12.5,15.,10.,5.,7.5)]
    protocol=dict(jobs=jobs,source_sha256=sha(__file__),
        selection='Accept only if at least one failure repaired and no success regresses. If both qualify prefer shorter duration. All cases known calibration cases.',
        scope='Reference timing differs, same pose knots and endpoint. Bias window follows reference phase; memory expiry remains physical time. Not unchanged-task policy comparison.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2));(output/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run_case,j) for j in jobs]):
            r=future.result();results.append(r);(output/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(r),flush=True)


if __name__=='__main__':main()
