"""Test existing references in the unchanged strict residual environment.

Zero actor actions isolate reference/controller compatibility. This does not
compare learned policies, and geometry-specific targets remain labelled.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha

ROOT=Path(__file__).resolve().parents[1]
PLANS={
    'original':'outputs/clearance_back20_leg14_plan_v1/plan.npz',
    'clearance':'outputs/early_swing14_lift26_plan_v1/plan.npz',
    'adapted':'outputs/highrod_endpoint_adapt_plan_v1/plan.npz',
}


def run_case(job):
    output,label,plan,bias,*variant=job
    case=variant[0] if variant else 'foot_z_plus10'
    offsets=variant[1] if variant else {'rung_01':[0.,0.,.01]}
    p=Path(output)/label;p.mkdir(exist_ok=False)
    from training.window_acceleration_qp_env import WindowAccelerationQPEnv
    env=WindowAccelerationQPEnv(plan=plan,cases={case:offsets},normal_bias_y_deg=bias)
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
    manifest['controller_modifier']=dict(type='oracle-acceleration-qp-window-v1',start_s=57.,end_s=65.,ramp_s=2.,contact_weight=1000.,normal_damping=20.)
    manifest['scope']='Local oracle acceleration/force QP blended with original torque; actual contacts bypass normal bias for new QP. Original biased controller/history still runs. No sensing or policy training claim.'
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
        (p/'qp_records.json').write_text(json.dumps(env.qp_records,indent=2))
        result['qp_records_sha256']=sha(p/'qp_records.json')
        (p/'summary.json').write_text(json.dumps(result,indent=2))
        return {k:result[k] for k in ('label','success','reason','time_s','wall_seconds')}
    finally:env.close()


def main():
    output=Path('outputs/window_acceleration_qp_calibration_v1');output.mkdir(exist_ok=False)
    plan='outputs/support_retarget_plan_v1/plan.npz'
    gate=Path('outputs/contact_acceleration_qp_soft_probe_v1/independent_audit.json');assert json.loads(gate.read_text())['passed']
    jobs=[(str(output),f'true_height_{h:g}mm',plan,-3.,f'foot_z_plus{h:g}',{'rung_01':[0.,0.,h/1000]}) for h in (12.5,15.,10.,5.,7.5)]
    protocol=dict(jobs=jobs,plan_sha256=sha(plan),offline_gate_sha256=sha(gate),source_sha256=sha(__file__),selection='At least one repaired support-reference failure and no success regression.',scope='Local oracle controller calibration; same reference and strict physical gates. New QP uses actual geometry, so not an isolated unbiased-estimation comparison.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2));(output/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run_case,j) for j in jobs]):
            result=future.result();results.append(result);(output/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
