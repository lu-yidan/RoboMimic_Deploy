"""Validate bounded reference feedback using a complete, uninterrupted simulation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from deploy_mujoco.run_closure_full_cycle import load_verified
from training.closure_waist_target import install_waist_target
from training.closure_body_target import install_body_target
from training.closure_foreaft_reference import install_adaptive_reference, SCALES
from tools.continue_vector_full_cycle_training import sha
from tools.audit_multirung_cycle import audit_endpoints


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['zero','probe'],required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    base,observer,teacher,manifest,previous=load_verified(args.parent,'foot_z_plus15')
    args.output.mkdir(exist_ok=False)
    sources={p:sha(p) for p in ('training/closure_foreaft_reference.py','tools/check_closure_foreaft_reference.py')}
    (args.output/'protocol.json').write_text(json.dumps(dict(mode=args.mode,parent_manifest_sha256=sha(args.parent/'manifest.json'),sources=sources,
        scope='Interface check, not PPO training or a selected policy improvement. Full time-zero simulation with original task gates.'),indent=2))
    obs,_=observer.reset(seed=20261022,options={'case':'foot_z_plus15'})
    gain=float(np.clip((float(obs[138])-.625)/.125,0.,1.));fixed=manifest['controller_modifier']
    install_waist_target(base,fixed['waist_bias'],geometry_gain=gain,ramp_seconds=fixed['waist_ramp_seconds'])
    install_body_target(base,gain*fixed['body_drop_m'])
    adaptive=install_adaptive_reference(base,fixed,geometry_gain=gain,record=True)
    command=np.zeros(5) if args.mode=='zero' else np.array([0.,0.,0.,0.,.25])
    arrays=dict(qpos=[base.env.data.qpos.copy()],qvel=[base.env.data.qvel.copy()],time_s=[float(base.env.data.time)],
        actions=[],observations=[],torques=[],rewards=[])
    try:
        while True:
            adaptive['set_action'](command)
            arrays['observations'].append(obs.copy());action=teacher.predict(obs,deterministic=True)[0]
            obs,reward,done,truncated,info=observer.step(action)
            arrays['actions'].append(action);arrays['rewards'].append(reward);arrays['torques'].append(base.last_torque.copy())
            arrays['qpos'].append(base.env.data.qpos.copy());arrays['qvel'].append(base.env.data.qvel.copy());arrays['time_s'].append(float(base.env.data.time))
            if len(arrays['actions'])%1000==0:print(json.dumps(dict(time_s=arrays['time_s'][-1])),flush=True)
            if done or truncated:break
        arrays={k:np.asarray(v) for k,v in arrays.items()};trace=np.asarray(adaptive['trace']).reshape(-1,21)
        np.savez_compressed(args.output/'trajectory.npz',**arrays,adaptive_reference_trace=trace)
        source=args.parent/'after_foot_z_plus15.npz';record=next(r for r in previous['after'] if r['case']=='foot_z_plus15')
        assert sha(source)==record['trajectory_sha256']
        with np.load(source) as original:
            exact={k:bool(np.array_equal(arrays[k],original[k])) for k in arrays}
            prefix={k:bool(np.array_equal(arrays[k][:4984 if k in ('qpos','qvel','time_s') else 4983],original[k][:4984 if k in ('qpos','qvel','time_s') else 4983])) for k in arrays}
            physical_change=float(abs(arrays['qpos']-original['qpos']).max()) if arrays['qpos'].shape==original['qpos'].shape else None
        u=np.clip((trace[:,0]-99.66)/3.,0.,1.);target=trace[:,1:6]*SCALES*gain*(u*u*(3.-2.*u))[:,None]
        predicted=np.zeros_like(target);last=np.zeros(5)
        for i,row in enumerate(target):last=last+.05*(row-last);predicted[i]=last
        velocity=np.diff(np.vstack([np.zeros(5),predicted]),axis=0)/.005
        checks=dict(finite=bool(np.isfinite(trace).all()),cadence=bool(np.allclose(np.diff(trace[:,0]),.005,atol=1e-7,rtol=0)),
            target=bool(np.allclose(trace[:,6:11],target,atol=1e-15,rtol=0)),filter=bool(np.allclose(trace[:,11:16],predicted,atol=1e-15,rtol=0)),
            velocity=bool(np.allclose(trace[:,16:21],velocity,atol=1e-13,rtol=0)),bounded=bool(np.all(abs(trace[:,11:16])<=SCALES+1e-15)),prefix_exact=all(prefix.values()))
        endpoints=audit_endpoints(base.env,arrays,base.plan)
        if args.mode=='zero':checks['zero_exact']=all(exact.values())
        else:checks['physical_effect']=physical_change is not None and physical_change>1e-8
        result=dict(passed=all(checks.values()),checks=checks,runtime=info,independent_endpoints=endpoints,
            task_success=bool(info['success'] and all(r['passed'] for r in endpoints)),exact=exact,maximum_qpos_change=physical_change,
            controller_samples=len(trace),trajectory_sha256=sha(args.output/'trajectory.npz'),protocol_sha256=sha(args.output/'protocol.json'),sources=sources)
        (args.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        assert result['passed']
    finally:observer.close()


if __name__=='__main__':main()
