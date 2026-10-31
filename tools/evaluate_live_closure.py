"""Full, uninterrupted trials of bounded leg feedback added to the selected actor.

Known-case controller calibration, not new learning or independent test evidence.
No state restoration, external forces, reference edits, or acceptance changes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from deploy_mujoco.run_geometry_full_cycle import load_verified
from tools.audit_multirung_cycle import audit_endpoints
from training.closure_live_target import install_acceleration_target
from tools.continue_vector_full_cycle_training import sha


def correction(base, gain):
    t = float(base.env.data.time)
    u = float(np.clip((t-99.66)/3., 0., 1.))
    envelope = u*u*(3.-2.*u)
    error = base.env.data.qpos[base.env.qadr[base.indices]] - base.plan['qpos'][0, base.env.qadr[base.indices]]
    feedback = np.zeros(26)
    feedback[:12] = -gain*error[:12]/.015
    return envelope*np.clip(feedback, -1., 1.)


def run(job):
    output, case, gain, drop = job
    torch.set_num_threads(1)
    p = Path(output)/f'gain{gain:g}_drop{drop:g}_{case}'
    p.mkdir(exist_ok=False)
    experiment = Path('outputs/closure_margin_control98304_v1')
    base, observer, model, manifest, _ = load_verified(experiment, case)
    e = base.env
    provenance = dict(case=case, gain=gain, qp_blend=drop, body_target_source_sha256=sha(Path(__file__).resolve().parents[1]/"training/closure_live_target.py"), start_s=99.66, ramp_s=3.,
        parent_checkpoint_sha256=sha(experiment/'policy.zip'),
        parent_manifest_sha256=sha(experiment/'manifest.json'),
        reference_sha256=sha(manifest['reference']), source_sha256=sha(__file__),
        protocol_sha256=sha(Path(output)/'protocol.json'),
        action_rule='clip(actor + cubic_ramp * clip(-gain*leg_closure_error/0.015,-1,1),-1,1)',
        scope='Full-cycle known-case feedback calibration; unchanged strict task and motor limits.')
    (p/'provenance.json').write_text(json.dumps(provenance, indent=2))
    obs, info = observer.reset(seed=20261022, options={'case':case})
    install_acceleration_target(base,drop)
    q=[e.data.qpos.copy()]; v=[e.data.qvel.copy()]; ts=[float(e.data.time)]
    actions=[]; actors=[]; observations=[]; torques=[]; rewards=[]
    started=time.perf_counter()
    try:
        while True:
            actor=np.asarray(model.predict(obs, deterministic=True)[0], dtype=float)
            action=np.clip(actor+correction(base,gain),-1.,1.)
            observations.append(obs.copy()); actors.append(actor); actions.append(action)
            obs,reward,done,truncated,info=observer.step(action)
            rewards.append(reward);torques.append(base.last_torque.copy())
            q.append(e.data.qpos.copy());v.append(e.data.qvel.copy());ts.append(float(e.data.time))
            if done or truncated:break
        arrays=dict(qpos=np.asarray(q),qvel=np.asarray(v),time_s=np.asarray(ts),
            actions=np.asarray(actions),actor_actions=np.asarray(actors),observations=np.asarray(observations),
            torques=np.asarray(torques),rewards=np.asarray(rewards))
        np.savez_compressed(p/'trajectory.npz',**arrays)
        keys=('post_step_time','post_step_qvel','solver_qacc','constraint_force','joint_friction_force','contact_world_force_sum','qp_prediction','actuator_force')
        np.savez_compressed(p/'live_trace.npz',**{key:np.asarray([r[i] for r in base.live_trace]) for i,key in enumerate(keys)})
        endpoints=audit_endpoints(e,arrays,base.plan)
        with np.load(experiment/f'after_{case}.npz') as original:
            n=int(np.searchsorted(arrays['time_s'],99.66,side='left'))
            nr=int(np.searchsorted(arrays['time_s'][1:],99.66,side='left'))
            prefix={key:bool(np.array_equal(original[key][:n if key in ('qpos','qvel','time_s','observations','actions') else nr],arrays[key][:n if key in ('qpos','qvel','time_s','observations','actions') else nr])) for key in ('qpos','qvel','time_s','observations','actions','torques','rewards')}
        checks=dict(finite=all(np.isfinite(a).all() for a in arrays.values()),
            action_bounds=bool(abs(arrays['actions']).max()<=1.),
            torque_bounds=bool(np.all(arrays['torques']<=e.limits[:,1]+1e-8) and np.all(arrays['torques']>=e.limits[:,0]-1e-8)),
            prefix_exact=all(prefix.values()))
        tail=arrays['time_s']>base.horizon-2.
        rise=arrays['qpos'][tail,2]-base.plan['qpos'][0,2]
        result=dict(case=case,gain=gain,qp_blend=drop,runtime=info,endpoints=endpoints,checks=checks,prefix=prefix,
            independently_accepted=bool(info['success'] and all(checks.values()) and all(r['passed'] for r in endpoints)),
            root_rise_min_m=float(rise.min()) if len(rise) else None,root_rise_max_m=float(rise.max()) if len(rise) else None,
            live_trace_sha256=sha(p/"live_trace.npz"),qp_stats=base.closure_qp_stats,wall_seconds=time.perf_counter()-started,trajectory_sha256=sha(p/'trajectory.npz'))
        (p/'summary.json').write_text(json.dumps(result,indent=2))
        return result
    finally:observer.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gains',type=float,nargs='+',default=[0.])
    parser.add_argument('--cases',nargs='+',default=['foot_z_plus15'])
    parser.add_argument('--drops',type=float,nargs='+',default=[1.])
    args=parser.parse_args()
    if not all(np.isfinite(g) and g>=0 for g in args.gains):raise ValueError('Invalid feedback gains')
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(gains=args.gains,drops=args.drops,cases=args.cases,source_sha256=sha(__file__),
        parent='closure_margin_control98304_v1',acceptance='Original strict endpoint and physical gates unchanged.',
        selection='Screen known +15mm failure, then require no regression on all five known cases and all twelve existing validation cases before adoption. Screening alone is not adoption.',
        scope='Selected actor unchanged (joint correction gain0), terminal-only acceleration/force QP blend with inner root/joint acceleration barriers (omega2, root8mm, joints38mrad). Original references, task gates and motor limits retained. Actual simulated contacts and soft sticking approximation; not a learned-policy improvement.')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (args.output/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    (args.output/'body_target_source.py').write_bytes((Path(__file__).resolve().parents[1]/'training/closure_live_target.py').read_bytes())
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run,(str(args.output),case,gain,drop)) for gain in args.gains for drop in args.drops for case in args.cases]):
            row=future.result();results.append(row)
            (args.output/'results.json').write_text(json.dumps(results,indent=2))
            print(json.dumps(row),flush=True)


if __name__=='__main__':main()
