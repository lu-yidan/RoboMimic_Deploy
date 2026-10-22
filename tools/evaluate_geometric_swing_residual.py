"""One preregistered motor-only clearance candidate derived from offline geometry."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.audit_multirung_cycle import audit_endpoints
from tools.continue_vector_full_cycle_training import sha


def smooth(x):
    x=float(np.clip(x,0.,1.));return x*x*(3.-2.*x)


def action_rule(time_s,phase,z_offset,start,end,amplitude=1.):
    action=np.zeros(26)
    if phase=='rf_swing':
        gain=amplitude*min(smooth(time_s-start),smooth(end-time_s))*float(np.clip(z_offset/.01,0.,1.))
        action[6:12]=gain*np.array([1.,-1.,1.,1.,-1.,1.])
    return action


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--case',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--amplitude',type=float,choices=[.5,.75,1.],default=1.)
    args=parser.parse_args()
    m=json.loads((args.parent/'manifest.json').read_text());a=json.loads((args.parent/'independent_training_audit.json').read_text())
    if not a['passed'] or sha(args.parent/'manifest.json')!=a['experiment_manifest_sha256']:raise ValueError('Audited parent required')
    for source,digest in m['sources'].items():
        if source.replace('\\','/').startswith('training/') and sha(source)!=digest:raise ValueError('Parent controller changed')
    if args.case not in m['evaluation_cases']:raise ValueError('Declared case required')
    env=FullCycleResidualEnv(plan=m['reference'],cases={args.case:m['evaluation_cases'][args.case]},
        normal_bias_y_deg=m['normal_bias_y_deg'],memory_enabled=m['force_history_memory'])
    obs,_=env.reset(options={'case':args.case});e=env.env
    ids=np.flatnonzero(env.plan['phase']=='rf_swing');start=float(env.plan['time'][ids[0]-1]);end=float(env.plan['time'][ids[-1]])
    sid=e.model.site('rung_01_target').id;dz=float(e.model.site_pos[sid,2]-env.nominal_geometry[1][sid,2])
    args.output.mkdir(exist_ok=False)
    manifest=dict(case=args.case,offsets=m['evaluation_cases'][args.case],parent=str(args.parent.resolve()),
        parent_audit_sha256=sha(args.parent/'independent_training_audit.json'),source_sha256=sha(__file__),
        swing_start_s=start,swing_end_s=end,ramp_seconds=1.,right_leg_signs=[1,-1,1,1,-1,1],
        geometry_scale_m=.01,amplitude=args.amplitude,measured_target_z_offset_m=dz,
        candidate_basis='One candidate fixed before this run from the41.8s local clearance probe of blind foot_z_plus10 failure.',
        scope='Privileged-geometry conditioned fixed residual, not a learned policy or unseen test. Original residual filter, +/-0.015rad reference bound and strict acceptance unchanged.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2));(args.output/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    q=[e.data.qpos.copy()];v=[e.data.qvel.copy()];times=[float(e.data.time)];actions=[];observations=[];torques=[];rewards=[]
    try:
        while True:
            phase=env.reference(float(e.data.time))[3]
            action=action_rule(float(e.data.time),phase,dz,start,end,args.amplitude)
            observations.append(obs.copy());actions.append(action.copy())
            obs,reward,done,truncated,info=env.step(action)
            q.append(e.data.qpos.copy());v.append(e.data.qvel.copy());times.append(float(e.data.time));torques.append(env.last_torque.copy());rewards.append(reward)
            if done or truncated:break
        path=args.output/'trajectory.npz'
        np.savez_compressed(path,qpos=q,qvel=v,time_s=times,actions=actions,observations=observations,torques=torques,rewards=rewards)
        endpoints=audit_endpoints(e,dict(qpos=np.asarray(q),qvel=np.asarray(v),time_s=np.asarray(times)),env.plan)
        replay=np.array([action_rule(t,env.reference(t)[3],dz,start,end,args.amplitude) for t in times[:-1]])
        exact=bool(np.array_equal(replay,np.asarray(actions)))
        result=dict(info,endpoints=endpoints,action_rule_replay_exact=exact,
            accepted=bool(info['success'] and exact and all(r['passed'] for r in endpoints)),
            trajectory_sha256=sha(path),manifest_sha256=sha(args.output/'manifest.json'),
            scope='Runtime checks plus saved-state endpoint and action-rule checks. Separate full-trajectory independent audit still required for a claimed reusable baseline.')
        (args.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    finally:env.close()


if __name__=='__main__':main()
