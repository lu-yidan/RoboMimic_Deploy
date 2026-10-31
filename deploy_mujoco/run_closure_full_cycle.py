"""MuJoCo-only runner for audited composite actors with fixed waist/body commands."""
import argparse
from contextlib import ExitStack
import inspect
import json
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from training.full_cycle_residual_env import ROOT
from training.strict_full_cycle_residual_env import FullCycleResidualEnv,CONTRACT
from training.bar_geometry_observation import BarGeometryObservation,OBSERVATION_CONTRACT
from tools.audit_multirung_cycle import audit_endpoints
from training.closure_waist_target import install_waist_target
from training.closure_body_target import install_body_target,body_target_offset
from tools.continue_vector_full_cycle_training import sha


def load_verified(experiment,case):
    from deploy_mujoco.run_geometry_full_cycle import load_verified as load_geometry
    from training.closure_composite_policy import ClosureCompositePolicy
    base,observer,actor,manifest,summary=load_geometry(experiment,case)
    try:
        if 'controller_modifier' not in manifest or 'prefix_checkpoint' not in manifest:
            raise ValueError('Explicit composite actor and waist/body controller required')
        for name in ('composite_prefix_audit.json','controller_modifier_audit.json'):
            report=json.loads((experiment/name).read_text())
            if not report['passed']:
                raise ValueError('Incomplete controller or prefix audit: '+name)
        controller=json.loads((experiment/'controller_modifier_audit.json').read_text())
        if controller['manifest_sha256']!=sha(experiment/'manifest.json') or controller['summary_sha256']!=sha(experiment/'summary.json'):
            raise ValueError('Controller audit does not match experiment')
        return base,observer,ClosureCompositePolicy(actor,manifest),manifest,summary
    except Exception:
        observer.close();raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=Path,required=True);parser.add_argument('--case',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--viewer',action='store_true',help='Open a passive MuJoCo viewer; no robot connection')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1);base,observer,model,m,s=load_verified(args.experiment,args.case)
    args.output.mkdir(parents=True)
    provenance=dict(experiment=str(args.experiment.resolve()),checkpoint_sha256=sha(args.experiment/'policy.zip'),
        experiment_manifest_sha256=sha(args.experiment/'manifest.json'),integrity_audit_sha256=sha(args.experiment/'independent_training_audit.json'),
        runner_sha256=sha(__file__),contract=m['contract'],observation_contract=OBSERVATION_CONTRACT,geometry_mode=m['geometry_mode'],
        case=args.case,offsets=m['evaluation_cases'][args.case],simulation_only=True,
        controller_modifier=m['controller_modifier'],prefix_checkpoint_sha256=m['prefix_checkpoint_sha256'],scope='Composite actor plus explicit fixed waist/body references in MuJoCo only. Oracle geometry, scripted reference and privileged contact controller remain active. No hardware connection or default policy replacement.')
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2));(args.output/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    obs,info=observer.reset(seed=20261022,options={'case':args.case});e=base.env
    modifier=m['controller_modifier'];gain=float(np.clip((float(obs[138])-.625)/.125,0.,1.))
    waist=install_waist_target(base,modifier['waist_bias'],geometry_gain=gain,ramp_seconds=modifier['waist_ramp_seconds'],record=True)
    drop=gain*modifier['body_drop_m'];install_body_target(base,drop)
    (args.output/'scene_variant.json').write_text(json.dumps(base.variant,indent=2))
    q=[e.data.qpos.copy()];v=[e.data.qvel.copy()];times=[float(e.data.time)];actions=[];observations=[];torques=[];rewards=[];started=time.perf_counter()
    try:
        with ExitStack() as stack:
            viewer=None
            if args.viewer:
                import mujoco.viewer
                viewer=stack.enter_context(mujoco.viewer.launch_passive(e.model,e.data))
            while True:
                if viewer is not None and not viewer.is_running():
                    info=dict(info,reason='viewer_closed',success=False);break
                step_start=time.perf_counter();observations.append(obs.copy());action=model.predict(obs,deterministic=True)[0]
                obs,reward,done,truncated,info=observer.step(action)
                actions.append(action);rewards.append(reward);torques.append(base.last_torque.copy());q.append(e.data.qpos.copy());v.append(e.data.qvel.copy());times.append(float(e.data.time))
                if viewer is not None:viewer.sync();time.sleep(max(0.,.02-(time.perf_counter()-step_start)))
                if len(actions)%500==0:print(json.dumps(dict(time_s=times[-1],reason=info['reason'])),flush=True)
                if done or truncated:break
        path=args.output/'trajectory.npz';np.savez_compressed(path,qpos=q,qvel=v,time_s=times,actions=actions,observations=observations,torques=torques,rewards=rewards,waist_reference_trace=np.asarray(waist['trace']).reshape(-1,10),body_reference_trace=np.array([body_target_offset(t,drop) for t in times]),controller_geometry_gain=np.array(gain))
        endpoints=audit_endpoints(e,dict(qpos=np.asarray(q),qvel=np.asarray(v),time_s=np.asarray(times)),base.plan)
        result=dict(runtime=info,independently_accepted=bool(info['success'] and all(r['passed'] for r in endpoints)),independent_endpoints=endpoints,
            wall_seconds=time.perf_counter()-started,trajectory_sha256=sha(path),simulation_only=True,autonomous_climbing=False)
        parent_record=next(row for row in s['after'] if row['case']==args.case)
        parent_path=args.experiment/f'after_{args.case}.npz'
        if sha(parent_path)!=parent_record['trajectory_sha256']:raise ValueError('Audited parent trajectory changed')
        with np.load(parent_path) as original,np.load(path) as reproduced:
            checks={key:bool(np.array_equal(original[key],reproduced[key])) for key in ('qpos','qvel','time_s','observations','actions','torques','rewards','waist_reference_trace','body_reference_trace')}
        result.update(exact_reproduction=all(checks.values()),exact_checks=checks,parent_trajectory_sha256=sha(parent_path),controller_geometry_gain=gain)
        (args.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        if not args.viewer and not result['exact_reproduction']:raise AssertionError('Headless runner differs from audited evaluation')
    finally:observer.close()


if __name__=='__main__':main()
