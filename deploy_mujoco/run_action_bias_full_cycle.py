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
from training.closure_foreaft_reference import install_adaptive_reference
from training.closure_action_bias_policy import ClosureActionBiasPolicy as ClosureCompositePolicy
from training.closure_body_target import install_body_target,body_target_offset
from tools.continue_vector_full_cycle_training import sha


def load_verified(experiment,case):
    m=json.loads((experiment/'manifest.json').read_text());s=json.loads((experiment/'summary.json').read_text());a=json.loads((experiment/'independent_candidate_audit.json').read_text())
    if not a['passed'] or s.get('evaluation_pending'):raise ValueError('Completed independent audit required')
    if m['contract']!=CONTRACT or m['observation_contract']!=OBSERVATION_CONTRACT or m['geometry_mode'] not in ('blind','oracle'):raise ValueError('Unsupported148-input policy contract')
    if case not in m['evaluation_cases']:raise ValueError('Choose a declared evaluation case')
    for filename,key in [('manifest.json','experiment_manifest_sha256'),('summary.json','experiment_summary_sha256'),('policy.zip','checkpoint_sha256')]:
        if sha(experiment/filename)!=a[key]:raise ValueError('Audited experiment changed: '+filename)
    if sha(m['reference'])!=m['reference_sha256']:raise ValueError('Reference changed')
    for path,digest in m['sources'].items():
        if sha(experiment/'source_snapshot'/path)!=digest:raise ValueError('Frozen source changed: '+path)
        if path.replace('\\','/').startswith('training/') and sha(ROOT/path)!=digest:raise ValueError('Runtime environment changed: '+path)
    model=PPO.load(experiment/'policy.zip',device='cpu')
    feature_source=Path(inspect.getfile(type(model.policy.features_extractor))).resolve();relative=feature_source.relative_to(ROOT)
    recorded=next((digest for path,digest in m['sources'].items() if Path(path)==relative),None)
    if recorded is None:
        supplement=json.loads((experiment/'actor_source_provenance.json').read_text())
        if Path(supplement['source'])!=relative or supplement['checkpoint_sha256']!=a['checkpoint_sha256'] or supplement['audit_sha256']!=sha(experiment/'independent_candidate_audit.json'):raise ValueError('Incomplete custom actor provenance')
        recorded=supplement['sha256']
    if sha(feature_source)!=recorded:raise ValueError('Custom actor implementation changed')
    base=FullCycleResidualEnv(plan=m['reference'],cases={case:m['evaluation_cases'][case]},normal_bias_y_deg=m['normal_bias_y_deg'],memory_enabled=m['force_history_memory'])
    try:
        if sha(base.env.bundle/'manifest.json')!=m['bundle_manifest_sha256']:raise ValueError('Scene changed')
        observer=BarGeometryObservation(base,mode=m['geometry_mode'])
        if model.observation_space.shape!=observer.observation_space.shape or model.action_space.shape!=observer.action_space.shape:raise ValueError('Policy dimensions differ')
    except Exception:
        base.close();raise
    try:
        if m.get('adaptive_reference_scales') != [.003,.003,.003,.00015,.001]:
            raise ValueError('Explicit five-command reference contract required')
        for name in ('controller_modifier_audit_v2.json','adaptive_reference_audit.json'):
            report=json.loads((experiment/name).read_text())
            if not report['passed'] or report['manifest_sha256']!=sha(experiment/'manifest.json') or report['summary_sha256']!=sha(experiment/'summary.json'):
                raise ValueError('Reference audit mismatch: '+name)
        if not json.loads((experiment/'composite_prefix_audit_v2.json').read_text())['passed']:
            raise ValueError('Frozen-prefix audit required')
        if m.get('policy_adapter')!='closure-action-bias-v1':raise ValueError('Explicit action-bias adapter required')
        model=ClosureCompositePolicy(model,m)
    except Exception:
        observer.close();raise
    return base,observer,model,m,s


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=Path,required=True);parser.add_argument('--case',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--viewer',action='store_true',help='Open a passive MuJoCo viewer; no robot connection')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1);base,observer,model,m,s=load_verified(args.experiment,args.case)
    args.output.mkdir(parents=True)
    provenance=dict(experiment=str(args.experiment.resolve()),checkpoint_sha256=sha(args.experiment/'policy.zip'),
        experiment_manifest_sha256=sha(args.experiment/'manifest.json'),integrity_audit_sha256=sha(args.experiment/'independent_candidate_audit.json'),
        runner_sha256=sha(__file__),contract=m['contract'],observation_contract=OBSERVATION_CONTRACT,geometry_mode=m['geometry_mode'],
        case=args.case,offsets=m['evaluation_cases'][args.case],simulation_only=True,
        policy_adapter=m['policy_adapter'],closure_action_bias=m['closure_action_bias'],controller_modifier=m['controller_modifier'],prefix_checkpoint_sha256=m['prefix_checkpoint_sha256'],scope='Composite actor plus declared bounded knee/ankle action bias and fixed reference commands in MuJoCo only; fixed calibration, not new PPO learning. Oracle geometry, scripted reference and privileged contact controller remain active. No hardware connection or default policy replacement.')
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2));(args.output/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    obs,info=observer.reset(seed=20261022,options={'case':args.case});e=base.env
    modifier=m['controller_modifier'];gain=float(np.clip((float(obs[138])-.625)/.125,0.,1.))
    waist=install_waist_target(base,modifier['waist_bias'],geometry_gain=gain,ramp_seconds=modifier['waist_ramp_seconds'],record=True)
    drop=gain*modifier['body_drop_m'];install_body_target(base,drop)
    extra=install_adaptive_reference(base,modifier,geometry_gain=gain,record=True);extra['set_action'](m['adaptive_reference_command'])
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
        path=args.output/'trajectory.npz';np.savez_compressed(path,qpos=q,qvel=v,time_s=times,actions=actions,observations=observations,torques=torques,rewards=rewards,waist_reference_trace=np.asarray(waist['trace']).reshape(-1,10),body_reference_trace=np.array([body_target_offset(t,drop) for t in times]),controller_geometry_gain=np.array(gain),adaptive_reference_trace=np.asarray(extra['trace']).reshape(-1,21))
        endpoints=audit_endpoints(e,dict(qpos=np.asarray(q),qvel=np.asarray(v),time_s=np.asarray(times)),base.plan)
        result=dict(runtime=info,independently_accepted=bool(info['success'] and all(r['passed'] for r in endpoints)),independent_endpoints=endpoints,
            wall_seconds=time.perf_counter()-started,trajectory_sha256=sha(path),simulation_only=True,autonomous_climbing=False)
        parent_record=next(row for row in s['after'] if row['case']==args.case)
        parent_path=args.experiment/f'after_{args.case}.npz'
        if sha(parent_path)!=parent_record['trajectory_sha256']:raise ValueError('Audited parent trajectory changed')
        with np.load(parent_path) as original,np.load(path) as reproduced:
            checks={key:bool(np.array_equal(original[key],reproduced[key])) for key in ('qpos','qvel','time_s','observations','actions','torques','rewards','waist_reference_trace','body_reference_trace','adaptive_reference_trace')}
        result.update(exact_reproduction=all(checks.values()),exact_checks=checks,parent_trajectory_sha256=sha(parent_path),controller_geometry_gain=gain)
        (args.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        if not args.viewer and not result['exact_reproduction']:raise AssertionError('Headless runner differs from audited evaluation')
    finally:observer.close()


if __name__=='__main__':main()
