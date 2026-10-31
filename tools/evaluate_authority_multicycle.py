"""Frozen calibrated PPO transfer with adopted v2 endpoint acceptance.

First-cycle physics must reproduce the adopted single-cycle record exactly.
No state restoration or training. Explicit changed residual authority; same-authority prefix required.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from training.physical_v2_cycle_env import PhysicalV2CycleEnv
from training.cycle_relative_bar_observation import CycleRelativeBarObservation
from training.closure_action_bias_policy_v2 import ClosureActionBiasV2Policy
from training.closure_multicycle_reference_control import (
    install_waist_target,install_body_target,install_adaptive_reference,body_target_offset,handoff_fade)
from tools.validate_swing_reference import ROOT,read,write,sha
from tools.compare_audited_closure_candidates import load_run


from tools.authority_intervention import install

def main():
    p=argparse.ArgumentParser()
    for name in ('candidate','plan','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--two-cycle-gate',type=Path)
    p.add_argument('--single-cycle-record',type=Path,required=True)
    p.add_argument('--authority',type=float,default=.01875)
    a=p.parse_args();torch.set_num_threads(1)
    install(a.authority)
    single=a.single_cycle_record.resolve()
    single_result=read(single/'result.json')
    assert single_result['case']==a.case and single_result['success']
    assert sha(single/f'after_{a.case}.npz')==single_result['trajectory_sha256']
    parent=a.candidate.resolve();m=read(parent/'manifest.json');verified=load_run(parent)
    chain=read(parent/'reference_chain_audit.json');physical=read(parent/'physical_v2_audit.json')
    assert chain['passed'] and chain['eligible_single_cycle_reference']
    assert chain['manifest_sha256']==sha(parent/'manifest.json') and chain['summary_sha256']==sha(parent/'summary.json')
    assert sha(parent/'physical_v2_audit.json')==chain['audits']['physical_v2_audit.json']
    assert next(r for r in physical['rows'] if r['case']==a.case)['physical_v2_passed']
    pm=read(a.plan.parent/'manifest.json');ref=dict(np.load(a.plan))
    assert pm['plan_sha256']==sha(a.plan) and read(a.plan.parent/'audit.json')['passed']
    ends=ref['cycle_ends_s'];assert len(ends) in (2,8)
    first_manifest=pm if len(ends)==2 else read(Path(pm['bridge_reference']).parent/'manifest.json')
    assert first_manifest['first_reference_sha256']==m['reference_sha256']
    assert all(pm['first_cycle_exact'].values())
    two=None;gate=None
    if len(ends)>2:
        assert a.two_cycle_gate is not None
        gate=read(a.two_cycle_gate/'v2_multicycle_audit.json');gm=read(a.two_cycle_gate/'manifest.json')
        assert gate['passed'] and gm['residual_authority_rad']==a.authority and gm['case']==a.case and gm['parent_experiment']==str(parent)
        assert gate['trajectory_sha256']==sha(a.two_cycle_gate/'trajectory.npz')
        assert gm['plan_sha256']==pm['bridge_sha256'] and all(pm['first_two_cycles_exact'].values())
        two=dict(np.load(a.two_cycle_gate/'trajectory.npz'))
    for rel,digest in m['sources'].items():
        assert sha(parent/'source_snapshot'/rel)==digest
        if rel.replace('\\','/').startswith('training/'):assert sha(ROOT/rel)==digest
    env=PhysicalV2CycleEnv(plan=a.plan,cases={a.case:m['evaluation_cases'][a.case]},
        normal_bias_y_deg=m['normal_bias_y_deg'],memory_enabled=m['force_history_memory'])
    observer=CycleRelativeBarObservation(env,mode=m['geometry_mode'])
    model=ClosureActionBiasV2Policy(PPO.load(parent/'policy.zip',device='cpu'),m)
    raw,_=observer.reset(seed=20261022,options={'case':a.case})
    c=m['controller_modifier'];gain=float(np.clip((float(raw[138])-.625)/.125,0,1));fade=pm['blend_seconds']
    waist=install_waist_target(env,c['waist_bias'],geometry_gain=gain,ramp_seconds=c['waist_ramp_seconds'],record=True,fade_seconds=fade)
    drop=gain*c['body_drop_m'];install_body_target(env,drop,fade_seconds=fade)
    extra=install_adaptive_reference(env,c,geometry_gain=gain,record=True,fade_seconds=fade)
    extra['set_action'](m['adaptive_reference_command'])
    original=dict(np.load(single/f'after_{a.case}.npz'))
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(a.plan,out/'reference.npz')
    sources={}
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if not file:continue
        path=Path(file).resolve()
        if path.suffix=='.py' and path.is_relative_to(ROOT):
            rel=path.relative_to(ROOT)
            if rel.parts[0] in ('training','tools'):sources[rel.as_posix()]=sha(path)
    for rel,digest in sources.items():
        dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
        assert sha(dest)==digest
    manifest=dict(residual_authority_rad=a.authority,single_cycle_record=str(single),single_cycle_result_sha256=sha(single/'result.json'),single_cycle_trajectory_sha256=sha(single/f'after_{a.case}.npz'),parent_experiment=str(parent),parent_chain_audit_sha256=sha(parent/'reference_chain_audit.json'),
        parent_manifest_sha256=sha(parent/'manifest.json'),checkpoint_sha256=sha(parent/'policy.zip'),
        prefix_checkpoint_sha256=m['prefix_checkpoint_sha256'],plan_sha256=sha(a.plan),plan_path=str(a.plan.resolve()),
        plan_manifest_sha256=sha(a.plan.parent/'manifest.json'),bundle=str(env.env.bundle),bundle_manifest_sha256=sha(env.env.bundle/'manifest.json'),
        sources=sources,case=a.case,offsets=m['evaluation_cases'][a.case],geometry_observation_roles=observer.role_rows,
        fade_seconds=fade,cycles=len(ends),acceptance='physical-v2',new_training_steps=0,
        scope='Authority intervention: first cycle exactly matches the separately recorded same-authority single cycle; all cycles use the same frozen composite actor with local clock and privileged bar roles. First seam blends carried actions for six seconds; first-cycle reference modifiers fade out. All later references also include the 8 mm swing lift. No physical state or motor-filter reset at seams. Endpoint acceptance is v2; legacy closure gate is reported only.')
    if gate:manifest.update(two_cycle_gate=str(a.two_cycle_gate.resolve()),two_cycle_audit_sha256=sha(a.two_cycle_gate/'v2_multicycle_audit.json'))
    write(out/'manifest.json',manifest);write(out/'scene_variant.json',env.variant)
    records=dict(qpos=[env.env.data.qpos.copy()],qvel=[env.env.data.qvel.copy()],time_s=[0.],
        actions=[],observations=[],raw_observations=[],torques=[],rewards=[],qacc_50hz=[])
    prefixes={};carried=None;boundary=round(float(ends[0])/.02);second_boundary=round(float(ends[1])/.02)
    started=time.perf_counter();last_cycles=0
    try:
        while True:
            t=float(env.env.data.time);idx=min(int(np.searchsorted(ends,t,side='right')),len(ends)-1)
            start=0. if idx==0 else float(ends[idx-1]);obs=raw.copy()
            obs[105]=max(0.,1.-(t-start)/(float(ends[idx])-start))
            action=model.predict(obs,deterministic=True)[0]
            if len(records['actions'])<boundary:carried=action.copy()
            else:
                weight=handoff_fade(t,float(ends[0]),fade)[0]
                action=(weight*carried+(1.-weight)*action).astype(np.float32)
            records['observations'].append(obs.copy());records['raw_observations'].append(raw.copy())
            raw,reward,done,truncated,info=observer.step(action)
            for k,v in dict(qpos=env.env.data.qpos.copy(),qvel=env.env.data.qvel.copy(),time_s=float(env.env.data.time),
                actions=action.copy(),torques=env.last_torque.copy(),rewards=float(reward),qacc_50hz=env.env.data.qacc.copy()).items():records[k].append(v)
            count=len(records['actions'])
            if count==boundary:
                exact={k:bool(np.array_equal(records[k],original[k])) for k in ('qpos','qvel','time_s','observations','actions','torques')}
                for k,values in [('waist_reference_trace',waist['trace']),('adaptive_reference_trace',extra['trace'])]:
                    exact[k]=bool(np.array_equal(np.asarray(values).reshape(-1,original[k].shape[1]),original[k]))
                exact['rewards_before_terminal']=bool(np.array_equal(records['rewards'][:-1],original['rewards'][:-1]))
                old_success=single_result['success']
                exact['terminal_reward_accounted']=bool(abs(reward-(original['rewards'][-1]-(20 if old_success else -20)))<1e-9)
                prefixes['first_cycle']=exact;write(out/'first_cycle_prefix.json',dict(passed=all(exact.values()),checks=exact))
                if not all(exact.values()):info.update(success=False,reason='first_cycle_prefix_mismatch');break
            if two is not None and count==second_boundary:
                exact={k:bool(np.array_equal(records[k],two[k])) for k in ('qpos','qvel','time_s','observations','actions','torques','qacc_50hz')}
                # Raw feature 105 is the whole-episode clock, so extending the
                # horizon must change it. The actor's cycle-local clock above
                # and every other raw feature must still reproduce exactly.
                raw_prefix=np.asarray(records['raw_observations'])
                exact['raw_observations_except_horizon_clock']=bool(np.array_equal(np.delete(raw_prefix,105,axis=1),np.delete(two['raw_observations'],105,axis=1)))
                exact['raw_horizon_clock_reconstructed']=bool(np.array_equal(raw_prefix[:,105],np.maximum(0.,1.-np.asarray(records['time_s'][:-1])/float(ends[-1])).astype(raw_prefix.dtype)))
                exact['gate_raw_horizon_clock_reconstructed']=bool(np.array_equal(two['raw_observations'][:,105],np.maximum(0.,1.-two['time_s'][:-1]/float(ends[1])).astype(two['raw_observations'].dtype)))
                exact['rewards_before_terminal']=bool(np.array_equal(records['rewards'][:-1],two['rewards'][:-1]))
                exact['terminal_reward_accounted']=bool(abs(reward-(two['rewards'][-1]-20))<1e-9)
                prefixes['two_cycles']=exact;write(out/'two_cycle_prefix.json',dict(passed=all(exact.values()),checks=exact))
                if not all(exact.values()):info.update(success=False,reason='two_cycle_prefix_mismatch');break
            if len(env.cycle_results)!=last_cycles:
                last_cycles=len(env.cycle_results);print(json.dumps(env.cycle_results[-1]),flush=True)
            if count%1000==0:print(json.dumps(dict(time_s=info['time_s'],steps=count)),flush=True)
            if done or truncated:break
        arrays={k:np.asarray(v) for k,v in records.items()}
        arrays.update(waist_reference_trace=np.asarray(waist['trace']).reshape(-1,10),
            adaptive_reference_trace=np.asarray(extra['trace']).reshape(-1,21),
            body_reference_trace=np.asarray([body_target_offset(t,drop,first_end=float(ends[0]),fade_seconds=fade) for t in records['time_s']]))
        np.savez_compressed(out/'trajectory.npz',**arrays)
        write(out/'endpoint_samples.json',env.v2_endpoint_records)
        prefix_ok=bool(prefixes) and all(all(c.values()) for c in prefixes.values()) and (two is None or 'two_cycles' in prefixes)
        rise=float(arrays['qpos'][-1,2]-arrays['qpos'][0,2]);total_ok=abs(rise-.25*len(ends))<.01
        summary=dict(info,passed=bool(info['success'] and prefix_ok and total_ok),first_cycle_prefix_exact=bool(prefixes.get('first_cycle') and all(prefixes['first_cycle'].values())),
            prefix_checks=prefixes,final_rise_m=rise,total_rise_within_10mm=total_ok,trajectory_sha256=sha(out/'trajectory.npz'),
            endpoint_samples_sha256=sha(out/'endpoint_samples.json'),wall_seconds=time.perf_counter()-started,
            sources_unchanged=all(sha(ROOT/k)==v for k,v in sources.items()))
        write(out/'summary.json',summary)
        print(json.dumps({k:v for k,v in summary.items() if k not in ('prefix_checks','cycle_results')}),flush=True)
    except Exception as exc:
        np.savez_compressed(out/'interrupted_trajectory.npz',**{k:np.asarray(v) for k,v in records.items()})
        write(out/'interrupted.json',dict(error=type(exc).__name__,message=str(exc),time_s=float(env.env.data.time),
            scope='Interrupted diagnostic record only; not an accepted run'))
        raise
    finally:env.close()


if __name__=='__main__':main()
