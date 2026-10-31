"""One predeclared new-reference pilot using the existing frozen controller.

Adapted from evaluate_compiled_hook_pair; no change to physical acceptance,
policy replay, first-cycle reproduction, torque authority or telemetry.
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
from training.compiled_rung_geometry import CompiledPhysicalV2CycleEnv as PhysicalV2CycleEnv
from training.physics_telemetry import install_physics_telemetry
from training.probe_hook_loading import install_hook_admission, ORIGINAL_COMMAND_SHA256
import subprocess
from training.cycle_relative_bar_observation import CycleRelativeBarObservation
from training.closure_action_bias_policy_v2 import ClosureActionBiasV2Policy
from training.closure_multicycle_reference_control import (
    install_waist_target,install_body_target,install_adaptive_reference,body_target_offset,handoff_fade)
from tools.validate_swing_reference import ROOT,read,write,sha
from tools.compare_audited_closure_candidates import load_run


EVALUATOR_ORIGIN_SHA256='b522149df6a2a22a2b5ed22cc0f19fbc4edb58e311cb76afccaf49abe3c45c65'


def main():
    p=argparse.ArgumentParser()
    for name in ('candidate','plan','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--experiment-protocol',type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--arm',choices=('adopted_v2','contact_confirmed'),required=True)
    a=p.parse_args();torch.set_num_threads(1)
    experiment=read(a.experiment_protocol)
    assert a.case==experiment['case'] and a.arm==experiment['arm']=='adopted_v2'
    assert a.candidate.resolve()==Path(experiment['candidate']).resolve()
    assert a.plan.resolve()==Path(experiment['plan']).resolve() and sha(a.plan)==experiment['plan_sha256']
    assert a.output.resolve()==Path(experiment['output']).resolve()
    assert sha(a.plan.parent/'manifest.json')==experiment['reference_manifest_sha256']
    assert sha(a.plan.parent/'audit.json')==experiment['reference_audit_sha256']
    parent=a.candidate.resolve();m=read(parent/'manifest.json');verified=load_run(parent)
    chain=read(parent/'reference_chain_audit.json');physical=read(parent/'physical_v2_audit.json')
    # The first-cycle gate is per case. Failed first cycles stay in the suite
    # denominator, but do not block other cases or condition candidate runs on
    # the outcome of the baseline's SECOND cycle.
    compiled=read(parent/'compiled_reference_chain_audit.json')
    assert chain['passed'] and compiled['passed']
    assert compiled['base_chain_audit_sha256']==sha(parent/'reference_chain_audit.json')
    assert compiled['manifest_sha256']==sha(parent/'manifest.json')
    assert next(r for r in compiled['rows'] if r['case']==a.case)['passed']
    assert next(r for r in chain['prefix_rows'] if r['case']==a.case)['passed']
    hook=read(parent/'hook_uncertainty_protocol.json')
    assert hook['cases']==m['evaluation_cases']
    assert sha(parent/'hook_uncertainty_protocol.json')==m['hook_protocol_sha256']
    assert a.case in hook['cases']
    assert chain['manifest_sha256']==sha(parent/'manifest.json') and chain['summary_sha256']==sha(parent/'summary.json')
    assert sha(parent/'physical_v2_audit.json')==chain['audits']['physical_v2_audit.json']
    assert next(r for r in physical['rows'] if r['case']==a.case)['physical_v2_passed']
    pm=read(a.plan.parent/'manifest.json');ref=dict(np.load(a.plan))
    assert pm['plan_sha256']==sha(a.plan) and read(a.plan.parent/'audit.json')['passed']
    ends=ref['cycle_ends_s'];assert len(ends)==2, 'This paired protocol permits two cycles only'
    first_manifest=pm if len(ends)==2 else read(Path(pm['bridge_reference']).parent/'manifest.json')
    assert first_manifest['first_reference_sha256']==m['reference_sha256']
    assert all(pm['first_cycle_exact'].values())
    two=None;gate=None
    for rel,digest in m['sources'].items():
        assert sha(parent/'source_snapshot'/rel)==digest
        if rel.replace('\\','/').startswith('training/'):assert sha(ROOT/rel)==digest
    env=PhysicalV2CycleEnv(plan=a.plan,cases={a.case:m['evaluation_cases'][a.case]},
        normal_bias_y_deg=m['normal_bias_y_deg'],memory_enabled=m['force_history_memory'])
    observer=CycleRelativeBarObservation(env,mode=m['geometry_mode'])
    model=ClosureActionBiasV2Policy(PPO.load(parent/'policy.zip',device='cpu'),m)
    raw,_=observer.reset(seed=20261022,options={'case':a.case})
    telemetry=install_physics_telemetry(env)
    admission=install_hook_admission(env,**hook['controller_parameters']) if a.arm=='contact_confirmed' else None
    c=m['controller_modifier'];gain=float(np.clip((float(raw[138])-.625)/.125,0,1));fade=pm['blend_seconds']
    waist=install_waist_target(env,c['waist_bias'],geometry_gain=gain,ramp_seconds=c['waist_ramp_seconds'],record=True,fade_seconds=fade)
    drop=gain*c['body_drop_m'];install_body_target(env,drop,fade_seconds=fade)
    extra=install_adaptive_reference(env,c,geometry_gain=gain,record=True,fade_seconds=fade)
    extra['set_action'](m['adaptive_reference_command'])
    original=dict(np.load(parent/f'after_{a.case}.npz'))
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
    manifest=dict(parent_experiment=str(parent),parent_chain_audit_sha256=sha(parent/'reference_chain_audit.json'),
        parent_manifest_sha256=sha(parent/'manifest.json'),checkpoint_sha256=sha(parent/'policy.zip'),
        prefix_checkpoint_sha256=m['prefix_checkpoint_sha256'],plan_sha256=sha(a.plan),plan_path=str(a.plan.resolve()),
        plan_manifest_sha256=sha(a.plan.parent/'manifest.json'),bundle=str(env.env.bundle),bundle_manifest_sha256=sha(env.env.bundle/'manifest.json'),
        sources=sources,case=a.case,offsets=m['evaluation_cases'][a.case],geometry_observation_roles=observer.role_rows,
        fade_seconds=fade,cycles=len(ends),acceptance='physical-v2',new_training_steps=0,
        scope='First cycle exactly matches adopted single-cycle physics; all cycles use the same frozen composite actor with local clock and privileged bar roles. First seam blends carried actions for six seconds; first-cycle reference modifiers fade out. All later references also include the 8 mm swing lift. No physical state or motor-filter reset at seams. Endpoint acceptance is v2; legacy closure gate is reported only.')
    manifest.update(scene_compiler=m['scene_compiler'],arm=a.arm,
        scene_geometry_sha256=env.variant['geometry_sha256'],
        compiled_parent_audit_sha256=sha(parent/'compiled_reference_chain_audit.json'),
        hook_protocol_sha256=sha(parent/'hook_uncertainty_protocol.json'),
        evaluator_origin_sha256=EVALUATOR_ORIGIN_SHA256,
        reference_retarget_protocol=str(a.experiment_protocol.resolve()),
        reference_retarget_protocol_sha256=sha(a.experiment_protocol),
        experimental_reference=True,automatic_promotion=False,
        scope='Single predeclared whole-body hand-reference pilot. NEW experimental reference; unchanged adopted controller and frozen policy. Same compiled scene and exact first-cycle reproduction required. Not part of the frozen three-arm grid. No training or automatic promotion.')
    if admission is not None:
        protocol=dict(parameters=hook['controller_parameters'],status='Development candidate; not adopted',
            original_qp_command_sha256=ORIGINAL_COMMAND_SHA256,source_sha256=sha(__file__),
            start='Only after the first complete cycle',new_training_steps=0,
            baseline_run=None,baseline_success_required=False,
            scope='Identical predeclared contact-confirmed probe configuration. Actual force feedback is privileged simulation feedback. No impact-force bound or hardware claim.')
        write(out/'hook_admission_protocol.json',protocol)
        manifest.update(contact_admission=protocol,hook_admission_protocol_sha256=sha(out/'hook_admission_protocol.json'))
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
                old_success=next(r for r in read(parent/'after.json') if r['case']==a.case)['success']
                exact['terminal_reward_accounted']=bool(abs(reward-(original['rewards'][-1]-(20 if old_success else -20)))<1e-9)
                prefixes['first_cycle']=exact;write(out/'first_cycle_prefix.json',dict(passed=all(exact.values()),checks=exact))
                if not all(exact.values()):info.update(success=False,reason='first_cycle_prefix_mismatch');break
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
    finally:
        # Preserve full-rate telemetry even for controller timeouts/exceptions;
        # an interrupted trajectory is diagnostic, never an accepted rollout.
        np.savez_compressed(out/'physics_telemetry.npz',**{k:v[:telemetry['count']] for k,v in telemetry['data'].items()})
        record=dict(telemetry_sha256=sha(out/'physics_telemetry.npz'),physics_samples=telemetry['count'],
            groups=telemetry['groups'],manifest_sha256=sha(out/'manifest.json'),
            trajectory_sha256=sha(out/'trajectory.npz') if (out/'trajectory.npz').exists() else None,
            interrupted_trajectory_sha256=sha(out/'interrupted_trajectory.npz') if (out/'interrupted_trajectory.npz').exists() else None)
        write(out/'physics_telemetry_record.json',record)
        if admission is not None:
            write(out/'hook_admission_trace.json',admission['trace'])
            write(out/'hook_admission_record.json',dict(record,trace_sha256=sha(out/'hook_admission_trace.json'),
                protocol_sha256=sha(out/'hook_admission_protocol.json')))
        env.close()
    subprocess.run([sys.executable,'-m','tools.audit_reference_retarget_pilot',str(out)],cwd=ROOT,check=True)


if __name__=='__main__':main()
