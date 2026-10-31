"""Run fresh v2 multi-cycle physics from a relocatable frozen simulation package."""
import argparse
from contextlib import ExitStack
import hashlib
from importlib import metadata
import json
from pathlib import Path
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_package(root):
    root=Path(root).resolve();m=json.loads((root/'release_manifest.json').read_text())
    if m['format']!='g1-physical-v2-multicycle' or not m['simulation_only']:raise ValueError('Unsupported package')
    for name,digest in m['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or sha(path)!=digest:raise ValueError('Package integrity mismatch: '+name)
    for name,version in m['runtime_versions'].items():
        if metadata.version(name)!=version:raise ValueError('Runtime version differs: '+name)
    for name in ('reference','checkpoint','golden_trajectory'):
        if not (root/m[name]).resolve().is_relative_to(root):raise ValueError('Package path must stay local')
    if not (root/m['composite']['prefix_checkpoint']).resolve().is_relative_to(root):raise ValueError('Prefix must stay local')
    audit=json.loads((root/'evidence/v2_multicycle_audit.json').read_text())
    assert audit['passed'] and audit['trajectory_sha256']==sha(root/m['golden_trajectory'])
    assert audit['manifest_sha256']==sha(root/'evidence/manifest.json')
    assert audit['summary_sha256']==sha(root/'evidence/summary.json')
    assert audit['independent_audit_sha256']==sha(root/'evidence/independent_audit.json')
    assert audit['policy_audit_sha256']==sha(root/'evidence/policy_transfer_audit.json')
    return m


def create_runtime(root,m):
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from training.physical_v2_cycle_env import PhysicalV2CycleEnv
    from training.cycle_relative_bar_observation import CycleRelativeBarObservation
    from training.closure_action_bias_policy_v2 import ClosureActionBiasV2Policy
    from training.closure_multicycle_reference_control import install_waist_target,install_body_target,install_adaptive_reference
    torch.set_num_threads(1)
    base=PhysicalV2CycleEnv(plan=root/m['reference'],cases={m['case']:m['offsets']},
        normal_bias_y_deg=m['normal_bias_y_deg'],memory_enabled=m['force_history_memory'])
    observer=CycleRelativeBarObservation(base,mode=m['geometry_mode'])
    composite=dict(m['composite']);composite['prefix_checkpoint']=str(root/composite['prefix_checkpoint'])
    actor=ClosureActionBiasV2Policy(PPO.load(root/m['checkpoint'],device='cpu'),composite)
    raw,_=observer.reset(seed=20261022,options={'case':m['case']})
    modifier=composite['controller_modifier'];gain=float(np.clip((float(raw[138])-.625)/.125,0,1))
    fade=m['fade_seconds'];waist=install_waist_target(base,modifier['waist_bias'],geometry_gain=gain,
        ramp_seconds=modifier['waist_ramp_seconds'],record=True,fade_seconds=fade)
    drop=gain*modifier['body_drop_m'];install_body_target(base,drop,fade_seconds=fade)
    extra=install_adaptive_reference(base,modifier,geometry_gain=gain,record=True,fade_seconds=fade)
    extra['set_action'](composite['adaptive_reference_command'])
    return base,observer,actor,raw,waist,extra,drop


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--viewer',action='store_true')
    p.add_argument('--verify-only',action='store_true');a=p.parse_args();root=Path(__file__).resolve().parents[1]
    m=verify_package(root)
    if a.verify_only:
        print(json.dumps(dict(verified=True,files=len(m['files']),case=m['case'],cycles=m['cycles'],package_root=str(root))),flush=True);return
    if a.output is None:p.error('--output required unless --verify-only')
    if a.output.exists():raise FileExistsError(a.output)
    import numpy as np
    from tools.audit_multirung_cycle import audit_endpoints
    from training.closure_multicycle_reference_control import body_target_offset,handoff_fade
    base,observer,actor,raw,waist,extra,drop=create_runtime(root,m)
    e=base.env;ends=base.ends;fade=m['fade_seconds'];out=a.output.resolve();out.mkdir(parents=True)
    records=dict(qpos=[e.data.qpos.copy()],qvel=[e.data.qvel.copy()],time_s=[0.],actions=[],
        observations=[],raw_observations=[],torques=[],rewards=[],qacc_50hz=[])
    boundary=round(float(ends[0])/.02);carried=None;started=time.perf_counter();info=None
    try:
        with ExitStack() as stack:
            viewer=None
            if a.viewer:
                import mujoco.viewer
                viewer=stack.enter_context(mujoco.viewer.launch_passive(e.model,e.data))
                viewer.opt.geomgroup[3],viewer.opt.geomgroup[4]=0,1
                viewer.cam.azimuth,viewer.cam.elevation,viewer.cam.distance=65.,-8.,2.5
            while True:
                if viewer is not None and not viewer.is_running():
                    info=base.info('viewer_closed',False);break
                step_started=time.perf_counter();t=float(e.data.time)
                index=min(int(np.searchsorted(ends,t,side='right')),len(ends)-1)
                start=0. if index==0 else float(ends[index-1]);obs=raw.copy()
                obs[105]=max(0.,1.-(t-start)/(float(ends[index])-start))
                action=actor.predict(obs,deterministic=True)[0]
                if len(records['actions'])<boundary:carried=action.copy()
                else:
                    weight=handoff_fade(t,float(ends[0]),fade)[0]
                    action=(weight*carried+(1.-weight)*action).astype(np.float32)
                records['observations'].append(obs.copy());records['raw_observations'].append(raw.copy())
                raw,reward,done,truncated,info=observer.step(action)
                for name,value in dict(qpos=e.data.qpos.copy(),qvel=e.data.qvel.copy(),time_s=float(e.data.time),
                    actions=action.copy(),torques=base.last_torque.copy(),rewards=float(reward),qacc_50hz=e.data.qacc.copy()).items():records[name].append(value)
                if viewer is not None:
                    viewer.cam.lookat[:]=e.data.xpos[e.pelvis]+np.array([0.,0.,.05])
                    viewer.set_texts((None,None,'LIVE MUJOCO - frozen PPO + scripted reference\nTime / completed cycles',
                        f'No hardware connection\n{e.data.time:.2f} s / {len(base.cycle_results)} of {len(ends)}'))
                    viewer.sync();time.sleep(max(0.,.02-(time.perf_counter()-step_started)))
                if len(records['actions'])%1000==0:print(json.dumps(dict(time_s=info['time_s'],completed_cycles=len(base.cycle_results))),flush=True)
                if done or truncated:break
        arrays={k:np.asarray(v) for k,v in records.items()}
        arrays.update(waist_reference_trace=np.asarray(waist['trace']).reshape(-1,10),adaptive_reference_trace=np.asarray(extra['trace']).reshape(-1,21),
            body_reference_trace=np.asarray([body_target_offset(t,drop,first_end=float(ends[0]),fade_seconds=fade) for t in records['time_s']]))
        np.savez_compressed(out/'trajectory.npz',**arrays)
        equal={};differences={}
        with np.load(root/m['golden_trajectory']) as golden:
            assert set(arrays)==set(golden.files)
            for key,value in arrays.items():
                same_shape=value.shape==golden[key].shape;equal[key]=bool(same_shape and np.array_equal(value,golden[key]))
                differences[key]=float(abs(value-golden[key]).max(initial=0)) if same_shape else None
        golden_summary=json.loads((root/'evidence/summary.json').read_text())
        golden_tail=json.loads((root/'evidence/endpoint_samples.json').read_text())
        tail_exact=base.v2_endpoint_records==golden_tail
        runtime_exact=all(info[k]==golden_summary[k] for k in ('reason','success','peaks','minimum_native_joint_margin_rad','cycle_results','physical_reset_count'))
        endpoints=audit_endpoints(e,arrays,base.plan)
        settled=all(all(row[k] is not None and row[k]<limit for k,limit in [('effector_error_m',.01),('tilt_deg',3),
            ('linear_speed_m_s',.01),('angular_speed_rad_s',.1),('joint_speed_rad_s',.1)]) for row in endpoints)
        total=float(arrays['qpos'][-1,2]-arrays['qpos'][0,2]);rise_ok=abs(total-.25*len(ends))<.01
        passed=bool(info['success'] and settled and rise_ok and tail_exact and runtime_exact and all(equal.values()))
        report=dict(passed=passed,runtime=info,exact_arrays=equal,maximum_differences=differences,endpoint_contact_records_exact=tail_exact,
            runtime_metrics_exact=runtime_exact,independent_endpoint_posture_passed=settled,independent_endpoints=endpoints,
            final_rise_m=total,total_rise_passed=rise_ok,trajectory_sha256=sha(out/'trajectory.npz'),
            release_manifest_sha256=sha(root/'release_manifest.json'),golden_audit_sha256=m['original_audit_sha256'],
            package_root=str(root),wall_seconds=time.perf_counter()-started,new_training_steps=0,simulation_only=True,
            scope='Fresh continuous dynamics and frozen-policy inference in the relocated package. Every saved state/action/reference/reward array and endpoint contact record must equal the accepted run. Historical strict closure verdicts inside endpoint rows are informational; physical-v2 acceptance applies. No perception or hardware claim.')
        (out/'reproduction_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        print(json.dumps({k:v for k,v in report.items() if k not in ('runtime','independent_endpoints')}),flush=True)
        if not passed:raise AssertionError('Portable run did not exactly reproduce the accepted record')
    finally:observer.close()


if __name__=='__main__':main()
