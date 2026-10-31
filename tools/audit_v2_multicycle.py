"""Independent recorded-state v2 audit, then frozen-policy transfer replay.

Native 2000 Hz checks and contact forces are runtime evidence. Saved kinematics,
terminal metrics, rewards, controller commands and actor actions are rebuilt.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import mujoco
import numpy as np
from training.physical_v2_cycle_env import PhysicalV2CycleEnv
from training.closure_multicycle_reference_control import body_target_offset
from tools.audit_multirung_cycle import audit_endpoints
from tools.validate_swing_reference import read,write,sha,ROOT


def write(path,data):
    """Normalize NumPy scalar metrics without changing their numeric values."""
    def native(value):
        if isinstance(value,np.generic):return value.item()
        raise TypeError(f'Unsupported report value: {type(value).__name__}')
    Path(path).write_text(json.dumps(data,indent=2,default=native),encoding='utf-8')


def audit(out):
    out=Path(out).resolve();m=read(out/'manifest.json');s=read(out/'summary.json');parent=Path(m['parent_experiment']);pm=read(parent/'manifest.json')
    assert sha(out/'trajectory.npz')==s['trajectory_sha256'] and sha(out/'endpoint_samples.json')==s['endpoint_samples_sha256']
    assert sha(out/'reference.npz')==m['plan_sha256']==sha(m['plan_path'])
    assert sha(parent/'manifest.json')==m['parent_manifest_sha256'] and sha(parent/'reference_chain_audit.json')==m['parent_chain_audit_sha256']
    assert sha(parent/'policy.zip')==m['checkpoint_sha256']
    assert all(sha(out/'source_snapshot'/k)==v for k,v in m['sources'].items())
    assert all(sha(ROOT/k)==v for k,v in m['sources'].items() if k.startswith('training/'))
    z=dict(np.load(out/'trajectory.npz'));ref=dict(np.load(out/'reference.npz'));ends=ref['cycle_ends_s'];n=len(z['actions'])
    env=PhysicalV2CycleEnv(plan=m['plan_path'],cases={m['case']:m['offsets']},normal_bias_y_deg=pm['normal_bias_y_deg'],memory_enabled=pm['force_history_memory'])
    env.reset(options={'case':m['case']});e=env.env
    checks=dict(finite=all(np.isfinite(v).all() for v in z.values()),
        timing=bool(len(z['time_s'])==n+1 and abs(z['time_s'][0])<1e-12 and np.max(abs(np.diff(z['time_s'])[:-1]-.02),initial=0)<1e-8 and 0<np.diff(z['time_s'])[-1]<=.02000001),
        dimensions=z['qpos'].shape==(n+1,e.model.nq) and z['qvel'].shape==(n+1,e.model.nv) and z['actions'].shape==(n,26) and z['observations'].shape==(n,148) and z['torques'].shape==(n,29) and z['qacc_50hz'].shape==(n,e.model.nv),
        initial_state=bool(np.array_equal(z['qpos'][0],e.data.qpos) and np.array_equal(z['qvel'][0],e.data.qvel)),
        runtime_reset_once=s['physical_reset_count']==1,
        no_training=m['new_training_steps']==0,
        sources_unchanged=s['sources_unchanged'],
        bundle=sha(e.bundle/'manifest.json')==m['bundle_manifest_sha256'],
        first_cycle_prefix=s['first_cycle_prefix_exact'])
    # Check stored first-cycle comparisons independently, including the terminal
    # reward difference between a single episode and an intermediate boundary.
    with np.load(parent/f"after_{m['case']}.npz") as old:
        count=round(ends[0]/.02)
        checks['first_cycle_arrays_exact']=n>=count and all(np.array_equal(z[k][:count+(k in ('qpos','qvel','time_s'))],old[k]) for k in ('qpos','qvel','time_s','observations','actions','torques'))
        checks['first_cycle_rewards_exact_before_terminal']=n>=count and np.array_equal(z['rewards'][:count-1],old['rewards'][:-1])
    if m.get('two_cycle_gate'):
        gate=Path(m['two_cycle_gate']);g=read(gate/'v2_multicycle_audit.json')
        assert g['passed'] and sha(gate/'v2_multicycle_audit.json')==m['two_cycle_audit_sha256']
        with np.load(gate/'trajectory.npz') as old:
            count=round(ends[1]/.02)
            checks['two_cycle_arrays_exact']=n>=count and all(np.array_equal(z[k][:count+(k in ('qpos','qvel','time_s'))],old[k]) for k in ('qpos','qvel','time_s','observations','actions','torques','qacc_50hz'))
            checks['two_cycle_raw_features_exact']=n>=count and np.array_equal(np.delete(z['raw_observations'][:count],105,axis=1),np.delete(old['raw_observations'],105,axis=1))
            checks['gate_raw_clock_reconstructed']=np.array_equal(old['raw_observations'][:,105],np.maximum(0.,1.-old['time_s'][:-1]/float(ends[1])).astype(old['raw_observations'].dtype))
            checks['two_cycle_rewards_exact_before_terminal']=n>=count and np.array_equal(z['rewards'][:count-1],old['rewards'][:-1])
    checks['raw_horizon_clock_reconstructed']=np.array_equal(z['raw_observations'][:,105],np.maximum(0.,1.-z['time_s'][:-1]/float(ends[-1])).astype(z['raw_observations'].dtype))
    track=nominal_velocity=velocity=reward_error=0.;previous=np.zeros(26)
    for i,t in enumerate(z['time_s'][:-1]):
        q,v,active,_=env.reference(float(t))
        track=max(track,float(abs(z['observations'][i,:29]-(z['qpos'][i,e.qadr]-q[e.qadr])).max()))
        velocity=max(velocity,float(abs(z['observations'][i,29:58]-z['qvel'][i,e.vadr]).max()))
        nominal_velocity=max(nominal_velocity,float(abs(z['observations'][i,70:99]-v[e.vadr]).max()))
        q_end=env.reference(float(z['time_s'][i+1]))[0];action=z['actions'][i].astype(float)
        expected=.02-2*np.mean((z['qpos'][i+1,e.qadr]-q_end[e.qadr])**2)-.001*np.mean(action**2)-.001*np.mean((action-previous)**2)
        if i==n-1 and s['reason'] not in ('first_cycle_prefix_mismatch','two_cycle_prefix_mismatch'):expected+=20 if s['success'] else -20
        reward_error=max(reward_error,abs(expected-z['rewards'][i]));previous=action
    checks.update(observation_tracking=track<1e-7,observation_velocity=velocity<1e-6,nominal_velocity=nominal_velocity<1e-6,reward_reconstruction=reward_error<1e-7)
    margin=float(np.minimum(z['qpos'][:,e.qadr]-e.model.jnt_range[e.joints,0],e.model.jnt_range[e.joints,1]-z['qpos'][:,e.qadr]).min())
    checks['sampled_torque_bounds']=bool(np.all(z['torques']>=e.limits[:,0]-1e-8) and np.all(z['torques']<=e.limits[:,1]+1e-8))
    endpoints=audit_endpoints(e,z,ref);tail=read(out/'endpoint_samples.json');cycles=[]
    for idx,end in enumerate(ends):
        samples=[x for x in tail if x['cycle']==idx+1]
        mask=(z['time_s']>end-2)&(z['time_s']<=end+1e-7);times=z['time_s'][mask]
        ep=endpoints[idx];expected={g:f'rung_{idx+1+(4 if "hook" in g else 0):02d}' for g in ('left_foot','right_foot','left_hook','right_hook')}
        rise=z['qpos'][mask,2]-z['qpos'][0,2]
        coverage=len(samples)==len(times) and len(samples)>=99
        rise_error=max((abs(x['rise_m']-r) for x,r in zip(samples,rise)),default=float('inf'))
        closure_error=abs(max((x['closure_rad'] for x in samples),default=0)-(ep['joint_closure_rad'] or 0))
        loaded_ok=bool(samples) and all(x['loaded_contacts'] and all(c['group'] in expected and c['external_geom']==expected[c['group']] for c in x['loaded_contacts'] if np.linalg.norm(c['force_on_robot_world_n'])>1) for x in samples)
        c=dict(tail_coverage=coverage,times=coverage and bool(np.allclose([x['time_s'] for x in samples],times,rtol=0,atol=1e-10)),
            rise=bool(len(rise)) and float(abs(rise-.25*(idx+1)).max())<.011,
            force_normal=bool(samples) and all(min(x['normal_n'].values())>10 for x in samples),
            force_vertical=bool(samples) and all(min(x['vertical_n'].values())>10 for x in samples),intended_rods=loaded_ok,
            telemetry_rise_matches_state=rise_error<1e-12,telemetry_closure_matches_state=closure_error<1e-12,
            settled=all(ep[k] is not None and ep[k]<limit for k,limit in [('effector_error_m',.01),('tilt_deg',3),('linear_speed_m_s',.01),('angular_speed_rad_s',.1),('joint_speed_rad_s',.1)]))
        cycles.append(dict(cycle=idx+1,passed=all(c.values()),checks=c,independent_endpoint=ep,
            rise_peak_m=float(rise.max()) if len(rise) else None,closure_mrad=(ep['joint_closure_rad'] or 0)*1000,
            min_normal_n=min((min(x['normal_n'].values()) for x in samples),default=None)))
    # Independent reconstruction of filter targets, residuals and derivatives.
    gain=float(np.clip((float(z['observations'][0,138])-.625)/.125,0,1));c=pm['controller_modifier']
    def fade(t):
        if t<=ends[0]+1e-7:return 1.
        u=float(np.clip((t-ends[0])/m['fade_seconds'],0,1));return 1-10*u**3+15*u**4-6*u**5
    controller_error=0.
    for key,dim,scales,bias,start,ramp in [('waist_reference_trace',3,np.full(3,.015),np.array(c['waist_bias']),99.66,c['waist_ramp_seconds']),('adaptive_reference_trace',5,np.array([.003,.003,.003,.00015,.001]),np.array(pm['adaptive_reference_command']),99.66,3.)]:
        trace=z[key];res=np.zeros(dim)
        for row in trace:
            t=row[0];u=float(np.clip((t-start)/ramp,0,1));target=scales*bias*gain*u*u*(3-2*u)*fade(t)
            old=res.copy();res+=.05*(target-res);vel=(res-old)/.005
            expected=np.r_[target,res,vel];actual=row[1:] if dim==3 else row[6:]
            controller_error=max(controller_error,float(abs(actual-expected).max()))
    body_expected=np.array([body_target_offset(t,gain*c['body_drop_m'],first_end=float(ends[0]),fade_seconds=m['fade_seconds']) for t in z['time_s']])
    controller_error=max(controller_error,float(abs(z['body_reference_trace']-body_expected).max()))
    checks['controller_reconstruction']=controller_error<1e-12
    native_ok=margin>=0 and s['minimum_native_joint_margin_rad']>=0
    runtime_ok=bool(s['reason']=='success' and max(s['peaks'][k] for k in ('other','self','flight'))<1 and abs(z['time_s'][-1]-ends[-1])<1e-7)
    total_rise=float(z['qpos'][-1,2]-z['qpos'][0,2]);total_ok=abs(total_rise-.25*len(ends))<.01
    integrity=all(checks.values());task=integrity and native_ok and runtime_ok and total_ok and all(c['passed'] for c in cycles)
    report=dict(passed=task,integrity_passed=integrity,checks=checks,cycle_endpoints=cycles,
        native_bounds_passed=native_ok,runtime_physics_passed=runtime_ok,total_rise_passed=total_ok,final_rise_m=total_rise,
        sampled_native_margin_rad=margin,tracking_error=track,velocity_error=velocity,nominal_velocity_error=nominal_velocity,
        reward_error=reward_error,controller_reconstruction_error=controller_error,
        trajectory_sha256=sha(out/'trajectory.npz'),manifest_sha256=sha(out/'manifest.json'),summary_sha256=sha(out/'summary.json'),
        endpoint_samples_sha256=sha(out/'endpoint_samples.json'),source_sha256=sha(__file__),
        scope='V2 physical/posture acceptance on saved 50 Hz states plus live 2000 Hz runtime checks and hashed tail contact telemetry. Policy replay is separate. Closure remains quality-only; final total rise tolerance is 10 mm without added deadband. No independent re-simulation of the complete dynamics.')
    write(out/'independent_audit.json',report);(out/'v2_independent_audit_source.py').write_bytes(Path(__file__).read_bytes())
    env.close()
    if not (out/'policy_transfer_audit.json').exists():
        with (out/'policy_transfer_audit.log').open('w',encoding='utf-8') as stream:
            subprocess.run([sys.executable,'-m','tools.audit_calibrated_policy_transfer','--run',str(out)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    policy=read(out/'policy_transfer_audit.json')
    assert policy['trajectory_sha256']==report['trajectory_sha256'] and policy['manifest_sha256']==report['manifest_sha256']
    final=dict(passed=bool(task and policy['passed']),physical_passed=task,policy_passed=policy['passed'],
        integrity_passed=integrity,cycles=len(ends),completed_cycles=len(s['cycle_results']),case=m['case'],
        first_cycle_prefix_exact=s['first_cycle_prefix_exact'],trajectory_sha256=report['trajectory_sha256'],
        manifest_sha256=report['manifest_sha256'],summary_sha256=report['summary_sha256'],
        independent_audit_sha256=sha(out/'independent_audit.json'),policy_audit_sha256=sha(out/'policy_transfer_audit.json'),
        source_sha256=sha(__file__),final_rise_m=total_rise)
    write(out/'v2_multicycle_audit.json',final)
    print(json.dumps(final),flush=True)
    return final


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);audit(p.parse_args().output)
