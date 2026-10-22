"""Independent source, state, reward and endpoint audit of zero-action transfer."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.audit_multirung_cycle import audit_endpoints
from tools.continue_vector_full_cycle_training import sha


def audit(p):
    m=json.loads((p/'manifest.json').read_text());s=json.loads((p/'summary.json').read_text())
    case=m.get('case','foot_z_plus10')
    modifier=m.get('reference_modifier')
    if modifier and modifier.get('type')=='smooth-unload-reference-v1':
        from training.smooth_unload_reference_env import SmoothUnloadReferenceEnv
        env=SmoothUnloadReferenceEnv(plan=m['reference'],cases={case:m['offsets']},normal_bias_y_deg=m['normal_bias_y_deg'],memory_enabled=m['memory_enabled'])
        if sha(modifier['gate'])!=modifier['gate_sha256']:raise ValueError('Smooth reference gate changed')
    elif modifier:
        from training.temporal_base_reference_env import TemporalBaseReferenceEnv
        if modifier['type']!='bounded-base-reference-57to65s-cubic2s-v1' or (modifier['start_s'],modifier['end_s'],modifier['ramp_s'])!=(57.,65.,2.):raise ValueError('Unsupported reference modifier')
        env=TemporalBaseReferenceEnv(plan=m['reference'],cases={case:m['offsets']},normal_bias_y_deg=m['normal_bias_y_deg'],memory_enabled=m['memory_enabled'],base_bias=modifier['base_bias'])
    elif m.get('controller_modifier',{}).get('type')=='reference-clock-unload-v1':
        from training.retimed_unload_env import RetimedUnloadEnv
        env=RetimedUnloadEnv(plan=m['reference'],cases={case:m['offsets']},normal_bias_y_deg=m['normal_bias_y_deg'],memory_enabled=m['memory_enabled'])
        if env.unload_factor!=m['controller_modifier']['factor'] or not np.allclose(m['bias_window_s'],[49.,52.2+env.unload_factor*(61.56-52.2)],rtol=0,atol=1e-12):raise ValueError('Invalid retimed disturbance declaration')
    elif m.get('controller_modifier',{}).get('type')=='oracle-acceleration-qp-window-roundoff-v2':
        modifier_record=m['controller_modifier']
        assert modifier_record==dict(type='oracle-acceleration-qp-window-roundoff-v2',start_s=57.,end_s=65.,ramp_s=2.,contact_weight=1000.,normal_damping=20.)
        # Reference/observation/reward contract is unchanged; no controller replay here.
        env=FullCycleResidualEnv(plan=m['reference'],cases={case:m['offsets']},normal_bias_y_deg=m['normal_bias_y_deg'],memory_enabled=m['memory_enabled'])
    elif m.get('controller_modifier'):
        from training.temporal_leg_gain_env import TemporalLegGainEnv
        c=m['controller_modifier']
        if c['type']!='leg-kp-57to65s-cubic2s-v1' or (c['start_s'],c['end_s'],c['ramp_s'])!=(57.,65.,2.):raise ValueError('Unsupported controller modifier')
        env=TemporalLegGainEnv(plan=m['reference'],cases={case:m['offsets']},normal_bias_y_deg=m['normal_bias_y_deg'],memory_enabled=m['memory_enabled'],peak_leg_kp=c['peak_leg_kp'])
    else:env=FullCycleResidualEnv(plan=m['reference'],cases={case:m['offsets']},normal_bias_y_deg=m['normal_bias_y_deg'],memory_enabled=m['memory_enabled'])
    env.reset(options={'case':case});e=env.env
    checks=dict(manifest_hash=sha(p/'manifest.json')==s['manifest_sha256'],trajectory_hash=sha(p/'trajectory.npz')==s['trajectory_sha256'],
        reference_hash=sha(m['reference'])==m['reference_sha256'],bundle_hash=sha(e.bundle/'manifest.json')==m['bundle_manifest_sha256'],
        protocol_hash=sha(p.parent/'protocol.json')==m['protocol_sha256'],
        frozen_sources=all(sha(p/'source_snapshot'/path)==digest for path,digest in m['sources'].items()),
        current_environment_sources=all(sha(path)==digest for path,digest in m['sources'].items() if Path(path).parts[0]=='training'))
    with np.load(p/'trajectory.npz') as z:
        q,v,t,obs,actions,torques,rewards=(z[k] for k in ('qpos','qvel','time_s','observations','actions','torques','rewards'))
        n=len(actions)
        checks['dimensions']=(q.shape==(n+1,e.model.nq) and v.shape==(n+1,e.model.nv) and t.shape==(n+1,) and obs.shape==(n,136) and actions.shape==(n,26) and torques.shape==(n,29) and rewards.shape==(n,))
        checks['finite']=all(bool(np.isfinite(x).all()) for x in (q,v,t,obs,actions,torques,rewards))
        checks['initial_state']=np.array_equal(q[0],e.data.qpos) and np.array_equal(v[0],e.data.qvel)
        rule=m.get('action_rule')
        if rule is None:
            checks['zero_actions_and_filtered_residual']=bool(np.all(actions==0.) and np.all(obs[:,110:136]==0.))
        else:
            if rule['type']!='fixed_vector_cubic_envelope':raise ValueError('Unknown fixed action rule')
            vector=np.asarray(rule['vector']);ramp=rule['ramp_s']
            if vector.shape!=(26,) or not np.isfinite(vector).all() or abs(vector).max()>1. or ramp<=0:raise ValueError('Invalid bounded action rule')
            entering=np.clip((t[:-1]-rule['start_s'])/ramp,0,1);leaving=np.clip((rule['end_s']-t[:-1])/ramp,0,1)
            gains=np.minimum(entering**2*(3-2*entering),leaving**2*(3-2*leaving))
            expected=gains[:,None]*vector
            checks['fixed_action_rule']=bool(np.max(abs(actions-expected))<1e-14)
            checks['action_bounds']=bool(abs(actions).max()<=1.)
            residual=np.zeros(26);difference=0.
            for i,action in enumerate(actions):
                difference=max(difference,float(abs(obs[i,110:136]-residual/.015).max()))
                for _ in range(4):residual+=.05*(.015*action-residual)
            checks['filtered_residual_observation']=difference<1e-7
        checks['timing']=bool(t[0]==0 and np.all(abs(np.diff(t)[:-1]-.02)<1e-8) and 0<t[-1]-t[-2]<=.02000001 and t[-1]==s['time_s'])
        checks['torque_bounds']=bool(np.all(torques>=e.limits[:,0]-1e-8) and np.all(torques<=e.limits[:,1]+1e-8))
        margin=float(np.minimum(q[:,e.qadr]-e.model.jnt_range[e.joints,0],e.model.jnt_range[e.joints,1]-q[:,e.qadr]).min())
        checks['native_or_declared_failure']=margin>=-1e-7 or s['reason']=='native_joint_position_violation'
        reward_error=tracking_error=penetration=0.;data=mujoco.MjData(e.model);previous_action=np.zeros(26)
        for i,time in enumerate(t):
            data.qpos[:]=q[i];mujoco.mj_kinematics(e.model,data);mujoco.mj_collision(e.model,data)
            for c in data.contact:
                a,b=int(e.model.geom_bodyid[c.geom1]),int(e.model.geom_bodyid[c.geom2]);ina,inb=a in e.robot_bodies,b in e.robot_bodies
                if not (ina or inb):continue
                expected=ina!=inb and (a if ina else b) in e.contact_groups
                if not expected:penetration=max(penetration,float(-c.dist))
            if i==n:continue
            ref=env.reference(float(time))[0]
            tracking_error=max(tracking_error,float(abs(obs[i,:29]-(q[i,e.qadr]-ref[e.qadr])).max()))
            ref_end=env.reference(float(t[i+1]))[0]
            reward=.02-2*np.mean((q[i+1,e.qadr]-ref_end[e.qadr])**2)-.001*np.mean(actions[i]**2)-.001*np.mean((actions[i]-previous_action)**2)
            if i==n-1:reward+=20 if s['success'] else -20
            reward_error=max(reward_error,abs(float(rewards[i])-reward))
            previous_action=actions[i]
        checks['tracking_observation']=tracking_error<1e-7
        checks['velocity_observation']=bool(abs(obs[:,29:58]-v[:-1,e.vadr]).max()<1e-6)
        root_reference=np.column_stack([np.interp(t[:-1],env.plan['time'],env.plan['qpos'][:,j]) for j in range(3)])
        if modifier and modifier.get('type')=='smooth-unload-reference-v1':
            root_reference=np.array([env.reference(float(time))[0][:3] for time in t[:-1]])
        elif modifier:
            u=np.clip((t[:-1]-57.)/2.,0,1);w=np.clip((65.-t[:-1])/2.,0,1)
            root_reference+=np.minimum(u*u*(3-2*u),w*w*(3-2*w))[:,None]*np.asarray(modifier['base_bias'])
        checks['base_reference_observation']=bool(abs(obs[:,61:64]-(q[:-1,:3]-root_reference)).max()<1e-7)
        checks['base_velocity_observation']=bool(abs(obs[:,64:70]-v[:-1,:6]).max()<1e-6)
        checks['reward_reconstruction']=reward_error<1e-7
        endpoints=audit_endpoints(e,dict(qpos=q,qvel=v,time_s=t),env.plan)
        runtime_acceptance=bool(s['success'] and abs(t[-1]-env.horizon)<1e-7 and s['minimum_native_joint_margin_rad']>=-1e-7 and max(s['peaks'][k] for k in ('flight','self','other'))<1. and s['peaks']['base']<.03 and s['peaks']['torque']<=1.+1e-8 and len(s['cycle_results'])==len(env.ends) and all(x['passed'] for x in s['cycle_results']))
    if m.get('controller_modifier',{}).get('type')=='oracle-acceleration-qp-window-roundoff-v2':
        records=json.loads((p/'qp_records.json').read_text())
        checks['qp_record_hash']=sha(p/'qp_records.json')==s['qp_records_sha256']
        checks['qp_window_and_constraints']=all(57.<r['time_s']<65. and 0.<r['gain']<=1. and r['constraint_residual']<1e-3 and r['solver_roundoff_clip_nm']<=1e-3 for r in records)
        checks['qp_intervention_recorded']=bool(records) if s['time_s']>57.005 else True
    passed=all(checks.values())
    accepted=bool(passed and runtime_acceptance and margin>=-1e-7 and penetration<1e-5 and all(r['passed'] for r in endpoints))
    result=dict(integrity_passed=bool(passed),independently_accepted=accepted,checks={k:bool(v) for k,v in checks.items()},
        sampled_native_margin_rad=margin,sampled_unexpected_penetration_m=penetration,reward_max_difference=reward_error,
        endpoints=endpoints,summary_sha256=sha(p/'summary.json'),trajectory_sha256=sha(p/'trajectory.npz'),audit_source_sha256=sha(__file__),
        scope='Saved-state audit at50Hz; 2000Hz force/native-limit extrema remain runtime evidence. Each reference defines its own endpoint target; cannot substitute this result into the original policy benchmark.')
    (p/'independent_audit.json').write_text(json.dumps(result,indent=2));(p/'independent_audit_source.py').write_bytes(Path(__file__).read_bytes());env.close()
    if not passed:raise AssertionError(result)
    return dict(label=m['label'],accepted=accepted,integrity_passed=bool(passed),reason=s['reason'],time_s=s['time_s'],endpoints=endpoints,audit_sha256=sha(p/'independent_audit.json'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('experiment',type=Path);args=parser.parse_args()
    results=json.loads((args.experiment/'results.json').read_text());protocol=json.loads((args.experiment/'protocol.json').read_text())
    assert {r['label'] for r in results}=={j[1] for j in protocol['jobs']}
    rows=[audit(args.experiment/r['label']) for r in results]
    (args.experiment/'independent_comparison.json').write_text(json.dumps(dict(results=rows,protocol_sha256=sha(args.experiment/'protocol.json'),source_sha256=sha(__file__)),indent=2))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
