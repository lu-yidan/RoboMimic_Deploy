"""Independent saved-state/action/reward checks for a geometric residual pilot."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.audit_multirung_cycle import audit_endpoints
from tools.continue_vector_full_cycle_training import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('experiment',type=Path);args=parser.parse_args();p=args.experiment
    m=json.loads((p/'manifest.json').read_text());s=json.loads((p/'summary.json').read_text())
    parent=Path(m['parent']);pm=json.loads((parent/'manifest.json').read_text())
    pa=json.loads((parent/'independent_training_audit.json').read_text())
    if not pa['passed'] or sha(parent/'independent_training_audit.json')!=m['parent_audit_sha256']:
        raise ValueError('Parent evidence changed')
    env=FullCycleResidualEnv(plan=pm['reference'],cases={m['case']:m['offsets']},
        normal_bias_y_deg=pm['normal_bias_y_deg'],memory_enabled=pm['force_history_memory'])
    env.reset(options={'case':m['case']});e=env.env
    checks=dict(manifest_hash=sha(p/'manifest.json')==s['manifest_sha256'],
        trajectory_hash=sha(p/'trajectory.npz')==s['trajectory_sha256'],
        runner_hash=sha(p/'runner_source.py')==m['source_sha256'],
        reference_hash=sha(pm['reference'])==pm['reference_sha256'],
        bundle_hash=sha(e.bundle/'manifest.json')==pm['bundle_manifest_sha256'],
        unchanged_training_sources=all(sha(source)==digest for source,digest in pm['sources'].items() if source.replace('\\','/').startswith('training/')))
    sid=e.model.site('rung_01_target').id;dz=float(e.model.site_pos[sid,2]-env.nominal_geometry[1][sid,2])
    ids=np.flatnonzero(env.plan['phase']=='rf_swing');start=float(env.plan['time'][ids[0]-1]);end=float(env.plan['time'][ids[-1]])
    checks['rule_metadata']=start==m['swing_start_s'] and end==m['swing_end_s'] and dz==m['measured_target_z_offset_m'] and m['right_leg_signs']==[1,-1,1,1,-1,1]
    with np.load(p/'trajectory.npz') as z:
        q,v,t,actions,obs,torques,rewards=(z[k] for k in ['qpos','qvel','time_s','actions','observations','torques','rewards'])
        n=len(actions);checks['dimensions']=q.shape==(n+1,e.model.nq) and v.shape==(n+1,e.model.nv) and obs.shape==(n,136) and torques.shape==(n,29) and actions.shape==(n,26) and rewards.shape==(n,) and t.shape==(n+1,)
        checks['finite']=all(bool(np.isfinite(x).all()) for x in (q,v,t,actions,obs,torques,rewards))
        checks['initial_state']=np.array_equal(q[0],e.data.qpos) and np.array_equal(v[0],e.data.qvel)
        margin=float(np.minimum(q[:,e.qadr]-e.model.jnt_range[e.joints,0],e.model.jnt_range[e.joints,1]-q[:,e.qadr]).min())
        checks['native_limits_or_declared_failure']=margin>=-1e-7 or s['reason']=='native_joint_position_violation'
        checks['torque_bounds']=bool(np.all(torques>=e.limits[:,0]-1e-8) and np.all(torques<=e.limits[:,1]+1e-8))
        checks['timing']=bool(t[0]==0. and np.all(abs(np.diff(t)[:-1]-.02)<1e-8) and 0.<t[-1]-t[-2]<=.02000001 and t[-1]==s['time_s'])
        expected=[];reward_error=0.;tracking_error=0.;previous=np.zeros(26)
        for i,time in enumerate(t[:-1]):
            reference,_,_,phase=env.reference(float(time));action=np.zeros(26)
            if phase=='rf_swing':
                u=float(np.clip(time-start,0.,1.));v_end=float(np.clip(end-time,0.,1.))
                gain=m.get('amplitude',1.)*min(u*u*(3-2*u),v_end*v_end*(3-2*v_end))*float(np.clip(dz/.01,0.,1.))
                action[6:12]=gain*np.array([1.,-1.,1.,1.,-1.,1.])
            expected.append(action)
            tracking_error=max(tracking_error,float(abs(obs[i,:29]-(q[i,e.qadr]-reference[e.qadr])).max()))
            ref_end=env.reference(float(t[i+1]))[0]
            reward=.02-2*np.mean((q[i+1,e.qadr]-ref_end[e.qadr])**2)-.001*np.mean(actions[i]**2)-.001*np.mean((actions[i]-previous)**2)
            if i==n-1:reward+=20. if s['success'] else -20.
            reward_error=max(reward_error,abs(float(rewards[i])-reward));previous=actions[i]
        checks['action_rule']=bool(np.array_equal(expected,actions));checks['action_bounds']=bool(abs(actions).max()<=1.)
        checks['tracking_observation']=tracking_error<1e-7;checks['velocity_observation']=bool(abs(obs[:,29:58]-v[:-1,e.vadr]).max()<1e-6)
        checks['reward_reconstruction']=reward_error<1e-7
        checks['success_horizon_and_forces']=not s['success'] or (abs(t[-1]-env.horizon)<1e-7 and max(s['peaks'][k] for k in ['other','self','flight'])<1. and margin>=-1e-7)
        endpoints=audit_endpoints(e,dict(qpos=q,qvel=v,time_s=t),env.plan)
    passed=all(checks.values());accepted=bool(passed and s['success'] and all(r['passed'] for r in endpoints))
    result=dict(integrity_passed=bool(passed),independently_accepted=accepted,checks={k:bool(v) for k,v in checks.items()},
        sampled_native_margin_rad=margin,reward_max_difference=reward_error,tracking_max_difference=tracking_error,endpoints=endpoints,
        summary_sha256=sha(p/'summary.json'),trajectory_sha256=sha(p/'trajectory.npz'),source_sha256=sha(__file__),
        scope='Independent recorded states/actions/rewards at50Hz; force peaks remain original2000Hz runtime evidence. Failed task may pass data integrity, never task acceptance.')
    (p/'independent_audit.json').write_text(json.dumps(result,indent=2));(p/'independent_audit_source.py').write_bytes(Path(__file__).read_bytes());env.close()
    print(json.dumps(result),flush=True)
    if not passed:raise AssertionError('Pilot data integrity failed')


if __name__=='__main__':main()
