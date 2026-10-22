"""Independent frozen policy residual reconstruction and matched continuous prefix audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from stable_baselines3 import PPO
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.wall_support_env import WallSupportEnv


def read(path):return json.loads(path.read_text())


def audit(run,raw,e):
    manifest=read(run/'manifest.json');baseline=read(raw/'manifest.json')
    records=read(run/'policy_control_records.json')
    ref=dict(np.load(run/'policy_online_reference.npz'));plan=dict(np.load(run/'reference.npz'))
    errors={'action':0.,'offset':0.,'qref':0.,'vref':0.};residual=np.zeros(6)
    policy_records=read(run/'policy_actions.json');torch.set_num_threads(1)
    model=PPO.load(run/'policy.zip',device='cpu')
    predicted=model.predict(np.asarray([r['observation'] for r in policy_records],dtype=np.float32),deterministic=True)[0]
    prediction_error=float(np.max(abs(predicted-np.asarray([r['action'] for r in policy_records]))))
    provenance=read(run/'policy_provenance.json')
    checkpoint_ok=hashlib.sha256((run/'policy.zip').read_bytes()).hexdigest()==provenance['sha256']==hashlib.sha256(Path(provenance['source']).read_bytes()).hexdigest()
    policy_i=0;window_steps=0;held=np.zeros(6);cadence_ok=True
    previous=np.zeros(6);history_ok=True

    margin=float('inf');active_ok=True;time_ok=True
    for i,row in enumerate(records):
        t=row['time_s']
        if 49.<=t<61.56:
            if window_steps%4==0:
                recorded=policy_records[policy_i];policy_i+=1
                cadence_ok &= abs(recorded['time_s']-t)<1e-9
                history_ok &= np.allclose(recorded['observation'][-6:],previous,atol=1e-7,rtol=0.)
                held=np.asarray(recorded['action'])
            window_steps+=1
            if window_steps%4==0:previous=held.copy()
            action=held
        else:action=np.zeros(6)
        old=residual.copy();residual+=.05*(.015*action-residual)
        errors['action']=max(errors['action'],float(np.max(abs(np.asarray(row['action'])-action))))
        errors['offset']=max(errors['offset'],float(np.max(abs(np.asarray(row['offset_rad'])-residual))))
        if i>=len(ref['time_s']):continue
        time_ok &= abs(t-ref['time_s'][i])<1e-10
        j=int(np.clip(np.searchsorted(plan['time'],t,side='right')-1,0,len(plan['time'])-2))
        dt=plan['time'][j+1]-plan['time'][j];u=np.clip((t-plan['time'][j])/dt,0.,1.)
        q=(1-u)*plan['qpos'][j]+u*plan['qpos'][j+1];v=np.zeros(e.model.nv)
        v[:3]=(plan['qpos'][j+1,:3]-plan['qpos'][j,:3])/dt
        v[e.vadr]=(plan['qpos'][j+1,e.qadr]-plan['qpos'][j,e.qadr])/dt
        q[e.qadr[6:12]]+=residual;v[e.vadr[6:12]]+=(residual-old)/.005
        errors['qref']=max(errors['qref'],float(np.max(abs(q-ref['qpos'][i]))))
        errors['vref']=max(errors['vref'],float(np.max(abs(v-ref['qvel'][i]))))
        active_ok &= np.array_equal(ref['active'][i],plan['active'][j+1])
        margin=min(margin,float(np.min(np.minimum(q[e.qadr]-e.model.jnt_range[e.joints,0],e.model.jnt_range[e.joints,1]-q[e.qadr]))))
    z=dict(np.load(run/'trajectory.npz'));z0=dict(np.load(raw/'trajectory.npz'))
    n=int(np.sum(z0['time_s']<=49.+1e-8))
    prefix=float(np.max(abs(z['qpos'][:n]-z0['qpos'][:n])))
    core=set(manifest['sources'])&set(baseline['sources'])
    configs=[dict(m['config']) for m in [manifest,baseline]]
    for cfg in configs:cfg.pop('output_dir')
    summary=read(run/'summary.json')
    checks=dict(source_snapshot=all(hashlib.sha256((run/'source_snapshot'/p).read_bytes()).hexdigest()==h for p,h in manifest['sources'].items()),same_shared_source=all(manifest['sources'][p]==baseline['sources'][p] for p in core),same_configuration=configs[0]==configs[1],same_plan=manifest['plan_sha256']==baseline['plan_sha256'],same_geometry=read(run/'scene_variant.json')['geometry_sha256']==read(raw/'scene_variant.json')['geometry_sha256'],identical_prefix=prefix<1e-10,reference_reconstruction=max(errors.values())<1e-10,reference_native_positions=margin>=-1e-7,active_schedule=bool(active_ok),reference_times=bool(time_ok),complete_reference=abs(ref['time_s'][-1]+.005-summary['simulation_time_s'])<1e-7)
    checks.update(checkpoint_unchanged=checkpoint_ok,policy_actions=prediction_error<2e-6,policy_cadence=bool(cadence_ok and policy_i==len(policy_records)),previous_action_observation=bool(history_ok))
    errors['batch_policy_prediction']=prediction_error
    checks={key:bool(value) for key,value in checks.items()}
    result=dict(passed=all(checks.values()),checks=checks,reconstruction_errors=errors,reference_joint_margin_rad=margin,prefix_qpos_difference=prefix,scope='Action only49..61.56s; filtered residual decays afterwards. Same shared controller code and configuration, different explicitly frozen runner. No artificial forces or state replacement. Reference/action audit separate from task success. Ideal force observation channels are runtime logs, not independent force reconstruction.')
    (run/'policy_reference_audit.json').write_text(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);p.add_argument('--baseline',type=Path,default=ROOT/'outputs/window_memory_fullcycles_v1');a=p.parse_args()
    e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');results=[]
    for case,_ in read(a.suite/'protocol.json')['cases']:
        run=a.suite/('policy__'+case);audit_result=audit(run,a.baseline/('raw__'+case),e)
        passed=all(read(run/name)['passed'] for name in ['summary.json','independent_audit.json','joint_limit_audit.json','contact_identity_audit.json'])
        results.append(dict(case=case,task_passed=passed,reference_audit=audit_result))
    e.close();result=dict(integrity_passed=all(r['reference_audit']['passed'] for r in results),successes=sum(r['task_passed'] for r in results),results=results)
    (a.suite/'paired_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if not result['integrity_passed']:raise SystemExit(1)


if __name__=='__main__':main()
