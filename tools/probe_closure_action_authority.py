"""Finite-difference response within the existing bounded residual action space.

Training-only cached suffixes, never counted as full-cycle task successes.
No change to controller, geometry, actuator limits or endpoint thresholds.
"""
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.closure_curriculum_env_v3 import ClosureCurriculumEnvV3
from tools.continue_vector_full_cycle_training import sha


def main():
    torch.set_num_threads(1)
    out=Path('outputs/closure_action_authority_20261024_v1')
    out.mkdir(exist_ok=False)
    parent=Path('outputs/closure_margin_control98304_v1')
    cache=Path('outputs/closure_prefix_cache_20261024_v4')
    model=PPO.load(parent/'policy.zip',device='cpu')
    env=ClosureCurriculumEnvV3(cache)
    env.reset(options={'case':'foot_z_plus15'})
    base=env.env.unwrapped
    names=[base.env.model.joint(int(base.env.joints[i])).name for i in base.indices]
    groups=[('left_ankle_pitch',{'left_ankle_pitch_joint':1.}),
      ('right_ankle_pitch',{'right_ankle_pitch_joint':1.}),
      ('paired_hip_pitch',{'left_hip_pitch_joint':1.,'right_hip_pitch_joint':1.}),
      ('paired_knee',{'left_knee_joint':1.,'right_knee_joint':1.}),
      ('opposed_hip_roll',{'left_hip_roll_joint':1.,'right_hip_roll_joint':-1.}),
      ('paired_shoulder_pitch',{'left_shoulder_pitch_joint':1.,'right_shoulder_pitch_joint':1.})]
    basis=np.zeros((26,len(groups)))
    for j,(_,joints) in enumerate(groups):
        for name,value in joints.items():basis[names.index(name),j]=value
    protocol=dict(parent=str(parent.resolve()),parent_checkpoint_sha256=sha(parent/'policy.zip'),
      cache_summary_sha256=sha(cache/'summary.json'),case='foot_z_plus15',
      groups=groups,basis=basis.tolist(),step=.5,ramp_seconds=3.,
      sources={name:sha(name) for name in (str(Path(__file__).resolve()),'training/closure_curriculum_env_v3.py')},
      scope='13 finite-difference cached suffix probes; unchanged action bounds. Search/diagnostic data on known failure case, not policy training or from-zero task evidence.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    parameters=[np.zeros(len(groups))]
    for j in range(len(groups)):
        for sign in (-1,1):
            p=np.zeros(len(groups));p[j]=sign*.5;parameters.append(p)
    rows=[]
    for trial,p in enumerate(parameters):
        obs,_=env.reset(options={'case':'foot_z_plus15'})
        base=env.env.unwrapped
        trace=dict(qpos=[base.env.data.qpos.copy()],qvel=[base.env.data.qvel.copy()],time_s=[float(base.env.data.time)],observations=[],actions=[],torques=[],base_rewards=[])
        while True:
            t=float(base.env.data.time)
            u=float(np.clip((t-99.66)/3.,0.,1.));ramp=u*u*(3-2*u)
            action=np.clip(model.predict(obs,deterministic=True)[0]+ramp*(basis@p),-1.,1.).astype(np.float32)
            trace['observations'].append(obs.copy());trace['actions'].append(action.copy())
            obs,reward,done,truncated,info=env.step(action)
            trace['qpos'].append(base.env.data.qpos.copy());trace['qvel'].append(base.env.data.qvel.copy())
            trace['time_s'].append(float(base.env.data.time));trace['torques'].append(base.last_torque.copy());trace['base_rewards'].append(info['base_reward'])
            if done or truncated:break
        arrays={k:np.asarray(v) for k,v in trace.items()}
        tail=(arrays['time_s']>111.66)&(arrays['time_s']<=113.6600001)
        q=arrays['qpos'][tail]
        joint=q[:,base.env.qadr]-base.env.initial[base.env.qadr] if len(q) else np.full((1,29),np.nan)
        rise=q[:,2]-base.env.initial[2] if len(q) else np.array([np.nan])
        path=out/f'trial_{trial:02d}.npz';np.savez_compressed(path,**arrays)
        row=dict(trial=trial,parameters=p.tolist(),success=info['success'],reason=info['reason'],
          steps=len(arrays['actions']),closure_rad=float(abs(joint).max()),rise_min_m=float(rise.min()),rise_max_m=float(rise.max()),
          mean_joint_errors=joint.mean(axis=0).tolist(),mean_rise_m=float(rise.mean()),
          info=info,trajectory_sha256=sha(path))
        rows.append(row);(out/f'trial_{trial:02d}.json').write_text(json.dumps(row,indent=2))
        print(json.dumps({k:row[k] for k in ('trial','parameters','success','closure_rad','rise_max_m')}),flush=True)
    env.close()
    (out/'summary.json').write_text(json.dumps(dict(completed=True,rows=rows,protocol_sha256=sha(out/'protocol.json')),indent=2))


if __name__=='__main__':main()
