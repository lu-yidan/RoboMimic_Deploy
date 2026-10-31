"""Seven declared waist-reference probes around the closest verified 26D seed."""
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.closure_curriculum_env_v3 import ClosureCurriculumEnvV3
from training.closure_waist_target import install_waist_target
from tools.continue_vector_full_cycle_training import sha


def main():
    torch.set_num_threads(1)
    search=Path('outputs/closure_full_action_search_20261024_v1')
    result=json.loads((search/'summary.json').read_text())
    assert result['completed'] and not any(r['success'] for r in result['rows']), 'Finish ordinary 26D candidate validation first if any passed'
    seed=Path('outputs/closure_seed_policy_20261024_v1')
    gate=json.loads((seed/'summary.json').read_text());assert gate['passed']
    model=PPO.load(seed/'policy_initial.zip',device='cpu')
    env=ClosureCurriculumEnvV3('outputs/closure_prefix_cache_20261024_v4')
    out=Path('outputs/closure_waist_authority_20261024_v1');out.mkdir(exist_ok=False)
    protocol=dict(parent_seed_sha256=sha(seed/'policy_initial.zip'),trigger_search_sha256=sha(search/'summary.json'),
      waist_axes=['yaw','roll','pitch'],waist_probe_magnitude=.25,waist_residual_limit_rad=.015,
      waist_ramp_seconds=1.,sources={name:sha(name) for name in [str(Path(__file__).resolve()),'training/closure_waist_target.py','training/closure_curriculum_env_v3.py','training/closure_seeded_policy.py']},
      scope='Extra three waist command DOFs beyond the published 26D actor; explicit control-authority expansion, not PPO improvement. Original motor/native/task gates. Cached suffix screen only.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    probes=[np.zeros(3)]
    for j in range(3):
        for sign in (-1.,1.):
            p=np.zeros(3);p[j]=sign*.25;probes.append(p)
    rows=[]
    for trial,p in enumerate(probes):
        obs,_=env.reset(options={'case':'foot_z_plus15'});base=env.env.unwrapped
        state=install_waist_target(base,p,ramp_seconds=1.,record=True)
        trace=dict(qpos=[base.env.data.qpos.copy()],qvel=[base.env.data.qvel.copy()],time_s=[float(base.env.data.time)],observations=[],actions=[],torques=[],base_rewards=[])
        while True:
            trace['observations'].append(obs.copy());action=model.predict(obs,deterministic=True)[0]
            obs,reward,done,truncated,info=env.step(action)
            trace['actions'].append(action.copy());trace['qpos'].append(base.env.data.qpos.copy());trace['qvel'].append(base.env.data.qvel.copy())
            trace['time_s'].append(float(base.env.data.time));trace['torques'].append(base.last_torque.copy());trace['base_rewards'].append(info['base_reward'])
            if done or truncated:break
        arrays={k:np.asarray(v) for k,v in trace.items()};arrays['waist_reference_trace']=np.asarray(state['trace']).reshape(-1,10)
        path=out/f'trial_{trial:02d}.npz';np.savez_compressed(path,**arrays)
        tail=(arrays['time_s']>111.66)&(arrays['time_s']<=113.6600001);q=arrays['qpos'][tail]
        metrics=dict(closure_rad=float(abs(q[:,base.env.qadr]-base.env.initial[base.env.qadr]).max()),rise_max_m=float((q[:,2]-base.env.initial[2]).max())) if len(q) else {}
        row=dict(trial=trial,waist_bias=p.tolist(),success=info['success'],info=info,metrics=metrics,maximum_waist_offset_rad=state['maximum_offset_rad'],trajectory_sha256=sha(path))
        assert state['maximum_offset_rad']<=.015+1e-12
        if trial==0:
            baseline=Path('outputs/closure_action_timing_20261024_v1/ramp_1s.npz')
            with np.load(baseline) as z:
                zero_differences={key:float(abs(arrays[key]-z[key]).max()) for key in ('qpos','qvel','actions','torques')}
            assert max(zero_differences.values())<=1e-9
            row['zero_waist_differences']=zero_differences
        (out/f'trial_{trial:02d}.json').write_text(json.dumps(row,indent=2));rows.append(row)
        print(json.dumps(dict(trial=trial,waist_bias=p.tolist(),success=info['success'],**metrics)),flush=True)
    env.close();(out/'summary.json').write_text(json.dumps(dict(completed=True,rows=rows,protocol_sha256=sha(out/'protocol.json')),indent=2))


if __name__=='__main__':main()
