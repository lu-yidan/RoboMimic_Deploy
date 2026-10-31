"""Seven declared waist-reference probes around the closest verified 26D seed."""
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.closure_curriculum_env_v3 import ClosureCurriculumEnvV3
from training.closure_waist_target import install_waist_target
from training.closure_body_target import install_body_target
from tools.continue_vector_full_cycle_training import sha


def main():
    torch.set_num_threads(1)
    search=Path('outputs/closure_waist_authority_20261024_v1')
    result=json.loads((search/'summary.json').read_text())
    assert result['completed'] and not any(r['success'] for r in result['rows']), 'Validate a passing waist-only candidate first'
    seed=Path('outputs/closure_seed_policy_20261024_v1')
    gate=json.loads((seed/'summary.json').read_text());assert gate['passed']
    model=PPO.load(seed/'policy_initial.zip',device='cpu')
    env=ClosureCurriculumEnvV3('outputs/closure_prefix_cache_20261024_v4')
    out=Path('outputs/closure_waist_body_probe_20261024_v1');out.mkdir(exist_ok=False)
    protocol=dict(parent_seed_sha256=sha(seed/'policy_initial.zip'),trigger_search_sha256=sha(search/'summary.json'),
      waist_bias=[0.,-.5,0.],body_drops_m=[0.,.0001,.0002,.0003,.0005],waist_residual_limit_rad=.015,
      waist_ramp_seconds=1.,sources={name:sha(name) for name in [str(Path(__file__).resolve()),'training/closure_waist_target.py','training/closure_body_target.py','training/closure_curriculum_env_v3.py','training/closure_seeded_policy.py']},
      scope='Fixed waist-roll residual plus small root-height reference drop, beyond the published 26D actor; explicit control-authority expansion, not PPO improvement. Original motor/native/task gates. Cached suffix screen only.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    for trial,drop in enumerate([0.,.0001,.0002,.0003,.0005]):
        p=np.asarray([0.,-.5,0.])
        obs,_=env.reset(options={'case':'foot_z_plus15'});base=env.env.unwrapped
        state=install_waist_target(base,p,ramp_seconds=1.,record=True)
        install_body_target(base,drop)
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
        row=dict(trial=trial,waist_bias=p.tolist(),body_drop_m=drop,success=info['success'],info=info,metrics=metrics,maximum_waist_offset_rad=state['maximum_offset_rad'],trajectory_sha256=sha(path))
        assert state['maximum_offset_rad']<=.015+1e-12
        (out/f'trial_{trial:02d}.json').write_text(json.dumps(row,indent=2));rows.append(row)
        print(json.dumps(dict(trial=trial,waist_bias=p.tolist(),body_drop_m=drop,success=info['success'],**metrics)),flush=True)
    env.close();(out/'summary.json').write_text(json.dumps(dict(completed=True,rows=rows,protocol_sha256=sha(out/'protocol.json')),indent=2))


if __name__=='__main__':main()
