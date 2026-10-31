"""Fixed seeded-policy sampling, declared whole-episode splits, verified low-case reuse."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from stable_baselines3 import PPO
from training.closure_curriculum_env_v3 import ClosureCurriculumEnvV3
from training.closure_exploration_policy import ClosureExplorationPolicy
from tools.continue_vector_full_cycle_training import sha

SEED=Path('outputs/closure_seed_policy_20261024_v1')

def collect(payload):
    episodes,output=payload
    torch.set_num_threads(1)
    model=PPO.load(SEED/'policy_initial.zip',device='cpu')
    env=ClosureCurriculumEnvV3('outputs/closure_prefix_cache_20261024_v4')
    rows=[]
    for episode in episodes:
        seed=20271024+episode;torch.manual_seed(seed)
        obs,_=env.reset(seed=seed,options={'case':'foot_z_plus15'})
        base=env.env.unwrapped
        arrays=dict(observations=[],actions=[],rewards=[],base_rewards=[],qpos=[base.env.data.qpos.copy()],qvel=[base.env.data.qvel.copy()],time_s=[float(base.env.data.time)],torques=[])
        while True:
            arrays['observations'].append(obs.copy());action=model.predict(obs,deterministic=False)[0]
            obs,reward,done,truncated,info=env.step(action)
            arrays['actions'].append(action.copy());arrays['rewards'].append(reward);arrays['base_rewards'].append(info['base_reward'])
            arrays['qpos'].append(base.env.data.qpos.copy());arrays['qvel'].append(base.env.data.qvel.copy())
            arrays['time_s'].append(float(base.env.data.time));arrays['torques'].append(base.last_torque.copy())
            if done or truncated:break
        path=Path(output)/f'foot_z_plus15_{episode:02d}.npz';np.savez_compressed(path,**arrays)
        row=dict(case='foot_z_plus15',episode=episode,seed=seed,
          split='train' if episode<16 else 'validation' if episode<20 else 'test',
          steps=len(arrays['actions']),success=info['success'],reason=info['reason'],file=path.name,
          trajectory_sha256=sha(path),info=info,origin='fresh_seeded_policy')
        path.with_suffix('.json').write_text(json.dumps(row,indent=2));rows.append(row)
        print(json.dumps(dict(episode=episode,split=row['split'],success=info['success'],closure=info['current_closure_rad'],rise=info['current_rise_m'])),flush=True)
    env.close();return rows


def main():
    torch.set_num_threads(1)
    gate=json.loads((SEED/'summary.json').read_text());assert gate['passed'] and sha(SEED/'policy_initial.zip')==gate['checkpoint_sha256']
    for name,digest in gate['sources'].items():assert sha(name)==digest
    old=Path('outputs/closure_critic_data_20261024_v1')
    old_summary=json.loads((old/'summary.json').read_text());assert old_summary['passed']
    model=PPO.load(SEED/'policy_initial.zip',device='cpu')
    original=PPO.load('outputs/closure_margin_control98304_v1/policy.zip',device='cpu',custom_objects={'policy_class':ClosureExplorationPolicy})
    out=Path('outputs/seeded_closure_data_20261024_v1');out.mkdir(exist_ok=False)
    protocol=dict(initial_policy_sha256=sha(SEED/'policy_initial.zip'),seed_gate_sha256=sha(SEED/'summary.json'),
      fresh_highest_episodes=24,highest_split=dict(train=16,validation=4,test=4),
      low_case_reuse_summary_sha256=sha(old/'summary.json'),low_case_reused_episodes=24,
      sources={name:sha(name) for name in [str(Path(__file__).resolve()),'training/closure_seeded_policy.py','training/closure_curriculum_env_v3.py']},
      scope='Four low cases reuse existing fixed-policy data only after exact mean/std checks on every saved observation. Fresh highest episodes split before collection. Test episodes reserved for value calibration assessment, not policy/generalization claims.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
    for previous in old_summary['rows']:
        if previous['case']=='foot_z_plus15':continue
        path=old/previous['file'];assert sha(path)==previous['trajectory_sha256']
        with np.load(path) as z:obs=torch.as_tensor(z['observations'].copy())
        with torch.no_grad():
            a=original.policy.get_distribution(obs).distribution;b=model.policy.get_distribution(obs).distribution
            assert torch.equal(a.mean,b.mean) and torch.equal(a.stddev,b.stddev)
        shutil.copy2(path,out/path.name)
        row=dict(previous,origin='verified_distribution_reuse',source_file=str(path.resolve()),policy_mean_and_std_exact=True)
        (out/path.with_suffix('.json').name).write_text(json.dumps(row,indent=2));rows.append(row)
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(collect,(list(range(start,start+8)),str(out.resolve()))) for start in (0,8,16)]
        for future in as_completed(futures):rows.extend(future.result())
    rows.sort(key=lambda r:(r['case'],r['episode']))
    result=dict(passed=len(rows)==48,rows=rows,protocol_sha256=sha(out/'protocol.json'))
    (out/'summary.json').write_text(json.dumps(result,indent=2));(out/'collector_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(episodes=len(rows),highest_successes=sum(r['success'] for r in rows if r['case']=='foot_z_plus15'))),flush=True)


if __name__=='__main__':main()
