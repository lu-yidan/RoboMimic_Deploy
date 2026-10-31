"""Fixed-distribution data for combined-control critic calibration, split by episode."""
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from stable_baselines3 import PPO
from training.closure_combined_curriculum_env import ClosureCombinedCurriculumEnv
from tools.continue_vector_full_cycle_training import sha

ENV=None
MODEL=None

def rollout(payload):
    global ENV,MODEL
    seed_dir,modifier,case,episode,index,output=payload;torch.set_num_threads(1)
    if ENV is None:
        ENV=ClosureCombinedCurriculumEnv('outputs/closure_prefix_cache_20261024_v4',controller_modifier=modifier)
        MODEL=PPO.load(Path(seed_dir)/'policy_initial.zip',device='cpu')
    seed=20291024+index;torch.manual_seed(seed);np.random.seed(seed)
    obs,_=ENV.reset(seed=seed,options={'case':case});base=ENV.env.unwrapped
    trace=dict(qpos=[base.env.data.qpos.copy()],qvel=[base.env.data.qvel.copy()],time_s=[float(base.env.data.time)],observations=[],actions=[],torques=[],rewards=[],base_rewards=[])
    while True:
        trace['observations'].append(obs.copy());action=MODEL.predict(obs,deterministic=False)[0]
        obs,reward,done,truncated,info=ENV.step(action)
        trace['actions'].append(action.copy());trace['qpos'].append(base.env.data.qpos.copy());trace['qvel'].append(base.env.data.qvel.copy());trace['time_s'].append(float(base.env.data.time))
        trace['torques'].append(base.last_torque.copy());trace['rewards'].append(reward);trace['base_rewards'].append(info['base_reward'])
        if done or truncated:break
    arrays={k:np.asarray(v) for k,v in trace.items()};assert all(np.isfinite(v).all() for v in arrays.values())
    file=f'episode_{index:03d}.npz';path=Path(output)/file;np.savez_compressed(path,**arrays)
    split='train' if episode<4 else 'validation' if episode<6 else 'test'
    row=dict(case=case,episode=episode,index=index,seed=seed,split=split,file=file,steps=len(arrays['actions']),success=info['success'],info=info,trajectory_sha256=sha(path))
    (path.with_suffix('.json')).write_text(json.dumps(row,indent=2));print(json.dumps({k:row[k] for k in ('case','episode','split','steps','success')}),flush=True);return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    seed=json.loads((a.seed/'summary.json').read_text());assert seed['passed'] and sha(a.seed/'policy_initial.zip')==seed['checkpoint_sha256']
    for name,value in seed['sources'].items():assert sha(name)==value
    parent=Path(seed['parent']);m=json.loads((parent/'manifest.json').read_text());cases=list(m['evaluation_cases'])
    a.output.mkdir(exist_ok=False)
    protocol=dict(seed_directory=str(a.seed.resolve()),seed_checkpoint_sha256=sha(a.seed/'policy_initial.zip'),seed_gate_sha256=sha(a.seed/'summary.json'),
        cases=cases,episodes_per_case=8,split=dict(train=4,validation=2,test=2),source_sha256=sha(__file__),controller_modifier=seed['controller_modifier'],
        sources={name:sha(name) for name in ('training/closure_combined_curriculum_env.py','training/closure_quality_reward.py','training/closure_waist_target.py','training/closure_body_target.py')},
        scope='Forty fresh fixed-policy stochastic suffix episodes, whole-episode split. No pilot episode reused; critic fitting only, not policy generalization evidence.')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    jobs=[(str(a.seed.resolve()),seed['controller_modifier'],case,episode,8*i+episode,str(a.output.resolve())) for i,case in enumerate(cases) for episode in range(8)]
    with ProcessPoolExecutor(max_workers=2) as pool:rows=[f.result() for f in as_completed([pool.submit(rollout,x) for x in jobs])]
    assert len(rows)==40 and len({r['index'] for r in rows})==40
    result=dict(passed=True,rows=sorted(rows,key=lambda r:r['index']),steps=sum(r['steps'] for r in rows),successes=sum(r['success'] for r in rows),
        protocol_sha256=sha(a.output/'protocol.json'),success_by_case={case:sum(r['success'] for r in rows if r['case']==case) for case in cases})
    (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()
