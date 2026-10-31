"""Predeclared exploration pilot for a passing combined-control seed."""
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
BASE_STD=None

def run(payload):
    global ENV,MODEL,BASE_STD
    parent,modifier,scale,episode,output,deterministic=payload
    torch.set_num_threads(1)
    if ENV is None:
        ENV=ClosureCombinedCurriculumEnv('outputs/closure_prefix_cache_20261024_v4',controller_modifier=modifier)
        MODEL=PPO.load(Path(parent)/'policy.zip',device='cpu');BASE_STD=MODEL.policy.log_std.detach().clone()
    with torch.no_grad():MODEL.policy.log_std.copy_(BASE_STD+np.log(scale))
    seed=20281024+episode;torch.manual_seed(seed);np.random.seed(seed)
    obs,_=ENV.reset(seed=seed,options={'case':'foot_z_plus15'});base=ENV.env.unwrapped
    trace=dict(qpos=[base.env.data.qpos.copy()],qvel=[base.env.data.qvel.copy()],time_s=[float(base.env.data.time)],observations=[],actions=[],torques=[],rewards=[],base_rewards=[])
    while True:
        trace['observations'].append(obs.copy());action=MODEL.predict(obs,deterministic=deterministic)[0]
        obs,reward,done,truncated,info=ENV.step(action)
        trace['actions'].append(action.copy());trace['qpos'].append(base.env.data.qpos.copy());trace['qvel'].append(base.env.data.qvel.copy());trace['time_s'].append(float(base.env.data.time))
        trace['torques'].append(base.last_torque.copy());trace['rewards'].append(reward);trace['base_rewards'].append(info['base_reward'])
        if done or truncated:break
    path=Path(output)/f'episode_{episode:02d}.npz';arrays={k:np.asarray(v) for k,v in trace.items()};np.savez_compressed(path,**arrays)
    if deterministic:
        with np.load(Path(parent)/'after_foot_z_plus15.npz') as original:
            assert np.max(abs(original['qpos'][4983:]-arrays['qpos']))<1e-10
            assert np.max(abs(original['qvel'][4983:]-arrays['qvel']))<1e-8
            assert np.max(abs(original['actions'][4983:]-arrays['actions']))<1e-6
        assert info['success']
    tail=arrays['time_s']>111.66;q=arrays['qpos'][tail]
    row=dict(episode=episode,scale=scale,seed=seed,deterministic=deterministic,success=info['success'],info=info,
        closure_rad=float(abs(q[:,base.env.qadr]-base.env.initial[base.env.qadr]).max()) if len(q) else None,
        rise_max_m=float((q[:,2]-base.env.initial[2]).max()) if len(q) else None,trajectory_sha256=sha(path))
    (Path(output)/f'episode_{episode:02d}.json').write_text(json.dumps(row,indent=2));print(json.dumps({k:row[k] for k in ('episode','scale','deterministic','success','closure_rad','rise_max_m')}),flush=True)
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--gate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=json.loads((a.parent/'manifest.json').read_text());audit=json.loads((a.parent/'independent_training_audit.json').read_text());gate=json.loads((a.gate/'summary.json').read_text())
    assert audit['passed'] and audit['experiment_manifest_sha256']==sha(a.parent/'manifest.json') and audit['experiment_summary_sha256']==sha(a.parent/'summary.json')
    assert sum(r['independently_accepted'] for r in audit['trajectories'] if r['label']=='after')==5
    assert gate['passed'] and gate['environment_source_sha256']==sha('training/closure_combined_curriculum_env.py')
    assert json.loads((a.parent/'composite_prefix_audit.json').read_text())['passed']
    a.output.mkdir(exist_ok=False)
    protocol=dict(parent=str(a.parent.resolve()),parent_audit_sha256=sha(a.parent/'independent_training_audit.json'),parent_checkpoint_sha256=sha(a.parent/'policy.zip'),
        scales=[1.,.25,.1],episodes_per_scale=4,selection='Most successes, largest exploration scale for ties; require at least one success. Pilot only, not statistical robustness.',
        environment_gate_sha256=sha(a.gate/'summary.json'),source_sha256=sha(__file__),controller_modifier=m['controller_modifier'],
        environment_source_sha256=sha('training/closure_combined_curriculum_env.py'),scope='Known highest-case suffix exploration selection; changes log standard deviation only, deterministic mean unchanged. No PPO training.')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with ProcessPoolExecutor(max_workers=2) as pool:
        first=pool.submit(run,(str(a.parent.resolve()),m['controller_modifier'],1.,0,str(a.output.resolve()),True)).result()
        jobs=[(str(a.parent.resolve()),m['controller_modifier'],scale,1+4*i+j,str(a.output.resolve()),False) for i,scale in enumerate(protocol['scales']) for j in range(4)]
        rows=[first]+[f.result() for f in as_completed([pool.submit(run,x) for x in jobs])]
    counts={str(scale):sum(r['success'] for r in rows if not r['deterministic'] and r['scale']==scale) for scale in protocol['scales']}
    scale=max(protocol['scales'],key=lambda s:(counts[str(s)],s));eligible=counts[str(scale)]>0
    result=dict(completed=True,rows=sorted(rows,key=lambda r:r['episode']),counts=counts,selected_scale=scale if eligible else None,eligible_for_training_preparation=eligible,protocol_sha256=sha(a.output/'protocol.json'))
    (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(counts=counts,selected_scale=result['selected_scale'])),flush=True)

if __name__=='__main__':main()
