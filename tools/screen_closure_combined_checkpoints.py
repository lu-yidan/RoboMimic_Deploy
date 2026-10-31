"""Development screening of every saved PPO checkpoint; no task-gate changes."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.closure_combined_curriculum_env import ClosureCombinedCurriculumEnv
from tools.continue_vector_full_cycle_training import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    declaration=json.loads(a.protocol.read_text());assert declaration['source_sha256']==sha(__file__) and Path(declaration['experiment']).resolve()==a.experiment.resolve()
    m=json.loads((a.experiment/'manifest.json').read_text());s=json.loads((a.experiment/'summary.json').read_text());audit=json.loads((a.experiment/'independent_training_audit.json').read_text())
    assert audit['passed'] and not s['evaluation_pending'] and audit['experiment_manifest_sha256']==sha(a.experiment/'manifest.json') and audit['experiment_summary_sha256']==sha(a.experiment/'summary.json')
    assert s['additional_steps']==98304
    files=[('initial',a.experiment/'policy_before.zip')]+[(f'collected_{n}_before_update',a.experiment/f'checkpoint_additional_{n}.zip') for n in range(8192,98305,8192)]+[('final',a.experiment/'policy.zip')]
    assert all(path.exists() for _,path in files)
    torch.set_num_threads(1);a.output.mkdir(exist_ok=False)
    env=ClosureCombinedCurriculumEnv(m['prefix_cache'],controller_modifier=m['controller_modifier']);rows=[]
    try:
        for label,path in files:
            model=PPO.load(path,device='cpu');obs,_=env.reset(options={'case':'foot_z_plus15'});base=env.env.unwrapped
            trace=dict(qpos=[base.env.data.qpos.copy()],qvel=[base.env.data.qvel.copy()],time_s=[float(base.env.data.time)],observations=[],actions=[],torques=[],base_rewards=[])
            while True:
                trace['observations'].append(obs.copy());action=model.predict(obs,deterministic=True)[0]
                obs,reward,done,truncated,info=env.step(action)
                trace['actions'].append(action.copy());trace['qpos'].append(base.env.data.qpos.copy());trace['qvel'].append(base.env.data.qvel.copy());trace['time_s'].append(float(base.env.data.time))
                trace['torques'].append(base.last_torque.copy());trace['base_rewards'].append(info['base_reward'])
                if done or truncated:break
            arrays={k:np.asarray(v) for k,v in trace.items()};tail=arrays['time_s']>111.66;q=arrays['qpos'][tail]
            complete=len(arrays['actions'])==700 and len(q)>=99
            closure=float(abs(q[:,base.env.qadr]-base.env.initial[base.env.qadr]).max()) if complete else None
            rise=float((q[:,2]-base.env.initial[2]).max()) if complete else None
            height_error=float(abs(q[:,2]-base.env.initial[2]-.25).max()) if complete else None
            score=min((.04-closure)/.001,(.01-height_error)/.0001) if complete else -1e9
            if label=='initial':
                with np.load(Path(m['parent_experiment'])/'after_foot_z_plus15.npz') as reference:
                    assert np.max(abs(reference['qpos'][4983:]-arrays['qpos']))<1e-10
                    assert np.max(abs(reference['qvel'][4983:]-arrays['qvel']))<1e-8
                assert info['success']
            target=a.output/f'{label}.npz';np.savez_compressed(target,**arrays)
            row=dict(label=label,checkpoint=str(path.resolve()),checkpoint_sha256=sha(path),success=info['success'],complete=complete,margin_score=score,
                closure_rad=closure,rise_max_m=rise,info=info,trajectory_sha256=sha(target))
            rows.append(row);(a.output/f'{label}.json').write_text(json.dumps(row,indent=2));print(json.dumps({k:row[k] for k in ('label','success','margin_score','closure_rad','rise_max_m')}),flush=True)
        best=max(rows,key=lambda r:(r['success'],r['margin_score']))
        result=dict(completed=True,rows=rows,best=best,improved_over_initial=bool(best['label']!='initial' and best['margin_score']>rows[0]['margin_score']+1e-5),
            protocol_sha256=sha(a.protocol),scope='Development checkpoint selection on the known highest case. Every checkpoint reported. Cached suffix only; selected candidate still needs five complete cycles and regression checks. No automatic replacement.')
        (a.output/'summary.json').write_text(json.dumps(result,indent=2))
    finally:env.close()

if __name__=='__main__':main()
