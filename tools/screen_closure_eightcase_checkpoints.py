"""Declared, finite checkpoint selection across all eight development cases."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.closure_combined_curriculum_env import ClosureCombinedCurriculumEnv
from tools.continue_vector_full_cycle_training import sha


LABELS=['initial','collected_32768_before_update','collected_65536_before_update','collected_98304_before_update','final']


def main():
    parser=argparse.ArgumentParser()
    for name in ('experiment','protocol','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();declaration=json.loads(args.protocol.read_text())
    assert declaration['source_sha256']==sha(__file__) and declaration['labels']==LABELS
    assert Path(declaration['experiment']).resolve()==args.experiment.resolve()
    manifest=json.loads((args.experiment/'manifest.json').read_text());summary=json.loads((args.experiment/'summary.json').read_text())
    audit=json.loads((args.experiment/'independent_candidate_audit.json').read_text())
    assert audit['passed'] and not summary['evaluation_pending'] and summary['additional_steps']==98304
    assert audit['experiment_manifest_sha256']==sha(args.experiment/'manifest.json') and audit['experiment_summary_sha256']==sha(args.experiment/'summary.json')
    assert audit['checkpoint_sha256']==sha(args.experiment/'policy.zip')
    assert declaration['manifest_sha256']==sha(args.experiment/'manifest.json')
    known=set(declaration['known_cases']);assert len(known)==5 and known < set(manifest['evaluation_cases'])
    torch.set_num_threads(1);args.output.mkdir(exist_ok=False)
    env=ClosureCombinedCurriculumEnv(manifest['prefix_cache'],controller_modifier=manifest['controller_modifier'])
    rows=[]
    try:
        for label in LABELS:
            filename='policy_before.zip' if label=='initial' else 'policy.zip' if label=='final' else f"checkpoint_additional_{label.split('_')[1]}.zip"
            path=args.experiment/filename;model=PPO.load(path,device='cpu');case_rows=[]
            destination=args.output/label;destination.mkdir()
            for case in manifest['evaluation_cases']:
                obs,_=env.reset(options={'case':case});base=env.env.unwrapped
                trace=dict(qpos=[base.env.data.qpos.copy()],qvel=[base.env.data.qvel.copy()],time_s=[float(base.env.data.time)],actions=[])
                while True:
                    action=model.predict(obs,deterministic=True)[0];obs,_,done,truncated,info=env.step(action)
                    trace['actions'].append(action);trace['qpos'].append(base.env.data.qpos.copy());trace['qvel'].append(base.env.data.qvel.copy());trace['time_s'].append(float(base.env.data.time))
                    if done or truncated:break
                arrays={key:np.asarray(value) for key,value in trace.items()};tail=arrays['time_s']>111.66;q=arrays['qpos'][tail]
                complete=len(arrays['actions'])==700 and len(q)>=99
                closure=float(abs(q[:,base.env.qadr]-base.env.initial[base.env.qadr]).max()) if complete else None
                height_error=float(abs(q[:,2]-base.env.initial[2]-.25).max()) if complete else None
                margin=min((.04-closure)/.001,(.01-height_error)/.0001) if complete else -1e9
                if label=='initial':
                    with np.load(Path(manifest['parent_experiment'])/f'after_{case}.npz') as original:
                        assert np.max(abs(original['qpos'][4983:]-arrays['qpos']))<1e-10
                        assert np.max(abs(original['qvel'][4983:]-arrays['qvel']))<1e-8
                target=destination/f'{case}.npz';np.savez_compressed(target,**arrays)
                row=dict(case=case,success=bool(info['success']),complete=complete,closure_rad=closure,height_error_m=height_error,margin_score=margin,trajectory_sha256=sha(target))
                case_rows.append(row);(destination/f'{case}.json').write_text(json.dumps(row,indent=2))
                print(json.dumps(dict(label=label,**row)),flush=True)
            row=dict(label=label,checkpoint=str(path.resolve()),checkpoint_sha256=sha(path),cases=case_rows,
                known_preserved=all(r['success'] for r in case_rows if r['case'] in known),successes=sum(r['success'] for r in case_rows),minimum_margin=min(r['margin_score'] for r in case_rows))
            rows.append(row);(destination/'summary.json').write_text(json.dumps(row,indent=2))
        best=max(rows,key=lambda r:(r['known_preserved'],r['successes'],r['minimum_margin']))
        result=dict(completed=True,rows=rows,best=best,protocol_sha256=sha(args.protocol),
            scope='Eight-case training/development checkpoint selection. Cached suffix only; selected weights require complete from-zero evaluation before any promotion. All declared outcomes retained.')
        (args.output/'summary.json').write_text(json.dumps(result,indent=2))
    finally:env.close()


if __name__=='__main__':main()
