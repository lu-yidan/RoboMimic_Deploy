"""Compare matched random episodes under two fixed reference seeds."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from tools.compare_audited_closure_candidates import sha
from tools.continue_vector_full_cycle_training import state_difference


def main():
    p=argparse.ArgumentParser()
    for key in ('before','after','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    paths=[a.before,a.after];summaries=[];protocols=[];models=[];seeds=[]
    for path in paths:
        s=json.loads((path/'summary.json').read_text());protocol=json.loads((path/'protocol.json').read_text())
        assert s['passed'] and s['protocol_sha256']==sha(path/'protocol.json')
        seed=Path(protocol['seed']);assert sha(seed/'policy_initial.zip')==protocol['seed_checkpoint_sha256']
        assert sha(seed/'protocol.json')==protocol['seed_protocol_sha256']
        summaries.append(s);protocols.append(protocol);seeds.append(json.loads((seed/'protocol.json').read_text()))
        models.append(PPO.load(seed/'policy_initial.zip',device='cpu'))
    assert state_difference(models[0].policy.state_dict(),models[1].policy.state_dict())==0
    for key in ('teacher_sha256','cases','gamma','episodes_per_case','split'):assert protocols[0][key]==protocols[1][key]
    for key in ('delta_scale','hold_steps','physical_gamma','controller_modifier','config'):assert seeds[0][key]==seeds[1][key]
    previous={row['index']:row for row in summaries[0]['rows']};rows=[]
    for new in summaries[1]['rows']:
        old=previous[new['index']]
        for key in ('case','seed','split','steps'):assert old[key]==new[key]
        files=[path/row['file'] for path,row in zip(paths,(old,new))]
        assert all(sha(file)==row['trajectory_sha256'] for file,row in zip(files,(old,new)))
        with np.load(files[0]) as x,np.load(files[1]) as y:
            exact=dict(actions=bool(np.array_equal(x['actions'],y['actions'])),
                initial_qpos=bool(np.array_equal(x['qpos'][0],y['qpos'][0])),
                initial_qvel=bool(np.array_equal(x['qvel'][0],y['qvel'][0])),
                times=bool(np.array_equal(x['time_s'],y['time_s'])))
        assert all(exact.values())
        rows.append(dict(index=new['index'],case=new['case'],seed=new['seed'],before_success=old['success'],after_success=new['success'],exact=exact))
    assert len(rows)==len(previous)==24
    counts={case:dict(before=sum(row['before_success'] for row in rows if row['case']==case),after=sum(row['after_success'] for row in rows if row['case']==case),episodes=3) for case in protocols[0]['cases']}
    result=dict(passed=True,rows=rows,by_case=counts,successes=[s['successes'] for s in summaries],
        input_summary_sha256=[sha(path/'summary.json') for path in paths],source_sha256=sha(__file__),
        scope='Same random actions and initial states, different fixed reference seeds. Cached 14-second stochastic suffixes, not new policy learning or independent geometry evaluation.')
    a.output.mkdir(exist_ok=False);(a.output/'summary.json').write_text(json.dumps(result,indent=2));(a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(passed=True,successes=result['successes'],by_case=counts)))


if __name__=='__main__':main()
