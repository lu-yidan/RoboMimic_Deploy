"""Freeze exploration scale selected by the predeclared pilot; means unchanged."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from tools.continue_vector_full_cycle_training import sha,state_difference


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    s=json.loads((a.pilot/'summary.json').read_text());protocol=json.loads((a.pilot/'protocol.json').read_text())
    assert s['completed'] and s['eligible_for_training_preparation'] and sha(a.pilot/'protocol.json')==s['protocol_sha256']
    parent=Path(protocol['parent']);assert sha(parent/'policy.zip')==protocol['parent_checkpoint_sha256']
    scale=s['selected_scale'];torch.set_num_threads(1)
    model=PPO.load(parent/'policy.zip',device='cpu');original=PPO.load(parent/'policy.zip',device='cpu')
    with torch.no_grad():model.policy.log_std.add_(np.log(scale))
    model.policy.optimizer.state.pop(model.policy.log_std,None)
    unchanged=all(torch.equal(v,model.policy.state_dict()[k]) for k,v in original.policy.state_dict().items() if k!='log_std');assert unchanged
    a.output.mkdir(exist_ok=False);model.save(a.output/'policy_initial.zip');loaded=PPO.load(a.output/'policy_initial.zip',device='cpu')
    assert state_difference(model.policy.state_dict(),loaded.policy.state_dict())==0.
    m=json.loads((parent/'manifest.json').read_text());difference=0.
    for case in m['evaluation_cases']:
        with np.load(parent/f'after_{case}.npz') as z:
            for obs in z['observations'][4983::7]:difference=max(difference,float(abs(original.predict(obs,deterministic=True)[0]-loaded.predict(obs,deterministic=True)[0]).max()))
    assert difference==0.
    result=dict(passed=True,scale=scale,deterministic_action_max_difference=difference,parameters_except_log_std_exact=unchanged,
        log_std_optimizer_moments_cleared=True,checkpoint_sha256=sha(a.output/'policy_initial.zip'),parent=str(parent.resolve()),
        parent_checkpoint_sha256=sha(parent/'policy.zip'),pilot_summary_sha256=sha(a.pilot/'summary.json'),controller_modifier=m['controller_modifier'],
        sources={name:sha(name) for name in ('training/closure_seeded_policy.py','training/closure_exploration_policy.py','training/closure_combined_curriculum_env.py','tools/prepare_closure_combined_training_seed.py')},
        scope='Sampling variance initialization only. Deterministic network and selected controller unchanged; no PPO steps.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
