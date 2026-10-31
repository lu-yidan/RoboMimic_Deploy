"""Freeze the best measured combined-control seed; no optimizer update."""
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from tools.continue_vector_full_cycle_training import sha,state_difference


def main():
    torch.set_num_threads(1)
    search=Path('outputs/closure_combined_margin_search_20261024_v1');s=json.loads((search/'summary.json').read_text());best=s['best']
    assert s['completed'] and best['success'] and best['margin_score']>.0931647272
    protocol=json.loads((search/'protocol.json').read_text());assert sha(search/'protocol.json')==s['protocol_sha256']
    for path,value in protocol['source_sha256'].items():assert sha(path)==value
    original=Path('outputs/closure_seed_policy_20261024_v1/policy_initial.zip');assert sha(original)==protocol['seed_policy_sha256']
    out=Path('outputs/closure_combined_margin_seed_20261024_v1');out.mkdir(exist_ok=False)
    model=PPO.load(original,device='cpu');previous={k:v.clone() for k,v in model.policy.state_dict().items()}
    with torch.no_grad():model.policy.closure_seed_bias.copy_(torch.as_tensor(best['seed_bias'],dtype=torch.float64))
    model.policy_kwargs['closure_seed_bias']=best['seed_bias']
    unchanged=state_difference({k:v for k,v in previous.items() if k!='closure_seed_bias'}, {k:v for k,v in model.policy.state_dict().items() if k!='closure_seed_bias'})==0.
    assert unchanged
    model.save(out/'policy_initial.zip');loaded=PPO.load(out/'policy_initial.zip',device='cpu')
    assert state_difference(model.policy.state_dict(),loaded.policy.state_dict())==0.
    trajectory=search/f"trial_{best['trial']:03d}.npz";assert sha(trajectory)==best['trajectory_sha256']
    with np.load(trajectory) as z:
        replay=np.asarray([loaded.predict(obs,deterministic=True)[0] for obs in z['observations']]);error=float(abs(replay-z['actions']).max());assert error==0.
    sources={p:sha(p) for p in ('training/closure_seeded_policy.py','training/closure_exploration_policy.py','training/closure_waist_target.py','training/closure_body_target.py','tools/prepare_closure_combined_margin_seed.py')}
    result=dict(passed=True,checkpoint_sha256=sha(out/'policy_initial.zip'),sources=sources,search_summary_sha256=sha(search/'summary.json'),
        seed_bias=best['seed_bias'],seed_ramp_seconds=1.,network_parameters_unchanged=unchanged,action_replay_max_error=error,
        controller_modifier=dict(waist_bias=best['waist_bias'],body_drop_m=best['body_drop_m'],waist_ramp_seconds=1.),
        scope='Fixed search seed only. Physical suffix passed; full-cycle five-case and regression validation still required. No PPO steps.')
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
