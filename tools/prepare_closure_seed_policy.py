"""Restore original parameters/optimizer, add declared search bias, verify likelihoods."""
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.closure_seeded_policy import ClosureSeededPolicy
from tools.continue_vector_full_cycle_training import sha,state_difference


def main():
    torch.set_num_threads(1)
    parent=Path('outputs/closure_margin_control98304_v1')
    timing=Path('outputs/closure_action_timing_20261024_v1')
    study=json.loads((timing/'summary.json').read_text());assert study['completed']
    bias=np.asarray(study['basis'])@np.asarray(study['best']['parameters'])
    assert abs(bias).max()<=1.+1e-9;bias=np.clip(bias,-1.,1.)
    duration=study['best']['ramp_seconds']
    original=PPO.load(parent/'policy.zip',device='cpu')
    model=PPO.load(parent/'policy.zip',device='cpu')
    model.policy_class=ClosureSeededPolicy
    model.policy_kwargs=dict(model.policy_kwargs,closure_seed_bias=bias.tolist(),seed_ramp_seconds=duration)
    model.policy=ClosureSeededPolicy(model.observation_space,model.action_space,model.lr_schedule,use_sde=model.use_sde,**model.policy_kwargs).to(model.device)
    missing,unexpected=model.policy.load_state_dict(original.policy.state_dict(),strict=False)
    assert missing==['closure_seed_bias'] and not unexpected
    model.policy.optimizer.load_state_dict(original.policy.optimizer.state_dict())
    assert state_difference({k:v for k,v in model.policy.state_dict().items() if k!='closure_seed_bias'},original.policy.state_dict())==0.
    assert state_difference(model.policy.optimizer.state_dict(),original.policy.optimizer.state_dict())==0.
    manifest=json.loads((parent/'manifest.json').read_text());rows=[]
    for case in manifest['evaluation_cases']:
        with np.load(parent/f'after_{case}.npz') as z:observations=z['observations'].copy()
        error=0.;unchanged=True
        for obs in observations:
            action=original.predict(obs,deterministic=True)[0]
            t=113.66*(1.-float(obs[105]));u=np.clip((t-99.66)/duration,0.,1.);ramp=u*u*(3.-2.*u)
            gain=np.clip((float(obs[138])-.625)/.125,0.,1.)
            expected=np.clip(action+ramp*gain*bias,-1.,1.).astype(np.float32)
            actual=model.predict(obs,deterministic=True)[0]
            error=max(error,float(abs(expected-actual).max()))
            if gain==0. or ramp==0.:unchanged=unchanged and np.array_equal(actual,action)
        assert error<=1e-7 and unchanged
        rows.append(dict(case=case,teacher_action_max_difference=error,inactive_region_exact=bool(unchanged)))
    sample=torch.as_tensor(observations[-5:].copy())
    sample[:,138]=.75;sample[:,105]=0.
    actions,values,logp=model.policy(sample)
    ev,lp,entropy=model.policy.evaluate_actions(sample,actions.detach())
    assert torch.equal(values,ev) and torch.equal(logp,lp)
    distribution=model.policy.get_distribution(sample).distribution
    manual=distribution.log_prob(actions).sum(-1);assert torch.allclose(manual,logp,atol=1e-6)
    model.policy.optimizer.zero_grad();(-lp.mean()-.001*entropy.mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.policy.parameters() if p.grad is not None)
    model.policy.optimizer.zero_grad()
    out=Path('outputs/closure_seed_policy_20261024_v1');out.mkdir(exist_ok=False)
    model.save(out/'policy_initial');restored=PPO.load(out/'policy_initial.zip',device='cpu')
    assert state_difference(restored.policy.state_dict(),model.policy.state_dict())==0.
    assert state_difference(restored.policy.optimizer.state_dict(),model.policy.optimizer.state_dict())==0.
    result=dict(passed=True,rows=rows,seed_bias=bias.tolist(),seed_ramp_seconds=duration,
      original_parameters_and_optimizer_exact=True,initial_distribution_changed=True,
      parent_checkpoint_sha256=sha(parent/'policy.zip'),checkpoint_sha256=sha(out/'policy_initial.zip'),
      timing_study_sha256=sha(timing/'summary.json'),seed_suffix_success=study['best']['success'],
      sources={name:sha(name) for name in [str(Path(__file__).resolve()),'training/closure_seeded_policy.py','training/closure_exploration_policy.py']},
      scope='Known-case search initialization remains a failed suffix; no policy success claim. Fixed feedforward activates from +12.5 to +15 mm using existing oracle geometry. Actor/critic parameters and optimizer restored; Gaussian means intentionally shifted. Additional search samples must be counted separately from PPO.')
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    (out/'preparation_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
