"""Check unchanged known low cases/prefix and declared intermediate activation."""
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.closure_geometry_bias_policy import ClosureGeometryBiasPolicy
from tools.continue_vector_full_cycle_training import sha


def main():
    torch.set_num_threads(1)
    parent=Path('outputs/closure_margin_control98304_v1')
    manifest=json.loads((parent/'manifest.json').read_text())
    policy=PPO.load(parent/'policy.zip',device='cpu')
    bias=np.linspace(-1.,1.,26)
    manifest.update(closure_action_bias=bias.tolist(),suffix_start_seconds=99.66,horizon_seconds=113.66,
      geometry_bias_activation=dict(lower_m=.0125,upper_m=.015,feature_index=138,scale_m=.02))
    wrapped=ClosureGeometryBiasPolicy(policy,manifest)
    rows=[]
    for case in manifest['evaluation_cases']:
        with np.load(parent/f'after_{case}.npz') as z:obs=z['observations'].copy()
        limit=4983 if case=='foot_z_plus15' else len(obs)
        actions=np.asarray([wrapped.predict(o,deterministic=True)[0] for o in obs[:limit]])
        original=np.asarray([policy.predict(o,deterministic=True)[0] for o in obs[:limit]])
        assert np.array_equal(actions,original)
        rows.append(dict(case=case,unchanged_action_samples=limit,exact=True))
    synthetic=obs[-1].copy();synthetic[105]=0.;synthetic[138]=.6875
    actual=wrapped.predict(synthetic,deterministic=True)[0]
    expected=np.clip(policy.predict(synthetic,deterministic=True)[0]+.5*bias,-1.,1.).astype(np.float32)
    assert np.array_equal(actual,expected) and abs(actual).max()<=1.
    out=Path('outputs/closure_geometry_bias_gate_20261024_v1');out.mkdir(exist_ok=False)
    result=dict(passed=True,rows=rows,intermediate_geometry_gain_checked=True,
      policy_source_sha256=sha('training/closure_geometry_bias_policy.py'),parent_checkpoint_sha256=sha(parent/'policy.zip'),source_sha256=sha(__file__),
      scope='Action-interface check on saved observations plus one synthetic midpoint. No new physical success claim.')
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    (out/'gate_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
