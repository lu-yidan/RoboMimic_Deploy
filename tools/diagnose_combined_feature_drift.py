"""Offline attribution of actor changes; does not replay or modify training."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from tools.continue_vector_full_cycle_training import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    m=json.loads((a.experiment/'manifest.json').read_text());parent=Path(m['parent_experiment'])
    initial=PPO.load(a.experiment/'policy_before.zip',device='cpu');current=PPO.load(a.checkpoint,device='cpu')
    geometry_only=PPO.load(a.experiment/'policy_before.zip',device='cpu');rest_only=PPO.load(a.checkpoint,device='cpu')
    assert initial.policy.share_features_extractor and current.policy.share_features_extractor
    initial_weight=initial.policy.features_extractor.geometry_projection.weight.detach().clone()
    current_weight=current.policy.features_extractor.geometry_projection.weight.detach().clone()
    with torch.no_grad():
        geometry_only.policy.features_extractor.geometry_projection.weight.copy_(current_weight)
        rest_only.policy.features_extractor.geometry_projection.weight.copy_(initial_weight)
    observations=[]
    for case in m['evaluation_cases']:
        with np.load(parent/f'after_{case}.npz') as z:observations.extend(z['observations'][4983::7])
    arrays={}
    for name,model in [('initial',initial),('current',current),('geometry_only',geometry_only),('rest_only',rest_only)]:arrays[name]=np.asarray([model.predict(obs,deterministic=True)[0] for obs in observations])
    metrics={name:dict(action_delta_rms=float(np.sqrt(np.mean((value-arrays['initial'])**2))),action_delta_max=float(abs(value-arrays['initial']).max())) for name,value in arrays.items() if name!='initial'}
    a.output.mkdir(exist_ok=False)
    result=dict(metrics=metrics,shared_geometry_weight_change_norm=float(torch.linalg.vector_norm(current_weight-initial_weight)),
        checkpoint_sha256=sha(a.checkpoint),initial_checkpoint_sha256=sha(a.experiment/'policy_before.zip'),observations=len(observations),source_sha256=sha(__file__),
        scope='Fixed saved development observations, deterministic action counterfactuals. Nonlinear/clipped contributions are not additive. This does not identify whether actor or critic gradients caused shared-feature updates, nor establish physical improvement.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
