"""Inspect raw Gaussian saturation separately from calibrated executed actions."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
import torch
from stable_baselines3 import PPO
from tools.compare_audited_closure_candidates import load_run, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--case',default='boundary_07');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);rows=[]
    for directory in a.runs:
        verified=load_run(directory);assert verified['rows'][a.case]['complete']
        manifest=json.loads((directory/'manifest.json').read_text())
        trace=directory/f'after_{a.case}.npz'
        with np.load(trace) as z:
            mask=z['time_s'][:-1]>manifest['horizon_seconds']-2
            obs=z['observations'][mask];actual=z['actions'][mask]
        model=PPO.load(directory/'policy.zip',device='cpu')
        assert not model.policy.squash_output
        with torch.no_grad():
            distribution=model.policy.get_distribution(torch.as_tensor(obs,device=model.device)).distribution
            means=distribution.mean.cpu().numpy();std=distribution.stddev.cpu().numpy()
        joints={}
        for i,name in [(3,'left_knee'),(4,'left_ankle_pitch')]:
            joints[name]=dict(raw_mean_min=float(means[:,i].min()),raw_mean_max=float(means[:,i].max()),
                gaussian_std_min=float(std[:,i].min()),gaussian_std_max=float(std[:,i].max()),
                deterministic_upper_clipped_fraction=float(np.mean(means[:,i]>1)),
                predicted_stochastic_upper_clip_probability=float(ndtr((means[:,i]-1)/std[:,i]).mean()),
                predicted_stochastic_lower_clip_probability=float(ndtr((-1-means[:,i])/std[:,i]).mean()),
                executed_min=float(actual[:,i].min()),executed_max=float(actual[:,i].max()))
        rows.append(dict(directory=str(directory.resolve()),checkpoint_sha256=sha(directory/'policy.zip'),
            trajectory_sha256=sha(trace),independent_audit_sha256=verified['audit_sha256'],case=a.case,
            samples=len(obs),joint_statistics=joints))
    result=dict(rows=rows,source_sha256=sha(__file__),
        scope='Saved terminal-state conditional raw Gaussian distribution. Clip probabilities are analytically predicted, not empirically sampled. Deterministic clipping is distinct from physical torque saturation. Nonzero probability below the action bound means exploration is not eliminated; no causal proof of training failure.')
    a.output.mkdir(exist_ok=False);(a.output/'summary.json').write_text(json.dumps(result,indent=2))
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result,indent=2))


if __name__=='__main__':main()
