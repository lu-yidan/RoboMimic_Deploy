"""Measure policy drift on fixed baseline observations; not a rollout result."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=Path)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    path = args.experiment/args.checkpoint
    model = PPO.load(path, device='cpu')
    cases = []
    for trajectory in sorted(args.experiment.glob('zero_*.npz')):
        with np.load(trajectory) as z:
            actions = model.predict(z['observations'], deterministic=True)[0]
            times = z['time_s'][:-1]
        windows = []
        for start, end in ((0., 29.), (29., 49.), (49., 61.56), (61.56, 96.72), (96.72, 110.73)):
            selected = actions[(times >= start)&(times < end)]
            if len(selected):
                windows.append(dict(start_s=start, end_s=end, samples=len(selected),
                    maximum_action=float(abs(selected).max()), rms_action=float(np.sqrt(np.mean(selected**2)))))
        cases.append(dict(trajectory=trajectory.name, sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest(),
                          windows=windows))
    result = dict(checkpoint=args.checkpoint, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  action_std=model.policy.log_std.detach().cpu().exp().numpy().tolist(), cases=cases,
                  scope='Inference on fixed zero-controller observations. Does not prove closed-loop behavior, safety, or improvement.')
    output=args.experiment/(path.stem+'_fixed_observation_shift.json')
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
