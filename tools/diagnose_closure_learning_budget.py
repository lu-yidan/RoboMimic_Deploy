"""Read-only accounting of new episodes, PPO updates and direct GAE weights."""
import argparse
from collections import Counter
import csv
import inspect
import json
import math
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.base_class import BaseAlgorithm
from tools.continue_vector_full_cycle_training import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=Path)
    args = parser.parse_args()
    p = args.experiment
    manifest = json.loads((p/'manifest.json').read_text())
    summary = json.loads((p/'summary.json').read_text())
    assert summary['additional_steps'] == manifest['additional_steps_requested']
    torch.set_num_threads(1)
    before = PPO.load(p/'policy_before.zip', device='cpu')
    after = PPO.load(p/'policy.zip', device='cpu')
    episodes = [json.loads(line) for line in (p/'training_episodes.jsonl').read_text().splitlines()]
    assert len(episodes) == summary['completed_training_episodes']
    assert sum(e['success'] for e in episodes) == summary['successful_training_episodes']
    progress = list(csv.DictReader((p/'progress.csv').open(newline='')))
    ratio = float(before.gamma*before.gae_lambda)
    direct_weights = {str(seconds): ratio**round(seconds/.02) for seconds in (.2, .5, 1., 3., 5.)}
    metrics = {}
    for key in ('train/approx_kl', 'train/clip_fraction', 'train/explained_variance', 'train/std'):
        values = [float(row[key]) for row in progress if row.get(key)]
        assert all(math.isfinite(v) for v in values)
        metrics[key] = dict(min=min(values), median=float(np.median(values)), max=max(values), last=values[-1])
    episode_counts = Counter(e['case'] for e in episodes)
    successful = Counter(e['case'] for e in episodes if e['success'])
    gae_source = inspect.getsourcefile(RolloutBuffer)
    setup_source = inspect.getsourcefile(BaseAlgorithm)
    result = dict(
        completed_episodes=len(episodes), successful_episodes=sum(e['success'] for e in episodes),
        per_case={case: dict(completed=episode_counts[case], successful=successful[case], sampled_steps=summary['case_samples'].get(case, 0)) for case in manifest['training_cases']},
        new_episode_return_mean=float(np.mean([e['episode']['r'] for e in episodes])),
        inherited_rolling_episode_count=len(before.ep_info_buffer or []),
        inherited_rolling_episode_mean=float(np.mean([r['r'] for r in before.ep_info_buffer])) if before.ep_info_buffer else None,
        updates_before=int(before._n_updates), updates_after=int(after._n_updates),
        nominal_training_epochs=manifest['config']['n_epochs'], rollout_seconds_per_worker=before.n_steps*.02,
        gamma=float(before.gamma), gae_lambda=float(before.gae_lambda), direct_gae_ratio=ratio,
        direct_gae_weight_by_lag_seconds=direct_weights, direct_gae_efold_seconds=-.02/math.log(ratio),
        logged_optimizer_metrics=metrics,
        evidence={file: sha(p/file) for file in ('manifest.json', 'summary.json', 'policy_before.zip', 'policy.zip', 'progress.csv', 'training_episodes.jsonl')},
        installed_gae_source=dict(path=gae_source, sha256=sha(gae_source)),
        installed_setup_source=dict(path=setup_source, sha256=sha(setup_source)), source_sha256=sha(__file__),
        interpretation='Direct GAE weights describe within-rollout TD-residual weighting, not the total learned credit horizon. Critic bootstrapping and repeated updates can carry longer-range information. No causal diagnosis or hyperparameter change follows from this diagnostic alone. Rolling training means may include inherited episodes; new episodes above exclude them. Training success is not evaluation success.')
    target = p/'learning_budget_diagnostic.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2))
    (p/'learning_budget_diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: result[k] for k in ('completed_episodes', 'per_case', 'inherited_rolling_episode_count', 'updates_before', 'updates_after', 'direct_gae_efold_seconds', 'direct_gae_weight_by_lag_seconds')}, indent=2))


if __name__ == '__main__':
    main()
