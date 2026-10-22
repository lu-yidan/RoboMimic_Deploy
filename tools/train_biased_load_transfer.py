"""Predeclared PPO study with zero/before/after and held-out angle evaluations."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from training.biased_load_transfer_residual_env import BiasedLoadTransferResidualEnv, CASES, TRAIN_ANGLES, EVAL_ANGLES, scenario_name


def evaluate(model, suite, out, label, seed):
    env = BiasedLoadTransferResidualEnv(suite)
    results = []
    for angle in EVAL_ANGLES:
        for case in CASES:
            name = scenario_name(case, angle)
            obs, _ = env.reset(seed=seed, options={'case': case, 'normal_bias_y_deg': angle})
            q = [env.env.data.qpos.copy()]; v = [env.env.data.qvel.copy()]; times = [float(env.env.data.time)]
            actions = []; observations = []; torques = []; rewards = []
            for _ in range(env.max_steps):
                observations.append(obs.copy())
                action = np.zeros(6) if model is None else model.predict(obs, deterministic=True)[0]
                obs, reward, term, trunc, info = env.step(action)
                actions.append(action); rewards.append(reward); torques.append(env.last_torque.copy())
                q.append(env.env.data.qpos.copy()); v.append(env.env.data.qvel.copy()); times.append(float(env.env.data.time))
                if term or trunc:
                    break
            np.savez_compressed(out / f'{label}_{name}.npz', qpos=q, qvel=v, time_s=times, actions=actions, observations=observations, torques=torques, rewards=rewards)
            results.append(dict(**info, evaluation_only_angle=angle not in TRAIN_ANGLES, return_sum=float(sum(rewards))))
            (out / (label + '.json')).write_text(json.dumps(results, indent=2))
            print(json.dumps(dict(evaluation=label, scenario=name, success=info['success'], reason=info['reason'], time_s=info['time_s'])), flush=True)
    env.close()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=8192)
    p.add_argument('--seed', type=int, default=20261017)
    a = p.parse_args()
    if a.steps < 256 or a.steps % 256:
        raise ValueError('steps must be a positive multiple of256')
    base = BiasedLoadTransferResidualEnv(a.suite)
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    sources = {}
    for rel in ['training/biased_load_transfer_residual_env.py', 'training/contact_load_transfer_residual_env.py', 'training/contact_foot_residual_env.py', 'training/contact_frame_controller.py', 'training/biased_contact_frame_controller.py', 'training/rung_residual_env.py', 'training/climb_cycle.py', 'training/climb_robustness.py', 'training/wall_support_env.py', 'training/climb_env.py', 'tools/train_biased_load_transfer.py']:
        target = a.out / 'source_snapshot' / rel
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / rel, target)
        sources[rel] = hashlib.sha256(target.read_bytes()).hexdigest()
    config = dict(n_steps=256, batch_size=64, n_epochs=5, learning_rate=3e-4, gamma=.995, gae_lambda=.95, policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), log_std_init=-2.))
    manifest = dict(seed=a.seed, steps=a.steps, config=config, sources=sources, cases=CASES, suite=str(a.suite.resolve()), torch=torch.__version__, sb3=stable_baselines3.__version__,
                    prefixes={name: hashlib.sha256((a.suite / name / 'trajectory.npz').read_bytes()).hexdigest() for name in CASES},
                    train_angles_deg=TRAIN_ANGLES, evaluation_angles_deg=EVAL_ANGLES,
                    training_distribution=[dict(case=case, normal_bias_y_deg=angle, probability=float(prob)) for (case, angle), prob in zip(base.scenarios, base.probabilities)],
                    contract='76obs/6right-leg actions, filtered residual capped0.015rad; native motor bounds and per0.5ms position check',
                    actor='No bias label or true rod coordinates. Ideal force feedback retained. Controller uses exact topology/points and biased normals.',
                    scope='One seed, finite-prefix load transfer. Train on0,+/-3deg with known failingZ+5/-3deg sampled50%; +/-1.5deg evaluation-only. Report every scenario, not only aggregate or training reward. No full-cycle or real-sensor claim.')
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    env = Monitor(base, str(a.out / 'episodes'), info_keywords=('scenario', 'normal_bias_y_deg', 'minimum_native_joint_margin_rad'))
    model = PPO('MlpPolicy', env, seed=a.seed, device='cpu', verbose=0, **config)
    model.set_logger(configure(str(a.out), ['csv']))
    model.save(a.out / 'policy_before')
    zero = evaluate(None, a.suite, a.out, 'zero', a.seed)
    before = evaluate(model, a.suite, a.out, 'before', a.seed)
    start = time.perf_counter(); model.learn(total_timesteps=a.steps); elapsed = time.perf_counter() - start
    model.save(a.out / 'policy')
    after = evaluate(model, a.suite, a.out, 'after', a.seed)
    loaded = PPO.load(a.out / 'policy.zip', device='cpu')
    difference = 0.
    for row in after:
        z = dict(np.load(a.out / f"after_{row['scenario']}.npz"))
        x = model.predict(z['observations'], deterministic=True)[0]
        y = loaded.predict(z['observations'], deterministic=True)[0]
        difference = max(difference, float(abs(x - y).max()))
    if difference != 0.:
        raise AssertionError('Checkpoint reload changed actions')
    result = dict(training_completed=True, actual_steps=model.num_timesteps, training_wall_s=elapsed, zero=zero, before=before, after=after, checkpoint_reload_max_difference=difference, checkpoint_sha256=hashlib.sha256((a.out / 'policy.zip').read_bytes()).hexdigest(), learned_full_cycle=False, multi_seed_validated=False)
    (a.out / 'summary.json').write_text(json.dumps(result, indent=2)); env.close()
    print(json.dumps(dict(training_completed=True, actual_steps=model.num_timesteps, training_wall_s=elapsed)), flush=True)


if __name__ == '__main__':
    main()
