"""Initialize a zero-delta feedback actor after the physics interface gate."""
import argparse
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from tools.compare_audited_closure_candidates import load_run, sha


class PolicySpaces(gym.Env):
    """Space metadata only; this object cannot generate training experience."""
    observation_space = gym.spaces.Box(-np.inf, np.inf, (153,), dtype=np.float32)
    action_space = gym.spaces.Box(-1., 1., (5,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(153, dtype=np.float32), {}

    def step(self, action):
        raise RuntimeError('PolicySpaces is not a simulator or training environment')


def main():
    p = argparse.ArgumentParser()
    for name in ('parent', 'gate', 'shards', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    parent = load_run(a.parent)
    m = json.loads((a.parent/'manifest.json').read_text())
    gate = json.loads((a.gate/'summary.json').read_text())
    assert gate['passed'] and gate['parent_audit_sha256'] == parent['audit_sha256']
    assert gate['shards_summary_sha256'] == sha(a.shards/'summary.json')
    assert len(gate['rows']) == 8 and all(r['passed'] for r in gate['rows'])
    for name, digest in gate['sources'].items():
        assert sha(name) == digest
    config = dict(n_steps=128, batch_size=64, n_epochs=5, gamma=.9995**5,
                  gae_lambda=.95, learning_rate=.0001, target_kl=.01,
                  seed=20261025, device='cpu', policy_kwargs=dict(
                      net_arch=dict(pi=[64, 64], vf=[64, 64]), log_std_init=float(np.log(.1))))
    model = PPO('MlpPolicy', PolicySpaces(), **config)
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
    samples = []
    for directory in sorted(a.gate.glob('worker_*')):
        if not directory.is_dir():
            continue
        for file in sorted(directory.glob('*.npz')):
            with np.load(file) as z:
                base = z['observations'][::5]
                samples.append(np.c_[base, np.zeros((len(base), 5))].astype(np.float32))
    obs = np.concatenate(samples)
    actions = model.predict(obs, deterministic=True)[0]
    assert np.array_equal(actions, np.zeros_like(actions))
    a.output.mkdir(exist_ok=False)
    model.save(a.output/'policy_initial')
    reloaded = PPO.load(a.output/'policy_initial.zip', device='cpu')
    assert np.array_equal(actions, reloaded.predict(obs, deterministic=True)[0])
    sources = dict(gate['sources'])
    sources[str(Path(__file__).relative_to(Path.cwd()))] = sha(__file__)
    protocol = dict(parent=str(a.parent.resolve()), parent_audit_sha256=parent['audit_sha256'],
        teacher_checkpoint=str((a.parent/'policy.zip').resolve()), teacher_sha256=parent['checkpoint_sha256'],
        controller_modifier=m['controller_modifier'], seed_command=m['adaptive_reference_command'],
        delta_scale=.25, hold_steps=5, physical_gamma=.9995, config=config,
        cases=parent['cases'], worker_shards=str(a.shards.resolve()),
        shards_summary_sha256=sha(a.shards/'summary.json'), environment_gate_sha256=sha(a.gate/'summary.json'),
        sources=sources, observation_contract='seeded-reference-153obs-5delta-10hz-v1',
        scope='New 5D reference-feedback actor initialized to zero deltas; frozen 26D teacher and unchanged reference limits. No training or performance gain claimed. Initial critic is uncalibrated.')
    (a.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    result = dict(passed=True, zero_mean_exact=True, observations_checked=len(obs),
                  checkpoint_sha256=sha(a.output/'policy_initial.zip'),
                  protocol_sha256=sha(a.output/'protocol.json'), new_training_steps=0,
                  critic_calibrated=False)
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    for name in sources:
        target = a.output/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes())
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
