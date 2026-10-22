"""Export a PPO development actor and verify it through the real adapter.

Frozen training artifacts remain unchanged. Verification covers recorded inputs
and a separate MuJoCo rollout using Climb.run; it makes no hardware claim.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import numpy as np
import onnx
import onnxruntime
import torch
from stable_baselines3 import PPO
from common.ctrlcomp import StateAndCmd, PolicyOutput
from policy.climb.Climb import Climb, OnnxActor
from training.climb_env import ClimbSupportEnv, DEFAULT_BUNDLE

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DeterministicActor(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        # PPO.predict clips the Gaussian mean for this unsquashed Box policy.
        features = self.policy.extract_features(obs)
        latent = self.policy.mlp_extractor.forward_actor(features)
        return torch.clamp(self.policy.action_net(latent), -1., 1.)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT/'outputs/climb_ppo_smoke_v1')
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    out = args.out or args.run/'export'
    if out.exists():
        raise FileExistsError('Choose a new export directory')
    manifest = json.loads((args.run/'manifest.json').read_text())
    summary = json.loads((args.run/'summary.json').read_text())
    if not summary['training_completed'] or sha(args.run/'policy.zip') != summary['checkpoint_sha256']:
        raise ValueError('Training checkpoint integrity mismatch')
    if sha(args.bundle/'manifest.json') != manifest['bundle_manifest_sha256']:
        raise ValueError('Training and export scene bundles differ')
    for name, expected in manifest['sources'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Training source changed: '+name)
    torch.set_num_threads(1)
    env = ClimbSupportEnv(args.bundle)
    contract = env.contract
    if contract.fingerprint != manifest['contract_sha256'] or manifest['observation_normalization']:
        raise ValueError('Unsupported contract or observation normalization')
    model = PPO.load(args.run/'policy.zip', device='cpu')
    model.policy.set_training_mode(False)
    if model.policy.squash_output or not model.policy.share_features_extractor:
        raise ValueError('Export supports only shared features and unsquashed PPO')
    out.mkdir(parents=True)
    model_path = out/'policy.onnx'
    torch.onnx.export(DeterministicActor(model.policy), torch.zeros((1, 100), dtype=torch.float32),
        str(model_path), input_names=['obs'], output_names=['actions'], opset_version=17,
        dynamo=False)
    onnx.checker.check_model(str(model_path))
    (out/'contract.json').write_bytes((args.bundle/'contract.json').read_bytes())
    metadata = dict(contract_sha256=contract.fingerprint, model_sha256=sha(model_path),
        checkpoint_sha256=summary['checkpoint_sha256'], input_shape=[1, 100], output_shape=[1, 29],
        action='clipped deterministic PPO mean', task='initial_support_pose',
        training_steps=summary['actual_steps'], hardware_validated=False)
    metadata_path = out/'metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2))
    actor = OnnxActor(model_path, metadata_path, contract)
    maximum_action_error = maximum_target_error = 0.
    observation_count = 0
    for label in ('before', 'after'):
        with np.load(args.run/(label+'.npz'), allow_pickle=False) as data:
            for obs in data['observations']:
                expected, _ = model.predict(obs, deterministic=True)
                actual = actor(obs)
                np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)
                expected_target, _ = contract.decode_action(expected)
                actual_target, _ = contract.decode_action(actual)
                np.testing.assert_allclose(actual_target, expected_target, rtol=0, atol=2e-6)
                maximum_action_error = max(maximum_action_error, float(np.max(abs(actual-expected))))
                maximum_target_error = max(maximum_target_error, float(np.max(abs(actual_target-expected_target))))
                observation_count += 1

    # A live simulation rollout also exercises field conversion, action history,
    # loaded ONNX inference, and PolicyOutput gains through the deployment class.
    state, output = StateAndCmd(29), PolicyOutput(29)
    adapter = Climb(state, output, config_path=out/'contract.json',
                    model_path=model_path, metadata_path=metadata_path)
    obs, _ = env.reset(seed=20261009)
    adapter.enter()
    observations = [obs.copy()]
    qpos = [env.data.qpos.copy()]
    actions, targets, rewards, times = [], [], [], [0.]
    live_action_error = observation_error = 0.
    for _ in range(env.max_steps):
        rotation = env.data.xmat[env.pelvis].reshape(3, 3)
        state.q = env.data.qpos[env.qadr].copy()
        state.dq = env.data.qvel[env.vadr].copy()
        state.gravity_ori = rotation.T @ np.array([0., 0., -1.])
        state.ang_vel = env.data.qvel[3:6].copy()
        state.climb_target_pos_b = rotation.T @ (env.data.geom_xpos[env.target]-env.data.xpos[env.pelvis])
        state.climb_target_axis_b = rotation.T @ env.data.geom_xmat[env.target].reshape(3, 3)[:, 2]
        state.climb_target_valid = True
        adapter.run()
        if adapter.fault:
            raise RuntimeError(adapter.fault)
        np.testing.assert_array_equal(adapter.last_observation, obs)
        observation_error = max(observation_error, float(np.max(abs(adapter.last_observation-obs))))
        expected_action, _ = model.predict(obs, deterministic=True)
        action = adapter.previous_action.copy()
        np.testing.assert_allclose(action, expected_action, rtol=0, atol=2e-6)
        live_action_error = max(live_action_error, float(np.max(abs(action-expected_action))))
        target, _ = contract.decode_action(action)
        np.testing.assert_array_equal(output.actions, target)
        np.testing.assert_array_equal(output.kps, contract.kps)
        np.testing.assert_array_equal(output.kds, contract.kds)
        targets.append(output.actions.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs.copy()); qpos.append(env.data.qpos.copy())
        actions.append(action); rewards.append(reward); times.append(float(env.data.time))
        if terminated or truncated:
            break
    np.savez_compressed(out/'adapter_rollout.npz', observations=observations, qpos=qpos,
        actions=actions, targets=targets, rewards=rewards, time_s=times)
    with np.load(args.run/'after.npz', allow_pickle=False) as prior:
        trajectory_error = (float(np.max(abs(prior['qpos']-np.asarray(qpos))))
                            if prior['qpos'].shape == np.asarray(qpos).shape else None)
    adapter.exit(); env.close()
    report = dict(export_verified=True, recorded_observations=observation_count,
        recorded_action_max_abs_error=maximum_action_error,
        recorded_target_max_abs_error_rad=maximum_target_error,
        live_action_max_abs_error=live_action_error, adapter_observation_max_abs_error=observation_error,
        adapter_rollout=dict(steps=len(actions), episode_reward=float(np.sum(rewards)),
            terminated=terminated, truncated=truncated, final=info),
        reference_qpos_max_abs_difference=trajectory_error,
        torch_version=torch.__version__, onnx_version=onnx.__version__, onnxruntime_version=onnxruntime.__version__,
        tolerance_action_abs=2e-6, training_manifest_sha256=sha(args.run/'manifest.json'),
        training_summary_sha256=sha(args.run/'summary.json'),
        sources={s: sha(ROOT/s) for s in ('tools/export_climb_policy.py', 'policy/climb/Climb.py',
                                         'training/climb_env.py', 'policy/climb/contract.py')},
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        scope='Recorded inputs and one deterministic development rollout; no robustness or hardware claim')
    (out/'verified_export.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
