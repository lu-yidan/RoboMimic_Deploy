"""Export a trusted RSL-RL 93D checkpoint, including its native normalizer.
Run in the training Python environment. Validate parity in the deployment environment.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    obs = TensorDict({'actor': torch.zeros(1, 93)}, batch_size=[1])
    actor = MLPModel(obs, {'actor': ['actor']}, 'actor', 29,
                     hidden_dims=[512, 256, 128], activation='elu', obs_normalization=True,
                     distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': .3,
                                       'std_type': 'scalar', 'learn_std': False})
    actor.load_state_dict(checkpoint['actor_state_dict'], strict=True)
    actor.eval()
    exported = actor.as_onnx(verbose=False).eval()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(exported, exported.get_dummy_inputs(), str(a.out),
                      input_names=['obs'], output_names=['actions'], opset_version=17, dynamo=False)
    generator = torch.Generator().manual_seed(20260919)
    x = torch.randn(512, 93, generator=generator)
    x[:, 35:64] *= 10  # Joint velocity component.
    with torch.inference_mode():
        y = actor(TensorDict({'actor': x}, batch_size=[512])).numpy()
    a.fixtures.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.fixtures, obs=x.numpy(), actions=y)
    manifest = {'source_checkpoint': str(a.checkpoint.resolve()),
                'checkpoint_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                'onnx_sha256': hashlib.sha256(a.out.read_bytes()).hexdigest(),
                'input': {'shape': [1, 93], 'normalizer_embedded': True},
                'output': {'shape': [1, 29], 'deterministic': True},
                'architecture': 'MLP 93-512-256-128-29, ELU', 'opset': 17,
                'parity_fixture': str(a.fixtures), 'hardware_validated': False}
    a.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

if __name__ == '__main__':
    main()
