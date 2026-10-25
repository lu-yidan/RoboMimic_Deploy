"""Load official ScaleBFM weights using its export wrapper, without TensorRT.

The upstream onboard exporter includes a CLI and a TensorRT compiler. Only its
imports and model definitions are loaded here; neither CLI nor compiler runs.
"""
import ast
import hashlib
import __future__
import json
from pathlib import Path
import sys
import types

import torch


def _export_definitions(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != 'df14138699427ab602a83286efdd80277ed0e7ae81a70e26221375e1600eebe6':
        raise ValueError('ScaleBFM exporter differs from validated source; run tools/motion/setup_scalebfm.py with a fresh external directory')
    name = '_lyd_scalebfm_export_' + str(abs(hash(str(path))))
    if name in sys.modules:
        return sys.modules[name]
    tree = ast.parse(path.read_text(), filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name != 'main':
            body.append(node)
        elif isinstance(node, ast.Import):
            kept = [alias for alias in node.names if alias.name != 'torch_tensorrt']
            if kept:
                node.names = kept
                body.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module != 'torch_tensorrt':
            body.append(node)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec', flags=__future__.annotations.compiler_flag), module.__dict__)
    except Exception:
        del sys.modules[name]
        raise
    return module


class BFMAgent:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        source = Path(config.export_source)
        upstream = _export_definitions(source)
        self.metadata = json.loads(Path(config.metadata).read_text())
        m = self.metadata
        if m.get('schema_version') != 1:
            raise ValueError('Unsupported ScaleBFM metadata schema')
        a = m['policy_architecture']
        actor = upstream.HumanoidTransformer(
            a['prop_obs_dim'], a['action_dim'], a['output_dim'],
            a['embedding_dim'], a['num_heads'], a['ff_dim'], a['num_layers'])
        embedder = upstream.TaskEmbedder(a['task_obs_dim'], a['embedding_dim'],
                                        a['reduced_task_dim'], a['task_embedder_hidden_dims'])
        weights = torch.load(config.checkpoint, map_location='cpu', weights_only=True)['model_state_dict']
        actor.load_state_dict({k[6:]: v for k, v in weights.items() if k.startswith('actor.')}, strict=True)
        prefix = 'actor_task_embedder.'
        embedder.load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}, strict=True)
        modes = torch.load(config.mode_table, map_location=device, weights_only=True)
        mappings = upstream.build_mode_mappings(modes, m['mode_feature_dims'], with_time=m['mode_mapping_with_time'])
        if a['task_obs_dim'] != mappings.shape[-1] + modes.shape[-1]:
            raise ValueError('ScaleBFM task observation dimension mismatch')
        bodies, joints, parents, axes, translations, rotations = upstream.parse_xml(config.xml_path, device)
        rotations = rotations / rotations.norm(dim=-1, keepdim=True)
        selected = torch.tensor([bodies.index(n) for n in m['selected_body_names']], device=device)
        joint_map = torch.tensor([m['joint_names'].index(n) for n in joints], device=device)
        self.policy = upstream.HumanoidTransformerPolicyWrapperWithMode(
            actor, embedder, mappings, modes,
            torch.tensor([m['default_dof_pos']], dtype=torch.float32, device=device),
            torch.tensor([m['action_scale']], dtype=torch.float32, device=device),
            m['history_buffer_size'], len(m['future_idx']), translations, rotations,
            parents, axes, selected, joint_map).eval().to(device)
        self.control_mode = torch.tensor([config.control_mode], dtype=torch.long, device=device)
        self._graph = None
        self._inputs = None

    def get_meta_data(self):
        return self.metadata

    @torch.inference_mode()
    def get_action(self, obs):
        args = [obs[k] for k in ('root_quat_buffer', 'base_ang_vel_buffer', 'dof_pos_buffer',
                                'dof_vel_buffer', 'actions_buffer', 'target_body_pos_future_to_robot_base',
                                'target_body_rot_future_to_robot_base')]
        args.extend((self.control_mode, obs['future_time_offsets']))
        if not self.config.get('cuda_graph', False):
            return self.policy(*args)
        if self._graph is None:
            self._inputs = [x.clone() for x in args]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.policy(*self._inputs)
            torch.cuda.current_stream().wait_stream(stream)
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._outputs = self.policy(*self._inputs)
        for dest, source in zip(self._inputs, args):
            dest.copy_(source)
        self._graph.replay()
        return self._outputs
