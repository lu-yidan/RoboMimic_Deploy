"""ScaleBridge policy and observations on the deployment FSM's live state.

No simulator is created here. Simulation or real DDS supplies the live state.
"""
from pathlib import Path
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from FSM.FSMState import FSMState
from common.utils import FSMStateName, FSMCommand


def validate_motion(path, metadata):
    """Require official 50 Hz, IsaacLab-order robot reference arrays."""
    with np.load(path, allow_pickle=False) as data:
        if float(data['fps']) != 50:
            raise ValueError('ScaleBFM motion must be resampled to 50 Hz, not just relabeled.')
        n = len(data['joint_pos'])
        if n < 2:
            raise ValueError('Motion requires at least two frames')
        b, j = len(metadata['body_names']), len(metadata['joint_names'])
        shapes = {'joint_pos': (n,j), 'joint_vel': (n,j),
                  'body_pos_w': (n,b,3), 'body_quat_w': (n,b,4),
                  'body_lin_vel_w': (n,b,3), 'body_ang_vel_w': (n,b,3)}
        for key, shape in shapes.items():
            if data[key].shape != shape or not np.isfinite(data[key]).all():
                raise ValueError(f'{key}: expected finite {shape}, got {data[key].shape}')
        if not np.allclose(np.linalg.norm(data['body_quat_w'], axis=-1), 1, atol=2e-3):
            raise ValueError('body_quat_w must contain normalized wxyz quaternions')
        # Names are optional in upstream files; when supplied, validate exact ordering.
        for key in ('joint_names', 'body_names'):
            if key in data and data[key].tolist() != metadata[key]:
                raise ValueError(f'{key} does not match checkpoint order')
    return n


class ScaleBFM(FSMState):
    def __init__(self, state_cmd, policy_output, config, joint_names):
        super().__init__()
        self.name = FSMStateName.SKILL_SCALEBFM
        self.name_str = 'ScaleBFM'
        self.state_cmd, self.policy_output = state_cmd, policy_output
        root = Path(config.root).expanduser().resolve()
        bridge = root / 'ScaleBridge'
        if not bridge.is_dir():
            raise FileNotFoundError(f'ScaleBFM checkout missing: {bridge}; set scalebfm.root in config/deploy.yaml')
        sys.path[:0] = [str(bridge), str(bridge / 'scalebridge')]
        from policy.scalebfm.torch_agent import BFMAgent
        from scalebridge.env.motion_tracking import MotionTrackingEnv
        from scalebridge.utils.torch_utils import (
            calc_heading_quat, calc_heading_quat_inv, quat_mul, quat_apply)

        artifact = root / 'artifacts/scalebfm/humanoid_transformer_m'
        agent_cfg = OmegaConf.create({
            'checkpoint': str(Path(config.checkpoint).expanduser().resolve()) if config.checkpoint else str(artifact / 'model_22200.pt'),
            'metadata': str(artifact / 'model_22200_tensorrt_metadata.json'),
            'mode_table': str(artifact / 'mode_table.pt'),
            'xml_path': str(root / 'ScaleTrack/source/scaletrack/scaletrack/assets/robots/g1_29dof/g1_29dof.xml'),
            'export_source': str(root / 'ScaleTrack/scripts/pretrain/rsl_rl/play_export_check_humanoid_transformer_onboard.py'),
            'backend': 'torch', 'control_mode': int(config.get('control_mode', 7)),
            'cuda_graph': bool(config.get('cuda_graph', False)),
        })
        self.agent = BFMAgent(agent_cfg, 'cuda')
        metadata = self.agent.get_meta_data()
        self.motion_path = Path(config.motion).expanduser().resolve() if config.motion else bridge / 'scalebridge/data/motion/g1_29dof/squat.npz'
        self.motion_len = validate_motion(self.motion_path, metadata)
        if len(joint_names) != 29 or set(joint_names) != set(metadata['joint_names']):
            raise ValueError('ScaleBFM requires the checkpoint\'s 29 G1 joints')
        read_idx = [joint_names.index(n) for n in metadata['joint_names']]
        self.write_idx = [joint_names.index(n) for n in metadata.get('target_joint_names', metadata['action_names'])]
        self.kps = np.zeros(29, np.float32)
        self.kds = np.zeros(29, np.float32)
        self.kps[self.write_idx] = metadata['stiffness']
        self.kds[self.write_idx] = metadata['damping']
        state = self.state_cmd

        class LiveState:
            high_dt = .02

            def refresh_sim(self):
                values = {'root_pos': state.pelvis_pos_w,
                          'root_quat_wxyz': state.pelvis_quat_w,
                          'base_ang_vel': state.root_ang_vel_b,
                          'dof_pos': state.q[read_idx], 'dof_vel': state.dq[read_idx]}
                return {k: torch.as_tensor(v, dtype=torch.float32, device='cuda').unsqueeze(0)
                        for k, v in values.items()}

        class LiveReference(MotionTrackingEnv):
            def _setup_simulator(self):
                self.simulator = LiveState()
                self.dt = .02

            def _calibrate(self):
                # Align fresh source data on every entry, without accumulating transforms.
                self.body_pos_w = self.source_pos.clone()
                self.body_quat_w = self.source_quat.clone()
                current = torch.as_tensor(state.pelvis_quat_w, device=self.device)
                align = quat_mul(calc_heading_quat(current), calc_heading_quat_inv(self.source_quat[0,0]))
                expand = align[None,None,:].expand_as(self.body_quat_w)
                offset = self.source_pos[0,0].clone()
                offset[2] = 0
                origin = torch.as_tensor(state.pelvis_pos_w, device=self.device).clone()
                origin[2] = 0
                self.body_pos_w = quat_apply(expand, self.source_pos - offset) + origin
                self.body_quat_w = quat_mul(expand, self.source_quat)

            def reset(self):
                super().reset()
                self._gather_reference_state()
                return self._update_observation_manager()

        env_cfg = OmegaConf.create({
            'motion_path': str(self.motion_path), 'future_idx': [0,1,2,3,4,5],
            'reference_forcing': bool(config.reference_forcing),
            'observation': OmegaConf.load(bridge / 'scalebridge/config/observation/bfm.yaml'),
        })
        self._env_type = LiveReference
        self._env_config = env_cfg
        self.env = LiveReference(env_cfg, metadata, 'cuda')
        self.env.source_pos = self.env.body_pos_w.clone()
        self.env.source_quat = self.env.body_quat_w.clone()
        self.motion_cache = {}
        self.require_cached_motion = False
        self.first = True
        self.failed = False
        self._progress_open = False
        print(f'[ScaleBFM] reference: {self.motion_path} ({self.motion_len / 50:.2f}s); end: hold final reference')
        # Warm up before the interactive simulation begins; enter() resets this history.
        with torch.inference_mode():
            obs = self.env.reset()
            for _ in range(20):
                self.agent.get_action(obs)
        torch.cuda.synchronize()

    @torch.inference_mode()
    def load_motion(self, path):
        path = Path(path).expanduser().resolve()
        if path in self.motion_cache:
            self._close_progress()
            self.env, self.motion_path, self.motion_len = self.motion_cache[path], path, len(self.motion_cache[path].source_pos)
            return
        if self.require_cached_motion:
            raise ValueError(f'Motion not preloaded: {path}')
        length = validate_motion(path, self.agent.get_meta_data())
        config = OmegaConf.create(OmegaConf.to_container(self._env_config, resolve=True))
        config.motion_path = str(path)
        # Build before replacing: invalid files keep the active reference intact.
        env = self._env_type(config, self.agent.get_meta_data(), 'cuda')
        env.source_pos = env.body_pos_w.clone()
        env.source_quat = env.body_quat_w.clone()
        self._close_progress()
        self.env, self.motion_path, self.motion_len = env, path, length
        print(f'[ScaleBFM] Loaded {path.name}: {length} frames; checkpoint reused')

    def preload_motions(self, paths):
        """Only call before starting the live control loop / DDS publisher."""
        initial = self.motion_path
        self.motion_cache[initial] = self.env
        for path in paths:
            self.load_motion(path)
            self.motion_cache[self.motion_path] = self.env
        with torch.inference_mode():
            for env in self.motion_cache.values():
                obs = env.reset()
                self.agent.get_action(obs)
            torch.cuda.synchronize()
        self.load_motion(initial)
        self.require_cached_motion = True

    @torch.inference_mode()
    def enter(self):
        self.obs = self.env.reset()
        self.first, self.failed = True, False
        self.policy_output.ghost_qpos = None
        print(f'[ScaleBFM] Started {self.motion_path.name}; START: FixedPose; SELECT/F1: Passive')
        self.progress_frame = 0
        self._progress_last = -1
        self._show_progress()

    def _show_progress(self):
        frame = min(self.progress_frame, self.motion_len)
        interval = 5 if sys.stdout.isatty() else 50
        if frame == self._progress_last:
            return
        if frame not in (0, self.motion_len) and frame - self._progress_last < interval:
            return
        ratio = frame / self.motion_len
        filled = int(24 * ratio)
        bar = '=' * filled + '-' * (24 - filled)
        status = '保持末帧' if frame == self.motion_len else '跟踪中'
        line = (f'[ScaleBFM] [{bar}] {ratio:6.1%} | '
                f'{frame}/{self.motion_len} 帧 | '
                f'{frame * self.control_dt:.2f}/{self.motion_len * self.control_dt:.2f}s | {status}')
        tty = sys.stdout.isatty()
        print(('\r' if tty else '') + line, end='' if tty else '\n', flush=True)
        self._progress_last = frame
        self._progress_open = tty

    def _close_progress(self):
        if self._progress_open:
            print(flush=True)
            self._progress_open = False

    def run(self):
        try:
            with torch.inference_mode():
                if not self.first:
                    self.env.episode_length_buf += 1
                    self.obs = self.env._compute_observation()
                self.first = False
                target, action = self.agent.get_action(self.obs)
                if not (torch.isfinite(target).all() and torch.isfinite(action).all()):
                    raise ValueError('nonfinite ScaleBFM output')
                self.env.action.copy_(action)
                output = np.zeros(29, np.float32)
                output[self.write_idx] = target.detach().cpu().numpy().reshape(-1)
                self.policy_output.actions = output
                self.policy_output.kps, self.policy_output.kds = self.kps, self.kds
                self.policy_output.viz_spheres = [
                    {'pos': p, 'radius': .025, 'rgba': [0.2, 0.9, 0.4, .7]}
                    for p in self.obs['body_pos_w_future'][0,0].cpu().numpy()]
                self.progress_frame += 1
                self._show_progress()
        except Exception as exc:
            self._close_progress()
            print(f'[ScaleBFM] Inference failed, switching to Passive: {exc}')
            self.failed = True
            self.policy_output.actions = self.state_cmd.q
            self.policy_output.kps = np.zeros(29)
            self.policy_output.kds = np.full(29, 2.)

    def exit(self):
        self._close_progress()
        self.policy_output.ghost_qpos = None
        self.policy_output.viz_spheres = None

    def checkChange(self):
        cmd = self.state_cmd.skill_cmd
        self.state_cmd.skill_cmd = FSMCommand.INVALID
        if self.failed or cmd == FSMCommand.PASSIVE:
            return FSMStateName.PASSIVE
        if cmd == FSMCommand.LOCO:
            return FSMStateName.LOCOMODE
        if cmd == FSMCommand.CMD_AMP:
            return FSMStateName.SKILL_AMP
        if cmd == FSMCommand.POS_RESET:
            return FSMStateName.FIXEDPOSE
        return self.name
