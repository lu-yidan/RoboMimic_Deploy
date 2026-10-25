"""PHP D435i student in the common FSM, using the current Holosoma wire contract.

Actor terms are ALPHABETICALLY ordered during training; the ONNX observation_names
metadata is declaration order and must NOT be used to concatenate observations.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import yaml

from FSM.FSMState import FSMState
from common.utils import FSMCommand, FSMStateName, get_gravity_orientation


def direction_command(velocity, high_speed=False):
    x, y = velocity[:2]
    code = 0
    if np.hypot(x, y) > .2:
        angle = np.degrees(np.arctan2(y, x))
        if abs(angle) > 112.5:
            code = 11
        elif abs(angle) <= 22.5:
            code = 1
        elif angle > 0:
            code = 2 if angle < 67.5 else 3
        else:
            code = 4 if angle > -67.5 else 5
        if high_speed and 1 <= code <= 5:
            code += 5
    result = np.zeros(15, dtype=np.float32)
    result[code] = 1
    return result


class PHPParkour(FSMState):
    def __init__(self, state_cmd, policy_output, config, joint_names):
        super().__init__()
        self.state_cmd, self.policy_output = state_cmd, policy_output
        self.name, self.name_str = FSMStateName.SKILL_PHP_PARKOUR, 'PHP_PARKOUR'
        self.config = config
        with (Path(__file__).parent / 'config/php.yaml').open() as stream:
            cfg = yaml.safe_load(stream)
        if joint_names is None or len(joint_names) != 29 or set(joint_names) != set(cfg['joint_names']):
            raise ValueError('PHP requires the 29 named G1 body joints')
        canonical_to_runtime = [cfg['joint_names'].index(n) for n in joint_names]
        self.default = np.array(cfg['default_joint_pos'], np.float32)[canonical_to_runtime]
        self.kps = np.array(cfg['kps'], np.float32)[canonical_to_runtime]
        self.kds = np.array(cfg['kds'], np.float32)[canonical_to_runtime]
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        def session(name):
            path = Path(config['model_dir']).expanduser() / (name + '.onnx')
            if hashlib.sha256(path.read_bytes()).hexdigest() != cfg[name + '_sha256']:
                raise ValueError('PHP checkpoint hash mismatch: {}'.format(path))
            return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        self.backbone, self.student = session('depth_backbone'), session('student')
        meta = self.student.get_modelmeta().custom_metadata_map
        def decode(value):
            try:
                return json.loads(value)
            except ValueError:
                return [x.strip() for x in value.split(',')]
        names = decode(meta['joint_names'])
        if len(names) != 29 or set(names) != set(joint_names):
            raise ValueError('Invalid PHP model joint mapping')
        self.to_model = np.array([joint_names.index(n) for n in names])
        self.to_runtime = np.argsort(self.to_model)
        self.scale = np.asarray(decode(meta['action_scale']), dtype=np.float32)
        if self.scale.shape not in ((), (29,)) or not np.isfinite(self.scale).all():
            raise ValueError('Invalid PHP action scale')
        if self.backbone.get_inputs()[0].shape != [1, 58, 87] or self.student.get_inputs()[0].shape != [1, 140]:
            raise ValueError('PHP requires the released D435i depth/student contract')
        self.last_action = np.zeros(29, np.float32)
        self.failed = False

    def depth_ready(self):
        frame = self.state_cmd.php_depth
        return (frame is not None and frame.shape == (58, 87) and np.isfinite(frame).all()
                and 0 <= time.monotonic() - self.state_cmd.php_depth_time <= .15)

    def enter(self):
        self.last_action.fill(0)
        self.failed = False
        self.state_cmd.php_high_speed = False
        self.policy_output.ghost_qpos = None
        self.policy_output.viz_spheres = None

    def build_observation(self, latent):
        s = self.state_cmd
        return np.concatenate((self.last_action, s.ang_vel,
                               (s.q - self.default)[self.to_model], s.dq[self.to_model],
                               get_gravity_orientation(s.torso_quat_w),
                               direction_command(s.vel_cmd, s.php_high_speed), latent)).astype(np.float32)[None]

    def run(self):
        try:
            if not self.depth_ready():
                raise ValueError('missing, invalid or stale PHP depth frame')
            latent = self.backbone.run(None, {'depth_image': self.state_cmd.php_depth[None]})[0][0]
            obs = self.build_observation(latent)
            if not np.isfinite(obs).all():
                raise ValueError('non-finite PHP observation')
            action = self.student.run(['actions'], {'obs': obs, 'time_step': np.zeros((1, 1), np.float32)})[0][0]
            if action.shape != (29,) or not np.isfinite(action).all():
                raise ValueError('invalid PHP action')
            self.last_action[:] = np.clip(action, -100, 100)
            self.policy_output.actions = self.default + (self.last_action * self.scale)[self.to_runtime]
            self.policy_output.kps = self.kps
            self.policy_output.kds = self.kds
        except Exception as exc:
            print('[PHP] {}; switching to damping'.format(exc))
            self.failed = True
            self.state_cmd.skill_cmd = FSMCommand.PASSIVE
            self.policy_output.actions = self.state_cmd.q
            self.policy_output.kps = np.zeros(29)
            self.policy_output.kds = self.kds

    def exit(self):
        self.last_action.fill(0)
        self.state_cmd.php_high_speed = False

    def checkChange(self):
        targets = {FSMCommand.PASSIVE: FSMStateName.PASSIVE,
                   FSMCommand.POS_RESET: FSMStateName.FIXEDPOSE,
                   FSMCommand.LOCO: FSMStateName.LOCOMODE,
                   FSMCommand.CMD_AMP: FSMStateName.SKILL_AMP}
        target = targets.get(self.state_cmd.skill_cmd, self.name)
        if target != self.name:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
        return target
