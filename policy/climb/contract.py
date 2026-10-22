"""NumPy-only climb-v1 training/deployment contract; no robot I/O.

Body means pelvis axes. Angles are rad, rates rad/s, targets metres.
This position/PD interface does not provide FT, contact truth or base position.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

JOINT_NAMES = tuple(
    [f'{side}_{name}_joint' for side in ('left', 'right') for name in
     ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
    + ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint']
    + [f'{side}_{name}_joint' for side in ('left', 'right') for name in
       ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw')]
)
OBSERVATION_ORDER = ('gravity_body', 'gyro_body', 'joint_position_relative',
                     'joint_velocity_scaled', 'previous_action', 'target_pos_body',
                     'target_axis_body', 'target_valid')


def _vector(value, size, name):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{name}: numeric vector required') from exc
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'{name}: finite shape ({size},) required')
    if np.any(np.abs(result) > np.finfo(np.float32).max):
        raise ValueError(f'{name}: outside float32 range')
    return result.astype(np.float32)


class ClimbContract:
    """One-frame 100D observation, 29D normalized position-residual action.

    previous_action is the previous clipped normalized actor output, before
    joint-limit clipping; initialize to zero on reset. No hidden observation
    history or world-frame state is appended. Invalid targets encode zeros.
    """
    @classmethod
    def from_json(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def __init__(self, config):
        # Canonical JSON hash includes semantics and provenance, not whitespace.
        canonical = json.dumps(config, sort_keys=True, separators=(',', ':'),
                               ensure_ascii=False, allow_nan=False)
        self.fingerprint = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
        if config['schema_version'] != 'climb-v1' or config['obs_dim'] != 100 or config['action_dim'] != 29:
            raise ValueError('Expected climb-v1 / 100 observations / 29 actions')
        if tuple(config['joint_names']) != JOINT_NAMES:
            raise ValueError('Joint order must match the declared G1 hardware order')
        if tuple(config['observation_order']) != OBSERVATION_ORDER:
            raise ValueError('Observation field order differs from climb-v1')
        self.control_dt = float(config['control_dt'])
        if self.control_dt != .02:
            raise ValueError('climb-v1 requires 50Hz / control_dt=0.02')
        self.joint_names = JOINT_NAMES
        self.obs_dim = 100
        self.action_dim = 29
        for name in ('default_q', 'action_scale', 'kps', 'kds', 'joint_lower', 'joint_upper'):
            value = _vector(config[name], 29, name)
            value.setflags(write=False)
            setattr(self, name, value)
        if np.any(self.action_scale <= 0) or np.any(self.kps < 0) or np.any(self.kds < 0):
            raise ValueError('Positive action scales and nonnegative gains required')
        if np.any(self.joint_lower >= self.joint_upper) or np.any(self.default_q < self.joint_lower) or np.any(self.default_q > self.joint_upper):
            raise ValueError('Joint bounds must contain default_q')
        self.dq_scale = float(config['dq_scale'])
        if not np.isfinite(self.dq_scale) or self.dq_scale <= 0:
            raise ValueError('Positive finite dq_scale required')

    def observation(self, q, dq, gravity_body, gyro_body, previous_action,
                    target_pos_body, target_axis_body, target_valid):
        q = _vector(q, 29, 'q'); dq = _vector(dq, 29, 'dq')
        gravity = _vector(gravity_body, 3, 'gravity_body')
        gyro = _vector(gyro_body, 3, 'gyro_body')
        previous = _vector(previous_action, 29, 'previous_action')
        position = _vector(target_pos_body, 3, 'target_pos_body')
        axis = _vector(target_axis_body, 3, 'target_axis_body')
        if not isinstance(target_valid, (bool, np.bool_)):
            raise ValueError('target_valid must be Boolean')
        if abs(float(np.linalg.norm(gravity)) - 1.) > 1e-4:
            raise ValueError('gravity_body must be a unit gravity direction')
        if np.any(np.abs(previous) > 1.):
            raise ValueError('previous_action must be the clipped normalized output')
        if target_valid:
            if abs(float(np.linalg.norm(axis)) - 1.) > 1e-4:
                raise ValueError('Valid target_axis_body must be a unit vector')
        else:
            position = np.zeros(3, np.float32); axis = np.zeros(3, np.float32)
        # Relative joint position is radians, explicitly NOT divided by action_scale.
        with np.errstate(over='ignore', invalid='ignore'):
            obs = np.concatenate((gravity, gyro, q-self.default_q, dq*self.dq_scale,
                                  previous, position, axis, [float(target_valid)])).astype(np.float32)
        if not np.isfinite(obs).all():
            raise ValueError('Observation arithmetic exceeded finite float32 range')
        return obs

    def decode_action(self, action):
        clipped = np.clip(_vector(action, 29, 'action'), -1., 1.)
        target = np.clip(self.default_q + self.action_scale*clipped,
                         self.joint_lower, self.joint_upper)
        if not np.isfinite(target).all():
            raise ValueError('Action arithmetic exceeded finite float32 range')
        return target.astype(np.float32), clipped.astype(np.float32)
