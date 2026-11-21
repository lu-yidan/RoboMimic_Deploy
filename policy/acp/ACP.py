"""Single-graph ACP controller with a hashed, named-joint training contract."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

from FSM.FSMState import FSMState
from common.utils import FSMCommand, FSMStateName
from policy.acp.observations import ObservationHistory, goal_command, grid_xy, rotation


class ACP(FSMState):
    def __init__(self, state_cmd, policy_output, config, joint_names):
        super().__init__()
        self.name, self.name_str = FSMStateName.SKILL_ACP, 'ACP'
        self.state_cmd, self.policy_output, self.config = state_cmd, policy_output, config
        age = float(config.get('max_age_s', .15))
        skew = float(config.get('max_sync_skew_s', .04))
        tilt = float(config.get('max_tilt_rad', .7))
        clearance = float(config.get('min_clearance', .3))
        entry_error = float(config.get('max_entry_error', .35))
        if (not np.isfinite([age,skew,tilt,clearance,entry_error]).all()
                or not 0 < age <= .5 or not 0 <= skew <= age or not 0 < tilt < np.pi/2
                or not 0 < clearance < 1.5 or not 0 < entry_error < 1.):
            raise ValueError('Invalid ACP timing, tilt, clearance, or entry limits')
        directory = Path(config['model_dir']).expanduser()
        self.manifest = m = json.loads((directory / 'manifest.json').read_text())
        if (m.get('schema_version') != 1 or m.get('stage') != 3 or m.get('planner_obs_dim') != 939
                or m.get('history_length') != 5 or m.get('history_reset') != 'repeat'
                or not np.isclose(m.get('control_dt', 0), .02)):
            raise ValueError('ACP requires a Stage 3, 939-input, 50 Hz bundle with repeat-reset history')
        names = m['joint_names']
        fixed = m['fixed_joints']
        wrists = {side + '_wrist_' + axis + '_joint' for side in ('left', 'right')
                  for axis in ('roll', 'pitch', 'yaw')}
        if (joint_names is None or len(joint_names) != 29 or len(set(joint_names)) != 29
                or len(names) != 23 or len(set(names)) != 23 or set(fixed) != wrists
                or set(names) & wrists or set(names) | wrists != set(joint_names)):
            raise ValueError('ACP needs 23 unique named action joints and six explicitly fixed wrists')
        self.to_model = np.array([joint_names.index(n) for n in names])
        self.default = np.zeros(29, np.float32)
        self.kps, self.kds = self.default.copy(), self.default.copy()
        self.effort_limits = self.default.copy()
        for field, dest in (('default_joint_pos', self.default), ('kps', self.kps), ('kds', self.kds),
                            ('effort_limits', self.effort_limits)):
            value = np.asarray(m[field], dtype=np.float32)
            if value.shape != (23,) or not np.isfinite(value).all():
                raise ValueError('Invalid ACP manifest ' + field)
            dest[self.to_model] = value
        for name, values in fixed.items():
            index = joint_names.index(name)
            self.default[index], self.kps[index], self.kds[index] = (values[k] for k in ('position', 'kp', 'kd'))
            self.effort_limits[index] = values['effort_limit']
        if (not all(np.isfinite(x).all() for x in (self.default, self.kps, self.kds))
                or np.any(self.kps <= 0) or np.any(self.kds < 0)):
            raise ValueError('Invalid ACP fixed pose or PD gains')
        if not np.isfinite(self.effort_limits).all() or np.any(self.effort_limits <= 0):
            raise ValueError('Invalid ACP effort limits')
        self.effort_limits29 = self.effort_limits
        self.scale = np.asarray(m['action_scale'], np.float32)
        if self.scale.shape not in ((), (23,)) or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError('Invalid ACP action scale')
        self.action_clip = m.get('action_clip')
        if self.action_clip is not None and (not np.isfinite(self.action_clip) or self.action_clip <= 0):
            raise ValueError('Invalid ACP action clip')
        self.radius = float(m['target_radius'])
        self.target = np.asarray(config.get('target_position', [1., 0.]), dtype=np.float32)
        self.heading = float(config.get('target_heading', 0.))
        if (self.target.shape != (2,) or not np.isfinite(self.target).all()
                or not np.isfinite(self.heading) or not np.isfinite(self.radius) or self.radius <= 0):
            raise ValueError('Invalid ACP fixed target')
        policy_file = directory / 'policy.onnx'
        if hashlib.sha256(policy_file.read_bytes()).hexdigest() != m['policy_sha256']:
            raise ValueError('ACP policy SHA256 does not match manifest')
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(policy_file), options, providers=['CPUExecutionProvider'])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if ([(x.name, x.shape, x.type) for x in inputs] != [('planner_obs', [1, 939], 'tensor(float)')]
                or {x.name: x.shape for x in outputs} != {'actions': [1,23], 'contacts': [1,9,3], 'occupancy': [1,9,1]}):
            raise ValueError('ACP ONNX interface mismatch')
        self.history = ObservationHistory()
        self.last_action = np.zeros(23, np.float32)
        self.failed = False
        self.last_error = None
        self.last_observation = None
        self.last_raw_action = None

    def error(self):
        s, now = self.state_cmd, time.monotonic()
        try:
            if not s.acp_pose_valid:
                return 'Pelvis localization unavailable'
            age_limit = float(self.config.get('max_age_s', .15))
            if not 0 < age_limit <= .5:
                return 'Invalid ACP input age limit'
            for stamp in (s.acp_pose_time, s.acp_terrain_time):
                if not 0 <= now-stamp <= age_limit:
                    return 'ACP localization or terrain is stale'
            if abs(s.acp_pose_time-s.acp_terrain_time) > float(self.config.get('max_sync_skew_s', .04)):
                return 'ACP localization and terrain timestamps disagree'
            for value, shape in ((s.q,(29,)), (s.dq,(29,)), (s.acp_ang_vel,(3,)),
                                  (s.pelvis_pos_w,(3,)), (s.acp_terrain,(187,3))):
                if np.shape(value) != shape or not np.isfinite(value).all():
                    return 'Invalid ACP robot or terrain sample'
            if not np.allclose(s.acp_terrain[:,:2], grid_xy(), atol=1e-5):
                return 'ACP terrain grid order mismatch'
            gravity_z = rotation(s.pelvis_quat_w)[2,2]
            if gravity_z < np.cos(float(self.config.get('max_tilt_rad', .7))):
                return 'ACP pelvis exceeds upright tilt limit'
            if s.acp_terrain[93,2]+.5 < float(self.config.get('min_clearance', .3)):
                return 'ACP pelvis clearance too low'
        except (AttributeError, TypeError, ValueError, IndexError) as exc:
            return str(exc)
        return None

    def ready(self):
        return self.error() is None

    def entry_error(self):
        problem = self.error()
        if problem:
            return problem
        if np.max(np.abs(self.state_cmd.q-self.default)) > float(self.config.get('max_entry_error', .35)):
            return 'Robot has not reached the ACP default pose'
        return None

    def warmup(self):
        self._infer(np.zeros((1,939), np.float32))

    def enter(self):
        self.history.reset()
        self.last_action.fill(0)
        self.failed, self.last_error = False, None
        self.last_observation = self.last_raw_action = None
        self.policy_output.ghost_qpos = self.policy_output.viz_spheres = None

    def build_observation(self):
        s = self.state_cmd
        proprio = self.history.append(s.acp_ang_vel, s.pelvis_quat_w,
            (s.q-self.default)[self.to_model], s.dq[self.to_model], self.last_action)
        goal = goal_command(s.pelvis_pos_w, s.pelvis_quat_w, self.target, self.heading, self.radius)
        obs = np.concatenate((proprio, s.acp_terrain.ravel(), goal)).astype(np.float32)[None]
        if obs.shape != (1,939) or not np.isfinite(obs).all():
            raise ValueError('Invalid ACP planner observation')
        self.last_observation = obs.copy()
        return obs

    def _infer(self, obs):
        result = self.session.run(['actions','contacts','occupancy'], {'planner_obs': obs})
        if [x.shape for x in result] != [(1,23),(1,9,3),(1,9,1)] or not all(np.isfinite(x).all() for x in result):
            raise ValueError('Invalid ACP inference output')
        if np.any(result[2] < 0) or np.any(result[2] > 1):
            raise ValueError('Invalid ACP occupancy probability')
        return result

    def run(self):
        try:
            problem = self.error()
            if problem:
                raise ValueError(problem)
            actions, self.contacts, self.occupancy = self._infer(self.build_observation())
            self.last_raw_action = actions[0].copy()
            # Check again after inference so a delayed result cannot command the robot.
            problem = self.error()
            if problem:
                raise ValueError(problem)
            self.last_action[:] = actions[0] if self.action_clip is None else np.clip(actions[0], -self.action_clip, self.action_clip)
            targets = self.default.copy()
            targets[self.to_model] += self.scale*self.last_action
            self.policy_output.actions, self.policy_output.kps, self.policy_output.kds = targets, self.kps, self.kds
            self.state_cmd.telemetry.update(acp_goal_distance=float(np.linalg.norm(self.target-self.state_cmd.pelvis_pos_w[:2])))
        except Exception as exc:
            self.failed, self.last_error = True, str(exc)
            print('[ACP] {}; switching to damping'.format(exc))
            self.state_cmd.skill_cmd = FSMCommand.PASSIVE
            self.policy_output.actions = self.state_cmd.q
            self.policy_output.kps, self.policy_output.kds = np.zeros(29), np.full(29,8.)

    def exit(self):
        self.history.reset()
        self.last_action.fill(0)

    def checkChange(self):
        transitions = {FSMCommand.PASSIVE: FSMStateName.PASSIVE, FSMCommand.POS_RESET: FSMStateName.FIXEDPOSE}
        state = transitions.get(self.state_cmd.skill_cmd, self.name)
        if state != self.name:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
        return state
