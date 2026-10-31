"""Landing curriculum with a real frozen-prefix rollout on every reset.

This slower reset path avoids serialized physics/controller state restoration.
Only transitions after 70.2 seconds are returned to the learning algorithm.
"""
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from training.strict_full_cycle_residual_env import FullCycleResidualEnv, CONTRACT
from training.bar_geometry_observation import BarGeometryObservation
from training.closure_composite_policy import ClosureCompositePolicy
from training.closure_quality_reward import ClosureQualityReward
from training.closure_waist_target import install_waist_target
from training.closure_body_target import install_body_target
from training.closure_foreaft_reference import install_adaptive_reference


class LivePrefixLandingEnv(gym.Env):
    def __init__(self, parent, *, cases=None, weight=.1, record=False):
        self.parent = Path(parent)
        self.manifest = json.loads((self.parent/'manifest.json').read_text())
        if self.manifest['contract'] != CONTRACT:
            raise ValueError('Strict single-cycle parent required')
        self.cases = dict(self.manifest['evaluation_cases'] if cases is None else cases)
        if not self.cases or any(self.manifest['evaluation_cases'].get(k) != v for k,v in self.cases.items()):
            raise ValueError('Select declared parent cases')
        self.teacher = PPO.load(self.parent/'policy.zip', device='cpu')
        self.prefix = ClosureCompositePolicy(self.teacher, self.manifest)
        self.observation_space = self.teacher.observation_space
        self.action_space = self.teacher.action_space
        self.start, self.weight, self.record = 70.2, float(weight), bool(record)
        self.env = None
        self.prefix_trace = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        keys = list(self.cases)
        case = (options or {}).get('case', keys[int(self.np_random.integers(len(keys)))])
        if case not in self.cases:
            raise ValueError('Undeclared case')
        if self.env is not None:
            self.env.close()
        m = self.manifest
        base = FullCycleResidualEnv(plan=m['reference'], cases={case:self.cases[case]},
            normal_bias_y_deg=m['normal_bias_y_deg'], memory_enabled=m['force_history_memory'])
        observer = BarGeometryObservation(base, mode=m['geometry_mode'])
        self.env = ClosureQualityReward(observer, weight=self.weight)
        obs, _ = self.env.reset(seed=20261022, options={'case':case})
        modifier = m['controller_modifier']
        gain = float(np.clip((float(obs[138])-.625)/.125,0.,1.))
        self.waist_state = install_waist_target(base, modifier['waist_bias'], geometry_gain=gain,
            ramp_seconds=modifier['waist_ramp_seconds'], record=self.record)
        install_body_target(base, gain*modifier['body_drop_m'])
        self.reference_state = install_adaptive_reference(base, modifier, geometry_gain=gain, record=self.record)
        self.reference_state['set_action'](m['adaptive_reference_command'])
        self.prefix_trace = dict(qpos=[base.env.data.qpos.copy()],qvel=[base.env.data.qvel.copy()],time_s=[0.],
                                 observations=[],actions=[],torques=[],base_rewards=[]) if self.record else None
        self.prefix_steps = 0
        while base.env.data.time < self.start-1e-8:
            action = self.prefix.predict(obs,deterministic=True)[0]
            previous = obs.copy()
            obs, _, done, truncated, info = self.env.step(action)
            self.prefix_steps += 1
            if self.record:
                for key,value in dict(qpos=base.env.data.qpos.copy(),qvel=base.env.data.qvel.copy(),
                    time_s=float(base.env.data.time),observations=previous,actions=action.copy(),
                    torques=base.last_torque.copy(),base_rewards=info['base_reward']).items():
                    self.prefix_trace[key].append(value)
            if done or truncated:
                raise RuntimeError(f'Frozen prefix failed before landing window: {info}')
        if abs(base.env.data.time-self.start)>1e-7 or self.prefix_steps != 3510:
            raise RuntimeError('Incorrect live prefix duration')
        self.last_obs = obs.copy()
        return obs, self.annotate(base.info('live_prefix_reset',False))

    def annotate(self, info):
        return dict(info, training_segment='left_foot_landing_and_closure', training_episode_seconds=43.46,
                    reset_method='time-zero physical rollout with frozen prefix', prefix_replay_steps=self.prefix_steps,
                    prefix_replay_seconds=70.2, complete_cycle_training=False)

    def step(self, action):
        if self.env is None:
            raise RuntimeError('Reset required')
        obs,reward,done,truncated,info = self.env.step(action)
        self.last_obs = obs.copy()
        return obs,reward,done,truncated,self.annotate(info)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
