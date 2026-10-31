"""Training-only suffix resets from audited continuous parent prefixes.

Each reset creates a distinct deep copy of a trusted local cached environment.
No ongoing episode state is overwritten, and complete-cycle evaluation must
use the ordinary environment from time zero with the declared prefix policy.
"""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import gymnasium as gym
import numpy as np
from training.closure_quality_reward import ClosureQualityReward


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ClosureCurriculumEnv(gym.Env):
    def __init__(self, cache_path, *, weight=.1, record_dir=None):
        self.cache_path = Path(cache_path)
        summary = json.loads((self.cache_path / 'summary.json').read_text())
        protocol = json.loads((self.cache_path / 'protocol.json').read_text())
        if not summary['passed'] or digest(self.cache_path / 'protocol.json') != summary['protocol_sha256']:
            raise ValueError('Audited prefix cache required')
        for source, expected in protocol['sources'].items():
            if source.replace('\\', '/').startswith('training/') and digest(source) != expected:
                raise ValueError('Prefix dynamics source changed: ' + source)
        self.cases = protocol['cases']
        self.start = protocol['start_seconds']
        self.horizon = protocol['horizon_seconds']
        self.caches = {}
        self.cache_metadata = {}
        self.weight = weight
        self.record_dir = Path(record_dir) if record_dir is not None else None
        if self.record_dir is not None:
            self.record_dir.mkdir(parents=True, exist_ok=False)
        self.record_number = 0
        self.trace = None
        for row in summary['rows']:
            if not row['passed']:
                raise ValueError('Failed cache verification')
            self.cache_metadata[row['case']] = row['cache_sha256']
        if set(self.cache_metadata) != set(self.cases):
            raise ValueError('Incomplete case caches')
        self.ensure_cached(next(iter(self.cases)))
        first = next(iter(self.caches.values()))
        self.observation_space = first.observation_space
        self.action_space = first.action_space
        self.env = None

    def ensure_cached(self, case):
        if case in self.caches:
            return
        # Retain only one immutable template per worker; MuJoCo models are large.
        for env in self.caches.values():
            env.close()
        self.caches.clear()
        path = self.cache_path / f'{case}.pkl.gz'
        if digest(path) != self.cache_metadata[case]:
            raise ValueError('Cached state changed')
        # Only locally generated, hash-verified caches are supported.
        with gzip.open(path, 'rb') as stream:
            self.caches[case] = pickle.load(stream)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        names = list(self.cases)
        case = (options or {}).get('case', names[int(self.np_random.integers(len(names)))])
        if case not in self.cases:
            raise ValueError('Undeclared curriculum case')
        if self.env is not None:
            self.env.close()
            self.env = None
        self.ensure_cached(case)
        state = copy.deepcopy(self.caches[case])
        base = state.unwrapped
        if abs(base.env.data.time - self.start) > 1e-7 or base.finished:
            raise ValueError('Invalid live suffix start')
        if base.contact.d is not base.env.data or base.contact.m is not base.env.model:
            raise ValueError('Controller aliases did not survive copy')
        self.env = ClosureQualityReward(state, weight=self.weight)
        obs = state.observation(base.observation())
        self.last_obs = obs.copy()
        self.trace = None
        if self.record_dir is not None and case == 'foot_z_plus15':
            self.trace = dict(qpos=[base.env.data.qpos.copy()], qvel=[base.env.data.qvel.copy()],
                time_s=[float(base.env.data.time)], observations=[], actions=[], torques=[],
                rewards=[], base_rewards=[])
        return obs, self.annotate(base.info('suffix_reset', False))

    def annotate(self, info):
        return dict(info, training_segment='closure_suffix_only',
            episode_start_time_s=self.start, complete_cycle_training=False,
            scope='Training suffix from a verified parent prefix; not a from-zero policy rollout.')

    def step(self, action):
        if self.env is None:
            raise RuntimeError('Reset required')
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.trace is not None:
            base = self.env.unwrapped
            self.trace['observations'].append(self.last_obs.copy())
            self.trace['actions'].append(np.clip(np.asarray(action), -1., 1.).copy())
            self.trace['qpos'].append(base.env.data.qpos.copy())
            self.trace['qvel'].append(base.env.data.qvel.copy())
            self.trace['time_s'].append(float(base.env.data.time))
            self.trace['torques'].append(base.last_torque.copy())
            self.trace['rewards'].append(reward)
            self.trace['base_rewards'].append(info['base_reward'])
            if terminated or truncated:
                self.record_number += 1
                name = f'highrod_suffix_{self.record_number:04d}'
                path = self.record_dir / (name + '.npz')
                np.savez_compressed(path, **self.trace)
                metadata = self.annotate(info)
                metadata.update(trajectory_sha256=digest(path),
                    cache_sha256=self.cache_metadata[info['case']],
                    recording='All completed highest-case training suffixes, including failures; sampled actions, not deterministic policy evaluation.')
                (self.record_dir / (name + '.json')).write_text(json.dumps(metadata, indent=2))
        self.last_obs = obs.copy()
        return obs, reward, terminated, truncated, self.annotate(info)

    def close(self):
        if self.env is not None:
            self.env.close()
        for env in self.caches.values():
            env.close()
