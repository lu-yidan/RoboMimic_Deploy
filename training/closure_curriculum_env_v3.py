"""Training-only reset using preallocated native buffers, never during evaluation.

All five trusted templates remain resident. Python controller history is copied;
MuJoCo model and all four data buffers are restored through the installed C API.
"""
import copy
import ctypes
import gzip
import pickle
from pathlib import Path
import gymnasium as gym
import mujoco
import numpy as np
from training.closure_curriculum_env import ClosureCurriculumEnv, digest
from training.closure_quality_reward import ClosureQualityReward


def native_objects(root):
    result = {}
    seen = set()
    def walk(value, path):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, (mujoco.MjModel, mujoco.MjData)):
            result[path] = value
        elif isinstance(value, dict):
            for key, child in value.items():
                walk(child, path + '.' + str(key))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, path + '.' + str(index))
        elif type(value).__module__.startswith(('training', 'gymnasium')):
            for key, child in vars(value).items():
                walk(child, path + '.' + key)
    walk(root, 'env')
    return result


class ClosureCurriculumEnvV3(ClosureCurriculumEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffers = None
        self.library = ctypes.CDLL(str(Path(mujoco.__file__).parent / 'mujoco.dll'))
        self.copy_model = self.library.mj_copyModel
        self.copy_model.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.copy_model.restype = ctypes.c_void_p
        self.copy_data = self.library.mj_copyData
        self.copy_data.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.copy_data.restype = ctypes.c_void_p
        for case in self.cases:
            self.ensure_cached(case)

    def ensure_cached(self, case):
        if case in self.caches:
            return
        path = self.cache_path / f'{case}.pkl.gz'
        if digest(path) != self.cache_metadata[case]:
            raise ValueError('Cached state changed')
        with gzip.open(path, 'rb') as stream:
            self.caches[case] = pickle.load(stream)

    def restore_template(self, template):
        sources = native_objects(template)
        if self.buffers is None:
            state = copy.deepcopy(template)
            self.buffers = native_objects(state)
            return state
        if set(sources) != set(self.buffers):
            raise ValueError('Native state graph changed')
        models = [k for k, v in sources.items() if isinstance(v, mujoco.MjModel)]
        if len(models) != 1:
            raise ValueError('Exactly one model expected')
        model = self.buffers[models[0]]
        for key in models:
            source, target = sources[key], self.buffers[key]
            if (source.nq, source.nv, source.nbuffer) != (target.nq, target.nv, target.nbuffer):
                raise ValueError('Incompatible native model sizes')
            assert self.copy_model(target._address, source._address) == target._address
        for key, source in sources.items():
            if isinstance(source, mujoco.MjData):
                target = self.buffers[key]
                assert self.copy_data(target._address, model._address, source._address) == target._address
        return copy.deepcopy(template, {id(value): self.buffers[key] for key, value in sources.items()})

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        names = list(self.cases)
        case = (options or {}).get('case', names[int(self.np_random.integers(len(names)))])
        if case not in self.cases:
            raise ValueError('Undeclared curriculum case')
        if self.env is not None:
            self.env.close()
            self.env = None
        self.ensure_cached(case)
        state = self.restore_template(self.caches[case])
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

