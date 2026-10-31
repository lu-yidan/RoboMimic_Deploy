"""Stress repeated native-state resets and compare each first step to evidence."""
import ctypes
from ctypes import wintypes
import gc
import json
from pathlib import Path
import numpy as np
from training.closure_curriculum_env_v3 import ClosureCurriculumEnvV3
from tools.continue_vector_full_cycle_training import sha


def working_set():
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('peak', 'working', 'peak_paged', 'paged',
                'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile', 'private')]
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    c = Counters()
    c.cb = ctypes.sizeof(c)
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    get_info.restype = wintypes.BOOL
    assert get_info(kernel.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return int(c.working)


def main():
    cache = Path('outputs/closure_prefix_cache_20261024_v4')
    env = ClosureCurriculumEnvV3(cache)
    parent = Path('outputs/closure_margin_control98304_v1')
    expected = {}
    for case in env.cases:
        with np.load(parent / f'after_{case}.npz') as z:
            expected[case] = (z['observations'][4983].copy(), z['actions'][4983].copy(),
                z['qpos'][4984].copy(), z['qvel'][4984].copy())
    records = []
    for cycle in range(10):
        for case in env.cases:
            obs, _ = env.reset(seed=20261024, options={'case': case})
            eo, action, eq, ev = expected[case]
            assert np.array_equal(obs, eo)
            env.step(action)
            physical = env.env.unwrapped.env
            assert float(abs(physical.data.qpos - eq).max()) <= 1e-7
            assert float(abs(physical.data.qvel - ev).max()) <= 1e-5
            del physical
        gc.collect()
        records.append(dict(reset_count=(cycle + 1) * 5, working_set_bytes=working_set()))
        print(json.dumps(records[-1]), flush=True)
    steady = [r['working_set_bytes'] for r in records[2:]]
    growth = max(steady) - min(steady)
    result = dict(passed=bool(growth < 256 * 2**20 and max(steady) < 4500 * 2**20),
        records=records, steady_growth_bytes=growth, reset_count=50,
        physics_first_step_checks_passed=True,
        environment_source_sha256=sha('training/closure_curriculum_env_v3.py'),
        base_environment_source_sha256=sha('training/closure_curriculum_env.py'),
        source_sha256=sha(__file__), scope='Single-worker 50-reset stress check; full four-worker training still required.')
    out = Path('outputs/closure_reset_memory_gate_20261024_v2')
    out.mkdir(exist_ok=False)
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    (out / 'gate_source.py').write_bytes(Path(__file__).read_bytes())
    env.close()
    print(json.dumps(result, indent=2))
    assert result['passed']


if __name__ == '__main__':
    main()
