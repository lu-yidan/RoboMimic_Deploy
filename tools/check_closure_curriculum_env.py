"""Check repeated suffix resets preserve cached state and physical continuation."""
import json
from pathlib import Path
import numpy as np
from training.closure_curriculum_env import ClosureCurriculumEnv
from training.closure_quality_reward import quality_bonus
from tools.continue_vector_full_cycle_training import sha


def main():
    cache = Path('outputs/closure_prefix_cache_20261024_v4')
    protocol = json.loads((cache / 'protocol.json').read_text())
    parent = Path(protocol['parent'])
    env = ClosureCurriculumEnv(cache)
    index = round(protocol['start_seconds'] / .02)
    rows = []
    for case in protocol['cases']:
        with np.load(parent / f'after_{case}.npz') as z:
            for repeat in range(2):
                obs, info = env.reset(seed=20261024, options={'case': case})
                assert np.array_equal(obs, z['observations'][index])
                assert info['training_segment'] == 'closure_suffix_only' and not info['complete_cycle_training']
                physical = env.env.unwrapped.env
                assert np.array_equal(physical.data.qpos, z['qpos'][index])
                assert np.array_equal(physical.data.qvel, z['qvel'][index])
                obs, reward, done, truncated, info = env.step(z['actions'][index])
                assert float(abs(obs - z['observations'][index + 1]).max()) <= 1e-5
                assert float(abs(physical.data.qpos - z['qpos'][index + 1]).max()) <= 1e-7
                assert float(abs(physical.data.qvel - z['qvel'][index + 1]).max()) <= 1e-5
                assert abs(info['base_reward'] - float(z['rewards'][index])) <= 1e-7
                assert reward == info['base_reward'] + quality_bonus(info['current_closure_rad'],
                    info['current_rise_m'], info['closure_cost_ramp'])
                if repeat == 0:
                    first = (obs.copy(), physical.data.qpos.copy(), physical.data.qvel.copy(), reward)
                else:
                    assert np.array_equal(first[0], obs) and np.array_equal(first[1], physical.data.qpos)
                    assert np.array_equal(first[2], physical.data.qvel) and first[3] == reward
                assert not done and not truncated
                assert not physical.data.xfrc_applied.any() and not physical.data.qfrc_applied.any()
                assert abs(env.caches[case].unwrapped.env.data.time - protocol['start_seconds']) < 1e-7
        rows.append(dict(case=case, repeated_reset_exact_and_bounded_first_transition=True))
    env.close()
    out = Path('outputs/closure_curriculum_env_gate_20261024_v1')
    out.mkdir(exist_ok=False)
    result = dict(passed=True, rows=rows, prefix_cache_summary_sha256=sha(cache / 'summary.json'),
        source_sha256=sha(__file__), environment_source_sha256=sha('training/closure_curriculum_env.py'))
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    (out / 'gate_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
