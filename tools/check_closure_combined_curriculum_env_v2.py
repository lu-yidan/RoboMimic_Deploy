"""Check repeated suffix resets preserve cached state and physical continuation."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.closure_combined_curriculum_env import ClosureCombinedCurriculumEnv as ClosureCurriculumEnv
from training.closure_quality_reward import quality_bonus
from tools.continue_vector_full_cycle_training import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    parent=args.parent
    manifest=json.loads((parent/'manifest.json').read_text());verified=json.loads((parent/'independent_training_audit.json').read_text())
    assert verified['passed'] and verified['experiment_manifest_sha256']==sha(parent/'manifest.json') and verified['experiment_summary_sha256']==sha(parent/'summary.json')
    expected_success_by_case={r['case']:r['independently_accepted'] for r in verified['trajectories'] if r['label']=='after'}
    assert len(expected_success_by_case)==5 and all(expected_success_by_case.values())
    out=args.output;out.mkdir(exist_ok=False)
    cache = Path('outputs/closure_prefix_cache_20261024_v4')
    protocol = json.loads((cache / 'protocol.json').read_text())
    env = ClosureCurriculumEnv(cache,controller_modifier=manifest['controller_modifier'],record_dir=out/'records')
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
        with np.load(parent / f'after_{case}.npz') as z:
            obs, _ = env.reset(seed=20261024, options={'case': case})
            differences = dict(qpos=0., qvel=0., torques=0., base_reward=0.)
            for step in range(index, len(z['actions'])):
                obs, reward, done, truncated, info = env.step(z['actions'][step])
                base = env.env.unwrapped
                for key, actual, expected in (
                    ('qpos', base.env.data.qpos, z['qpos'][step+1]),
                    ('qvel', base.env.data.qvel, z['qvel'][step+1]),
                    ('torques', base.last_torque, z['torques'][step]),
                    ('base_reward', info['base_reward'], z['rewards'][step])):
                    differences[key] = max(differences[key], float(np.max(np.abs(actual-expected))))
            assert done or truncated
            for key, tolerance in dict(qpos=1e-7,qvel=1e-5,torques=.001,base_reward=1e-6).items():
                assert differences[key] <= tolerance, (case,key,differences)
            expected_success = expected_success_by_case[case]
            assert info['success'] == expected_success
        row = dict(case=case, repeated_reset_exact_and_bounded_first_transition=True,
                   full_suffix_max_differences=differences, success=info['success'])
        rows.append(row)
        print(json.dumps(row), flush=True)
    env.close()
    result = dict(passed=True, rows=rows, prefix_cache_summary_sha256=sha(cache / 'summary.json'),
        source_sha256=sha(__file__),parent_audit_sha256=sha(parent/'independent_training_audit.json'),controller_modifier=manifest['controller_modifier'],environment_source_sha256=sha('training/closure_combined_curriculum_env.py'))
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    (out / 'gate_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
