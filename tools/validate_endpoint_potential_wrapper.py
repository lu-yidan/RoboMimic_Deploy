"""Paired complete simulation gate for the training-only potential wrapper."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from training.endpoint_potential_wrapper import EndpointPotentialWrapper
from deploy_mujoco.run_trained_full_cycle import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gamma = .9995
    kwargs = dict(plan='outputs/clearance_back20_leg14_plan_v1/plan.npz', cases={'nominal': {}})
    plain = FullCycleResidualEnv(**kwargs)
    wrapped = EndpointPotentialWrapper(FullCycleResidualEnv(**kwargs), gamma=gamma)
    a, _ = plain.reset(seed=42, options={'case': 'nominal'})
    b, _ = wrapped.reset(seed=42, options={'case': 'nominal'})
    assert np.array_equal(a, b)
    rewards, potentials = [], [wrapped.previous_potential]
    steps = 0
    try:
        while True:
            action = np.zeros(26)
            a, ra, ta, xa, ia = plain.step(action)
            b, rb, tb, xb, ib = wrapped.step(action)
            assert np.array_equal(a, b)
            assert np.array_equal(plain.env.data.qpos, wrapped.env.env.data.qpos)
            assert np.array_equal(plain.env.data.qvel, wrapped.env.env.data.qvel)
            assert np.array_equal(plain.last_torque, wrapped.env.last_torque)
            assert plain.env.data.time == wrapped.env.env.data.time
            assert (ta, xa, ia) == (tb, xb, {k: ib[k] for k in ia})
            assert ra == ib['base_reward']
            rewards.append((ra, rb, ib['shaping_reward']))
            potentials.append(ib['reward_potential'])
            steps += 1
            if steps % 1024 == 0:
                # Merely observing a PPO-sized boundary does not reset state.
                assert wrapped.previous_potential == potentials[-1]
                print(json.dumps({'steps': steps, 'time_s': ia['time_s']}), flush=True)
            if ta or xa:
                break
        rewards = np.asarray(rewards)
        difference = float(gamma**np.arange(steps)@(rewards[:, 1]-rewards[:, 0]))
        assert abs(difference) < 1e-10
        assert potentials[-1] == 0.
        assert np.count_nonzero(potentials) > 0
        np.savez_compressed(args.output/'rewards.npz', rewards=rewards, potentials=potentials)
        result = dict(passed=True, steps=steps, terminal=ia, gamma=gamma,
            exact_observations_states_torques_times_and_original_info=True,
            discounted_return_difference=difference, nonzero_potential_samples=int(np.count_nonzero(potentials)),
            sources={p: sha(p) for p in [__file__, 'training/endpoint_potential_wrapper.py',
                'training/endpoint_potential.py', 'training/strict_full_cycle_residual_env.py']})
        (args.output/'summary.json').write_text(json.dumps(result, indent=2))
        for source in result['sources']:
            (args.output/Path(source).name).write_bytes(Path(source).read_bytes())
        print(json.dumps(result), flush=True)
    finally:
        plain.close()
        wrapped.close()


if __name__ == '__main__':
    main()
