"""Headless adapter smoke check. Never sends commands or advances physics."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from common.ctrlcomp import StateAndCmd, PolicyOutput
from policy.climb.Climb import Climb


def main():
    state, output = StateAndCmd(29), PolicyOutput(29)
    seen = []
    def zero_test_actor(obs):
        seen.append(obs.copy())
        return np.zeros(29, dtype=np.float32)
    policy = Climb(state, output, actor=zero_test_actor)
    state.q = policy.contract.default_q.copy()
    state.gravity_ori = np.array([0., 0., -1.])
    state.climb_target_pos_b[:] = [.4, -.2, .3]
    state.climb_target_axis_b[:] = [0., 1., 0.]
    state.climb_target_valid = True
    policy.enter()
    for _ in range(5):
        policy.run()
    if policy.fault:
        raise RuntimeError(policy.fault)
    expected, _ = policy.contract.decode_action(np.zeros(29, dtype=np.float32))
    np.testing.assert_array_equal(output.actions, expected)
    assert len(seen) == 5 and all(x.shape == (100,) and x.dtype == np.float32 for x in seen)
    print(json.dumps(dict(passed=True,contract_sha256=policy.contract.fingerprint,
        observation_dim=100,action_dim=29,policy_hz=1/policy.control_dt,
        actor="explicit_zero_test_fixture_not_trained",dynamics_steps=0,
        robot_commands_sent=0,climbing_success_verified=False),indent=2))


if __name__ == "__main__":
    main()
