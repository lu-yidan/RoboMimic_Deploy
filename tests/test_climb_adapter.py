"""Contract/FSM integration checks without robot SDK, actor weights or physics."""
import unittest
import numpy as np
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.utils import FSMCommand, FSMStateName
from FSM.FSM import FSM
from policy.climb.Climb import Climb


def state_and_output():
    state, output = StateAndCmd(29), PolicyOutput(29)
    state.gravity_ori = np.array([0., 0., -1.])
    state.climb_target_pos_b[:] = [.4, -.2, .3]
    state.climb_target_axis_b[:] = [0., 1., 0.]
    state.climb_target_valid = True
    return state, output


class AdapterTests(unittest.TestCase):
    def test_missing_model_does_not_silently_use_zero_actor(self):
        state, output = state_and_output()
        policy = Climb(state, output)
        policy.enter(); policy.run()
        self.assertIsNotNone(policy.fault)
        self.assertEqual(policy.checkChange(), FSMStateName.PASSIVE)
        np.testing.assert_array_equal(output.kps, np.zeros(29))

    def test_history_reset_and_output_ownership(self):
        state, output = state_and_output()
        supplied = np.full(29, .2, dtype=np.float32)
        policy = Climb(state, output, actor=lambda obs: supplied)
        state.q = policy.contract.default_q.copy()
        policy.enter(); policy.run()
        self.assertIsNone(policy.fault)
        self.assertEqual(policy.last_observation.shape, (100,))
        expected, clipped = policy.contract.decode_action(supplied)
        np.testing.assert_array_equal(output.actions, expected)
        supplied[:] = -.5
        np.testing.assert_array_equal(output.actions, expected)
        np.testing.assert_array_equal(policy.previous_action, clipped)
        policy.exit(); policy.enter()
        np.testing.assert_array_equal(policy.previous_action, np.zeros(29))

    def test_bad_model_output_and_missing_target_return_damping(self):
        for bad in (np.zeros(28), np.full(29, np.nan)):
            state, output = state_and_output()
            policy = Climb(state, output, actor=lambda obs: bad)
            policy.enter(); policy.run()
            self.assertIsNotNone(policy.fault)
            self.assertTrue(np.isfinite(output.actions).all())
            np.testing.assert_array_equal(output.kps, np.zeros(29))
        state, output = state_and_output()
        state.climb_target_valid = False
        policy = Climb(state, output, actor=lambda obs: self.fail("Actor must not run"))
        policy.enter(); policy.run()
        self.assertEqual(policy.checkChange(), FSMStateName.PASSIVE)

    def test_climb_profile_transitions_without_other_model_dependencies(self):
        state, output = state_and_output()
        fsm = FSM(state, output, profile="climb", climb_options={"actor": lambda obs: np.zeros(29, dtype=np.float32)})
        state.q = fsm.climb_policy.contract.default_q.copy()
        state.skill_cmd = FSMCommand.POS_RESET
        fsm.run(); fsm.run()
        self.assertEqual(fsm.cur_policy.name, FSMStateName.FIXEDPOSE)
        state.skill_cmd = FSMCommand.CMD_CLIMB
        fsm.run(); fsm.run()
        self.assertEqual(fsm.cur_policy.name, FSMStateName.SKILL_CLIMB)
        self.assertIsNone(fsm.climb_policy.fault)
        state.skill_cmd = FSMCommand.PASSIVE
        fsm.run(); fsm.run()
        self.assertEqual(fsm.cur_policy.name, FSMStateName.PASSIVE)
        np.testing.assert_array_equal(output.kps, np.zeros(29))


if __name__ == "__main__":
    unittest.main()
