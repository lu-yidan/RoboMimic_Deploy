import unittest
from training.closure_quality_reward import quality_bonus


def test_quality_tracks_both_failure_metrics():
    original = quality_bonus(.049244, .260513, 1.)
    assert quality_bonus(.048244, .260513, 1.) > original
    assert quality_bonus(.049244, .260263, 1.) > original
    assert quality_bonus(.050244, .260513, 1.) < original
    assert quality_bonus(.049244, .260763, 1.) < original


def test_bonus_is_nonnegative_bounded_and_inactive_before_closure():
    for closure in (0., .032, .04, .049244, 1.):
        for rise in (0., .24, .25, .26, 1.):
            assert 0 <= quality_bonus(closure, rise, 1.) <= .1
            assert quality_bonus(closure, rise, 0.) == 0
    assert quality_bonus(.031, .25, 1.) == .1


def test_invalid_reward_inputs_fail():
    for args in [(-1., .25, 1.), (.04, float('nan'), 1.), (.04, .25, 1.1)]:
        with unittest.TestCase().assertRaises(ValueError):
            quality_bonus(*args)


if __name__ == "__main__":
    suite = unittest.TestSuite(unittest.FunctionTestCase(f) for f in (test_quality_tracks_both_failure_metrics, test_bonus_is_nonnegative_bounded_and_inactive_before_closure, test_invalid_reward_inputs_fail))
    result = unittest.TextTestRunner().run(suite)
    raise SystemExit(not result.wasSuccessful())
