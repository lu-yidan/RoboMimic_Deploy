"""Acceptance-gate regressions independent of a learned policy or hardware."""
import json
import numpy as np
import pytest

from tools.acp.evaluate_mujoco import BASE_STAND_HEIGHT, final_hold_result, final_pose_metrics


def standing_tail():
    rows = np.zeros((50,9))
    rows[:,0] = np.arange(1,51)*.02
    rows[:,1] = .85
    rows[:,3] = BASE_STAND_HEIGHT
    rows[:,8] = 1.
    return rows


def test_slow_late_arrival_does_not_count_as_one_second_at_goal():
    rows = standing_tail()
    rows[:,1] = np.linspace(.75,.81,50)  # final distance .19, but most of the second outside .2
    rows[:,4] = .06  # every speed sample satisfies the former stop-only gate
    result = final_hold_result(rows,[1.,0.],0.)
    assert np.linalg.norm(rows[-1,1:3]-[1.,0.]) < .2
    assert result['stopped_last_second']
    assert not result['held_all_criteria_last_second']


def test_all_criteria_hold_for_complete_final_second():
    result = final_hold_result(standing_tail(),[1.,0.],0.)
    assert result['stopped_last_second'] and result['held_all_criteria_last_second']


@pytest.mark.parametrize('failure',['short','height','heading','tilted','nonfinite'])
def test_each_standing_gate_must_hold_for_every_frame(failure):
    rows = standing_tail()
    if failure == 'short': rows = rows[1:]
    elif failure == 'height': rows[0,3] = .5  # above fall clearance, below the .78 +/- .1 standing target
    elif failure == 'heading': rows[0,7] = .26
    elif failure == 'tilted': rows[0,8] = np.cos(.71)
    elif failure == 'nonfinite': rows[0,6] = np.nan
    assert not final_hold_result(rows,[1.,0.],0.)['held_all_criteria_last_second']


def test_nonfinite_pose_keeps_a_structured_failure_receipt():
    metrics = final_pose_metrics(np.array([np.nan,0.,.78,1.,np.nan,0.,0.]),[1.,0.],0.)
    assert metrics['distance'] is None and metrics['heading_error'] is None
    assert json.loads(json.dumps(metrics,allow_nan=False))['final_position'] == [None,0.,.78]
