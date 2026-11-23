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


def test_initialization_support_is_released_before_policy_physics(monkeypatch):
    """A deliberately passive controller must fall after the held entry ramp.

    Run real MuJoCo/G1 physics: leaking the root fixture into ACP would mask the
    fall, so this checks the boundary that matters to the acceptance result.
    """
    import tools.acp.evaluate_mujoco as evaluation

    class PassivePolicy:
        def __init__(self,state,output,config,names):
            self.output = output
            self.default = np.zeros(29)
            self.default[[0,6]],self.default[[3,9]],self.default[[4,10]] = -.3,.62,-.33
            self.kps,self.kds,self.effort_limits = np.ones(29)*40,np.ones(29),np.ones(29)*100
            self.failed,self.last_error = False,None
        def entry_error(self):return None
        def enter(self):pass
        def run(self):
            self.output.actions = self.default.copy()
            self.output.kps = np.zeros(29)
            self.output.kds = np.zeros(29)

    monkeypatch.setattr(evaluation,'ACP',PassivePolicy)
    result,trajectory = evaluation.evaluate({},[0.,0.],0.,15.,support_during_fixedpose=True)
    assert result['policy_entered'] and result['failure_phase'] == 'acp'
    assert 2. < result['sim_seconds'] < 17.
    assert result['final_position'][2] < .3
    assert 0. < result['unassisted_acp_seconds'] < 15.
    assert not result['passed']
