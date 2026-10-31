"""Estimate errors must not move rods, alter state, or follow the wrong role."""
from pathlib import Path
import unittest
import numpy as np
from training.compiled_rung_geometry import CompiledPhysicalV2CycleEnv
from training.cycle_relative_bar_observation import CycleRelativeBarObservation
from training.estimated_bar_observation import EstimatedBarObservation
from tools.validate_swing_reference import read

ROOT=Path(__file__).resolve().parents[1]


class EstimatedGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        m=read(ROOT/'outputs/swing_lift_phase_sine_policy_20261029_v2/manifest.json')
        cls.base=CompiledPhysicalV2CycleEnv(plan=ROOT/'outputs/v2_swing_two_reference_20261029_v1/plan.npz',
            cases={'case':{'rung_01':[-.001,0.,.015],'rung_06':[.002,0.,-.002]}},
            normal_bias_y_deg=m['normal_bias_y_deg'],memory_enabled=m['force_history_memory'])
        cls.oracle=CycleRelativeBarObservation(cls.base,mode='oracle')
        cls.base.reset(options={'case':'case'})

    @classmethod
    def tearDownClass(cls):cls.base.close()

    def test_empty_override_is_exact_oracle(self):
        wrapper=EstimatedBarObservation(self.base,estimated_offsets={})
        for t in (0.,113.6601,227.31):
            self.base.env.data.time=t;obs=self.base.observation()
            np.testing.assert_array_equal(wrapper.observation(obs),self.oracle.observation(obs))

    def test_rung06_changes_correct_feature_and_never_geometry(self):
        offsets={'rung_06':np.zeros(3)}
        wrapper=EstimatedBarObservation(self.base,estimated_offsets=offsets)
        offsets['rung_06'][:]=.01 # Caller mutation cannot silently retune the estimator.
        model=self.base.env.model;before=[model.geom_pos.copy(),model.site_pos.copy(),self.base.env.data.qpos.copy()]
        for t,slot in ((0.,3),(113.6601,2)):
            self.base.env.data.time=t;obs=self.base.observation();oracle=self.oracle.observation(obs)
            estimate=wrapper.observation(obs);expected=oracle.copy();expected[136+slot*3:139+slot*3]=0.
            np.testing.assert_array_equal(estimate,expected)
            self.assertGreater(float(abs(oracle[136+slot*3:139+slot*3]).max()),.09)
            np.testing.assert_array_equal(estimate[:136],oracle[:136])
        for current,old in zip((model.geom_pos,model.site_pos,self.base.env.data.qpos),before):np.testing.assert_array_equal(current,old)

    def test_invalid_estimates_rejected(self):
        for mapping in ({'rung_99':[0,0,0]},{'rung_06':[0,0]},{'rung_06':[np.nan,0,0]},
                        {'rung_06':[.02001,0,0]},None):
            with self.assertRaises(ValueError):EstimatedBarObservation(self.base,estimated_offsets=mapping)


if __name__=='__main__':unittest.main()
