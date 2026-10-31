"""Regression of scene compilation against the original dynamics and compiler."""
from pathlib import Path
import json
import unittest
import mujoco
import numpy as np
from training.physical_v2_cycle_env import PhysicalV2CycleEnv
from training.compiled_rung_geometry import CompiledPhysicalV2CycleEnv,compiled_variant
from training.cycle_relative_bar_observation import CycleRelativeBarObservation

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'outputs/v2_swing_two_reference_20261029_v1/plan.npz'
GRID=json.loads((ROOT/'outputs/hook_rung06_uncertainty_protocol_20261029_v1/protocol.json').read_text())['cases']


class CompiledGeometryTests(unittest.TestCase):
    def test_nominal_reset_and_short_dynamics_remain_exact(self):
        old=PhysicalV2CycleEnv(plan=PLAN,cases={'nominal':{}})
        new=CompiledPhysicalV2CycleEnv(plan=PLAN,cases={'nominal':{}})
        try:
            a,_=old.reset(seed=7,options={'case':'nominal'})
            b,_=new.reset(seed=7,options={'case':'nominal'})
            np.testing.assert_array_equal(a,b)
            for _ in range(20):
                ra=old.step(np.zeros(26));rb=new.step(np.zeros(26))
                np.testing.assert_array_equal(ra[0],rb[0]);self.assertEqual(ra[1:4],rb[1:4])
                np.testing.assert_array_equal(old.env.data.qpos,new.env.data.qpos)
                np.testing.assert_array_equal(old.env.data.qvel,new.env.data.qvel)
                np.testing.assert_array_equal(old.last_torque,new.last_torque)
        finally:old.close();new.close()

    def test_topology_changing_case_has_correct_oracle_features(self):
        case='hook_grid_x0_z0';offsets=GRID[case]
        old=PhysicalV2CycleEnv(plan=PLAN,cases={case:offsets})
        try:
            with self.assertRaisesRegex(ValueError,'changed BVH topology'):
                old.reset(options={'case':case})
        finally:old.close()
        base=CompiledPhysicalV2CycleEnv(plan=PLAN,cases={case:offsets})
        observer=CycleRelativeBarObservation(base,mode='oracle')
        try:
            observation,_=observer.reset(options={'case':case})
            np.testing.assert_allclose(observation[-3:],np.array(offsets['rung_06'])/.02,rtol=0,atol=1e-7)
            self.assertEqual(base.env.data.time,0)
            self.assertFalse(np.any(base.env.data.qfrc_applied))
            self.assertFalse(np.any(base.env.data.xfrc_applied))
            self.assertGreater(base.variant['model_arrays_checked'],400)
        finally:observer.close()

    def test_repeated_reset_rebuilds_original_model_without_stale_tree(self):
        cases={'a':GRID['hook_grid_x0_z0'],'b':GRID['hook_grid_x3_z3'],'nominal':{}}
        base=CompiledPhysicalV2CycleEnv(plan=PLAN,cases=cases)
        try:
            for name in cases:base.reset(options={'case':name})
            original=mujoco.MjModel.from_xml_path(str(base.env.bundle/'model.xml'))
            for name in dir(original):
                value=getattr(original,name)
                if isinstance(value,np.ndarray):np.testing.assert_array_equal(getattr(base.env.model,name),value,err_msg=name)
            self.assertEqual(base.physical_reset_count,3)
        finally:base.close()

    def test_invalid_displacement_rejected_without_mutation(self):
        base=CompiledPhysicalV2CycleEnv(plan=PLAN,cases={'nominal':{}})
        try:
            initial=base.env.model.geom_pos.copy()
            for offsets in ({'rung_06':[0,.001,0]},{'rung_06':[.021,0,0]},
                            {'rung_06':[float('nan'),0,0]},{'rung_07':[.001,0,0]}):
                with self.assertRaises(ValueError):compiled_variant(base,offsets)
                np.testing.assert_array_equal(base.env.model.geom_pos,initial)
        finally:base.close()


if __name__=='__main__':unittest.main()
