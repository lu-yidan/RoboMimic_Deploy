"""Headless geometry/config checks; no joystick or hardware connection."""
from pathlib import Path
import ast
import unittest
import mujoco
from hydra import compose, initialize_config_dir
from deploy_mujoco.scene_builder import load_model, apply_initial_state
ROOT = Path(__file__).resolve().parents[1]

class TableConfigTests(unittest.TestCase):
    def test_composed_geometry_and_reset(self):
        with initialize_config_dir(config_dir=str(ROOT/'deploy_mujoco/config'), version_base=None):
            cfg = compose(config_name='mujoco_table')
        self.assertEqual(cfg.simulation_dt * cfg.control_decimation, .02)
        self.assertEqual(cfg.startup_policy, 'passive')
        m = load_model(ROOT/cfg.xml_path, cfg.scene)
        d = mujoco.MjData(m)
        apply_initial_state(m, d, cfg.initial_state)
        top = m.geom('table_top').id
        self.assertAlmostEqual(d.geom_xpos[top,2]+m.geom_size[top,2], .4)
        self.assertAlmostEqual(d.geom_xpos[top,2]-m.geom_size[top,2], .36)
        for i in range(4):
            gid=m.geom(f'table_leg{i}').id
            self.assertAlmostEqual(d.geom_xpos[gid,2]-m.geom_size[gid,2], 0)
            self.assertAlmostEqual(d.geom_xpos[gid,2]+m.geom_size[gid,2], .36)
        self.assertTrue(all(c.dist >= -.0011 for c in d.contact))
        self.assertAlmostEqual(d.qpos[0], -.1)
        self.assertAlmostEqual(d.qpos[1], .1)
        original=d.qpos.copy()
        d.qpos[:3]=[3,3,3]
        mujoco.mj_resetData(m,d)
        apply_initial_state(m,d,cfg.initial_state)
        self.assertTrue((d.qpos==original).all())

    def test_invalid_table(self):
        for opts in [{'height':.02,'thickness':.04}, {'leg_inset':.4}, {'top_size':[.8,.3,.02]}]:
            with self.assertRaises(ValueError):
                load_model(ROOT/'g1_description/g1_liao.xml', {'name':'table','table':opts})

    def test_start_remains_fixedpose(self):
        tree=ast.parse((ROOT/'deploy_mujoco/deploy_mujoco.py').read_text())
        matches=[n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.dump(n.test)==ast.dump(ast.parse('joystick.is_button_released(JoystickButton.START)',mode='eval').body)]
        self.assertEqual(len(matches),1)
        self.assertEqual(ast.dump(ast.Module(body=matches[0].body, type_ignores=[])), ast.dump(ast.parse('state_cmd.skill_cmd = FSMCommand.POS_RESET')))

if __name__=='__main__': unittest.main()
