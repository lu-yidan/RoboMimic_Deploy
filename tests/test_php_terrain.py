"""External course geometry stays identical to the source and keeps flat spawn ground."""
import unittest
from pathlib import Path
import mujoco
import numpy as np
from deploy_mujoco.scene_builder import load_model
from policy.php_parkour.terrain import terrain_path

ROOT = Path(__file__).resolve().parents[1]

class PHPTerrainTest(unittest.TestCase):
    def test_external_convex_components_and_surface_heights(self):
        import trimesh
        try:
            source = terrain_path({})
        except FileNotFoundError as exc:
            self.skipTest(str(exc))
        mesh = trimesh.load(str(source), process=False)
        parts = list(mesh.split(only_watertight=False))
        model = load_model(ROOT/'g1_description/scene.xml', {'name':'php_terrain'})
        ids = [i for i in range(model.ngeom) if (model.geom(i).name or '').startswith('php_terrain_')]
        self.assertEqual(len(ids),len(parts))
        self.assertEqual(len(ids),13)
        self.assertNotIn('floor',[model.geom(i).name for i in range(model.ngeom)])
        # All shipped source components are exact cuboids, within OBJ vertex rounding precision.
        data=mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        for geom, part in zip(ids,parts):
            self.assertEqual(model.geom_type[geom],mujoco.mjtGeom.mjGEOM_BOX)
            local=(part.vertices-data.geom_xpos[geom]) @ data.geom_xmat[geom].reshape(3,3)
            np.testing.assert_allclose(np.abs(local),np.broadcast_to(model.geom_size[geom],local.shape),atol=2e-6)
            self.assertAlmostEqual(np.prod(model.geom_size[geom]*2),part.volume,places=5)
        model.geom_group[:] = 1
        model.geom_group[ids] = 0
        data=mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        for x,height in [(0,0),(2,0),(5,.38),(10,.38)]:
            distance=mujoco.mj_ray(model,data,np.array([float(x),0.,3.]),np.array([0.,0.,-1.]),
                                  np.array([1,0,0,0,0,0],dtype=np.uint8),True,-1,np.zeros(1,np.int32))
            self.assertAlmostEqual(3-distance,height,places=5)

    def test_non_box_component_is_not_approximated(self):
        import trimesh
        from policy.php_parkour.terrain import exact_box
        self.assertIsNone(exact_box(trimesh.creation.icosphere(subdivisions=1)))
        skew = trimesh.creation.box()
        skew.vertices[0,0] += .02
        self.assertIsNone(exact_box(skew))

    def test_passive_fall_then_y_remains_stable(self):
        from tools.validate_php_recovery import run_case
        try:
            terrain_path({})
        except FileNotFoundError as exc:
            self.skipTest(str(exc))
        # Original mesh contacts fail around 4.5 s / 11.9 s for these timings.
        for trigger in (1.,10.):
            result=run_case(trigger, duration=20.)
            self.assertIsNone(result['failure'], result)
            self.assertEqual(result['elapsed_s'],20.)
            self.assertGreater(result['final_height'],.6)
