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
        self.assertEqual(sum(model.mesh_vertnum[model.geom_dataid[i]] for i in ids),len(mesh.vertices))
        model.geom_group[:] = 1
        model.geom_group[ids] = 0
        data=mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        for x,height in [(0,0),(2,0),(5,.38),(10,.38)]:
            distance=mujoco.mj_ray(model,data,np.array([float(x),0.,3.]),np.array([0.,0.,-1.]),
                                  np.array([1,0,0,0,0,0],dtype=np.uint8),True,-1,np.zeros(1,np.int32))
            self.assertAlmostEqual(3-distance,height,places=5)
