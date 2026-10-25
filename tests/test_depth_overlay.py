"""Check actual GL inset placement, orientation, resize and reference-geometry copy."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest


class DepthOverlayTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which('xvfb-run'), 'requires Xvfb')
    def test_live_inset_pixels_and_resize(self):
        script = r'''
import time
import glfw
import mujoco
import numpy as np
from deploy_mujoco.overlay_viewer import OverlayViewer
m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="2 2 .1"/></worldbody></mujoco>')
d=mujoco.MjData(m)
mujoco.mj_forward(m,d)
sensor=mujoco.Renderer(m,height=60,width=106)
with OverlayViewer(m,d) as viewer:
    assert glfw.get_window_attrib(viewer.window,glfw.VISIBLE)
    sensor.update_scene(d)
    sensor.render()
    image=np.full((58,87),.5,dtype=np.float32)
    image[:29]=-.5
    original=viewer._draw_depth
    checked=[]
    def check(width,height):
        original(width,height)
        pixels=np.zeros((height,width,3),np.uint8)
        mujoco.mjr_readPixels(pixels.ravel(),None,mujoco.MjrRect(0,0,width,height),viewer.context)
        # Top of inset is near/yellow, bottom is far/blue; no upside-down image.
        scale=min(3,(width-32)//87,(height-80)//58)
        y=height-58*scale-56
        np.testing.assert_allclose(pixels[height-65,width-30],[255,220,0],atol=1)
        np.testing.assert_allclose(pixels[y+5,width-30],[0,60,200],atol=1)
        checked.append((width,height))
    viewer._draw_depth=check
    viewer.set_depth(image,time.monotonic(),'PASSIVE')
    mujoco.mjv_initGeom(viewer.user_scn.geoms[0],mujoco.mjtGeom.mjGEOM_SPHERE,
                       np.ones(3)*.1,np.array([0.,0.,1.]),np.eye(3).ravel(),np.ones(4,dtype=np.float32))
    viewer.user_scn.ngeom=1
    viewer.sync()
    assert viewer.scene.ngeom >= 2
    glfw.set_window_size(viewer.window,640,480)
    for _ in range(5):
        glfw.poll_events()
        time.sleep(.01)
    viewer.sync()
    assert checked[-1] == (640,480), checked
sensor.close()
'''
        env = dict(os.environ)
        env.pop('MUJOCO_GL',None)
        result=subprocess.run(['xvfb-run','-a',sys.executable,'-c',script],
                              cwd=str(Path(__file__).resolve().parents[1]),env=env,
                              capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
