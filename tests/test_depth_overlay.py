"""Native sidebar, Tracking and actual inset pixels in an Xvfb window."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest


class DepthOverlayTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which('xvfb-run') and shutil.which('ffmpeg') and shutil.which('xwininfo'), 'requires Xvfb and ffmpeg')
    def test_native_inset_and_camera(self):
        from deploy_mujoco.native_depth_overlay import require_native_overlay
        require_native_overlay()
        script = r'''
import ctypes, os, re, subprocess, tempfile, time
import mujoco, mujoco.viewer
import numpy as np
from PIL import Image
from deploy_mujoco.native_depth_overlay import NativeDepthOverlay
m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="2 2 .1"/><body pos="0 0 .5"><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
d=mujoco.MjData(m)
mujoco.mj_forward(m,d)
# Ensure the offscreen renderer's GLFW hints do not hide the native window.
sensor=mujoco.Renderer(m,height=60,width=106)
sensor.update_scene(d);sensor.render()
with mujoco.viewer.launch_passive(m,d) as viewer:
    overlay=NativeDepthOverlay(viewer)
    assert viewer._sim().ui0_enable and viewer._sim().ui1_enable
    image=np.full((58,87),.5,dtype=np.float32);image[:29]=-.5
    with viewer.lock():
        viewer.cam.type=mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid=1
        viewer.cam.azimuth=123
    def update():
        for _ in range(20):
            overlay.set_depth(image,time.monotonic(),'PASSIVE')
            viewer.sync();time.sleep(.025)
    def capture():
        with tempfile.TemporaryDirectory() as directory:
            path=directory+'/screen.png'
            subprocess.run(['ffmpeg','-y','-loglevel','error','-f','x11grab',
                            '-video_size','1280x1024','-i',os.environ['DISPLAY'],
                            '-frames:v','1',path],check=True)
            pixels=np.asarray(Image.open(path))[:,:,:3]
        near=np.argwhere(np.all(pixels==[255,220,0],axis=2))
        far=np.argwhere(np.all(pixels==[0,60,200],axis=2))
        assert len(near)>1000 and len(far)>1000, (len(near),len(far))
        assert near[:,0].max()<far[:,0].min() # sensor image is not upside down
        assert near[:,0].min()<150
        assert near[:,1].min()>250
        return near[:,1].max()
    update()
    assert viewer.cam.type==mujoco.mjtCamera.mjCAMERA_TRACKING
    assert viewer.cam.trackbodyid==1 and viewer.cam.azimuth==123
    first_right=capture()
    # Native sidebar toggle changes the viewport; inset must move with it.
    viewer._sim().ui1_enable=False
    # A resize event asks native MuJoCo to recalculate panel/scene rectangles.
    tree=subprocess.check_output(['xwininfo','-root','-tree'],text=True)
    window=int(re.search(r'(0x[0-9a-f]+) "MuJoCo[^"\n]*"',tree).group(1),16)
    x11=ctypes.CDLL('libX11.so.6')
    x11.XOpenDisplay.restype=ctypes.c_void_p
    x11.XResizeWindow.argtypes=[ctypes.c_void_p,ctypes.c_ulong,ctypes.c_uint,ctypes.c_uint]
    x11.XFlush.argtypes=[ctypes.c_void_p]
    x11.XCloseDisplay.argtypes=[ctypes.c_void_p]
    display=x11.XOpenDisplay(None)
    x11.XResizeWindow(display,window,1000,750);x11.XFlush(display)
    update()
    x11.XCloseDisplay(display)
    assert capture()>first_right
    with viewer.lock():
        viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    update()
    assert viewer.cam.type==mujoco.mjtCamera.mjCAMERA_FREE
    overlay.clear()
sensor.close()
time.sleep(.3)
'''
        env = dict(os.environ)
        env.pop('MUJOCO_GL', None)
        result = subprocess.run(['xvfb-run', '-a', '-s', '-screen 0 1280x1024x24',
                                 sys.executable, '-c', script],
                                cwd=str(Path(__file__).resolve().parents[1]), env=env,
                                capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_debug_connectors_use_supported_mujoco_api(self):
        import mujoco
        import numpy as np
        from deploy_mujoco.deploy_mujoco import _draw_viz_spheres
        model = mujoco.MjModel.from_xml_string('<mujoco/>')
        scene = mujoco.MjvScene(model, maxgeom=10)
        _draw_viz_spheres(scene, [
            {'geom': 'line', 'from': [0, 0, 0], 'to': [0, 0, 2]},
            {'geom': 'arrow', 'from': [0, 0, 0], 'to': [2, 0, 0]},
            {'pos': [1, 2, 3], 'radius': .1},
        ])
        self.assertEqual(scene.ngeom, 3)
        self.assertEqual(scene.geoms[0].type, mujoco.mjtGeom.mjGEOM_CAPSULE)
        self.assertEqual(scene.geoms[1].type, mujoco.mjtGeom.mjGEOM_ARROW)
        np.testing.assert_allclose(scene.geoms[0].pos, [0, 0, 1])
        np.testing.assert_allclose(scene.geoms[2].pos, [1, 2, 3])
