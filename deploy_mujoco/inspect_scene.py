"""Static assembly inspection from the deployment entry, without controller execution."""
from pathlib import Path
from queue import SimpleQueue
import time

import mujoco
import mujoco.viewer
import numpy as np

VIEWS = {
    49: ([.30, 0, .92], 2.4, -40, -12),
    50: ([.365, .20, 1.29], .34, -65, -15),
    51: ([.365, -.20, 1.29], .34, 60, -15),
}


def inspect_scene(cfg, project_root):
    model = mujoco.MjModel.from_xml_path(str(Path(project_root)/cfg.xml_path))
    data = mujoco.MjData(model)
    pose = np.load(Path(project_root)/cfg.initial_qpos_path)
    if pose.shape != (model.nq,) or not np.isfinite(pose).all():
        raise ValueError('Invalid inspection pose')
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    keys = SimpleQueue()
    views = VIEWS
    if cfg.get('inspection_cameras'):
        views = {ord(str(c.key)): (list(c.lookat),float(c.distance),float(c.azimuth),float(c.elevation))
                 for c in cfg.inspection_cameras}
    with mujoco.viewer.launch_passive(model, data, key_callback=keys.put) as viewer:
        def camera(key):
            look, distance, azimuth, elevation = views[key]
            viewer.cam.lookat[:] = look
            viewer.cam.distance = distance
            viewer.cam.azimuth = azimuth
            viewer.cam.elevation = elevation
        with viewer.lock():
            camera(49)
            viewer.opt.geomgroup[3] = 0
            viewer.opt.geomgroup[4] = 1
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        help_text = cfg.get('inspection_help','1 full, 2 left, 3 right')
        print(f'READY: STATIC INSPECTION | {help_text}, A adapter envelope, C collision proxies, R reset. No dynamics or policy.', flush=True)
        while viewer.is_running():
            with viewer.lock():
                while not keys.empty():
                    key = keys.get()
                    if key in views:
                        camera(key)
                    elif key == 65:
                        viewer.opt.geomgroup[4] ^= 1
                    elif key == 67:
                        viewer.opt.geomgroup[3] ^= 1
                    elif key == 82:
                        data.qpos[:] = pose
                        data.qvel[:] = 0
                        mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1/60)
