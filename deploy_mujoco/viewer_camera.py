"""Viewer-only tracking; independent of the PHP depth sensor."""
import mujoco


def configure_camera(camera, model, config):
    config = config or {}
    camera.distance = float(config.get('distance', 3.2))
    camera.azimuth = float(config.get('azimuth', 45))
    camera.elevation = float(config.get('elevation', -25))
    if config.get('follow', True):
        name = str(config.get('body', 'torso_link'))
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body <= 0:
            raise ValueError(f'Camera follow body not found: {name}')
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = body
    else:
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.trackbodyid = -1
        camera.lookat[:] = config.get('lookat', [0, 0, .8])
