"""Viewer-only tracking; independent of the PHP depth sensor."""
import mujoco


def configure_camera(camera, model, config):
    config = config or {}
    # Preserve the viewer/XML defaults unless a scene explicitly sets a view.
    for field in ('distance', 'azimuth', 'elevation'):
        if field in config:
            setattr(camera, field, float(config[field]))
    if config.get('follow', False):
        name = str(config.get('body', 'torso_link'))
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body <= 0:
            raise ValueError(f'Camera follow body not found: {name}')
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = body
    else:
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.trackbodyid = -1
        if 'lookat' in config:
            camera.lookat[:] = config['lookat']
