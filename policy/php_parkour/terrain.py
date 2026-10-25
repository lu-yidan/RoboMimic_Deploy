"""Reuse external PHP/Holosoma OBJ components without importing its training stack."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from common.deployment_config import ROOT, load_deployment_config

DEFAULT_OBJ = 'thirdparty/holosoma/src/holosoma/holosoma/data/terrains/terrain.obj'


def terrain_path(config):
    php = load_deployment_config().get('php', {})
    root = Path(php.get('root', '~/workspace/php_parkour')).expanduser()
    if not root.is_absolute():
        root = ROOT / root
    path = Path(config.get('obj_path') or DEFAULT_OBJ).expanduser()
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        raise FileNotFoundError(f'PHP terrain missing: {path}; initialize thirdparty/holosoma in php.root')
    return path


def exact_box(component):
    """Recognize cuboids within the source OBJ's micrometre rounding error."""
    import mujoco
    if len(component.vertices) != 8:
        return None
    import trimesh
    to_origin, extents = trimesh.bounds.oriented_bounds(component)
    local = trimesh.transform_points(component.vertices, to_origin)
    if not np.allclose(np.abs(local), extents / 2, atol=2e-6, rtol=0):
        return None
    pose = np.linalg.inv(to_origin)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(pose[:3,:3]).ravel())
    return extents / 2, pose[:3,3], quat


def add_training_terrain(root, worldbody, config):
    import trimesh
    path = terrain_path(config)
    mesh = trimesh.load(str(path), force='mesh', process=False)
    components = list(mesh.split(only_watertight=False))
    if not components or any(not c.is_watertight or not c.is_convex for c in components):
        raise ValueError('PHP terrain must contain closed convex components; concave assets need decomposition')
    assets = root.find('asset')
    if assets is None:
        assets = ET.SubElement(root, 'asset')
    # The source already has its own ground slab. Remove only the base scene's
    # plane, avoiding coincident visual surfaces and duplicated ground contacts.
    for geom in list(worldbody.findall('geom')):
        if geom.get('type') == 'plane':
            worldbody.remove(geom)
    for i, part in enumerate(components):
        name = f'php_terrain_{i}'
        box = exact_box(part)
        if box is not None:
            size, position, quat = box
            geometry = dict(type='box', size=' '.join(f'{v:.12g}' for v in size),
                            pos=' '.join(f'{v:.12g}' for v in position),
                            quat=' '.join(f'{v:.12g}' for v in quat))
        else:
            ET.SubElement(assets, 'mesh', name=name,
                          vertex=' '.join(f'{v:.9g}' for v in np.asarray(part.vertices).ravel()),
                          face=' '.join(str(int(v)) for v in np.asarray(part.faces).ravel()),
                          smoothnormal='false')
            geometry = dict(type='mesh', mesh=name)
        ET.SubElement(worldbody, 'geom', name=name, **geometry,
                      rgba='0.55 0.57 0.6 1', friction='0.7 0.005 0.001',
                      solimp='0.99 0.99 0.01 0.5 2', solref='0.001 1',
                      contype='2', conaffinity='1')
    print(f'[PHP terrain] {path}; {len(components)} separate collision components')
