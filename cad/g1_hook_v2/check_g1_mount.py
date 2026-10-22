"""Audit explicit hook pairs (including parent wrists), then render static views."""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy_mujoco.inspect_scene import VIEWS


def mesh_aabb_gap(m, d, a, b):
    bounds = []
    for geom in (a, b):
        if m.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            return None
        mesh = m.geom_dataid[geom]
        start = m.mesh_vertadr[mesh]
        vertices = m.mesh_vert[start:start+m.mesh_vertnum[mesh]]
        world = vertices@d.geom_xmat[geom].reshape(3,3).T+d.geom_xpos[geom]
        bounds.append((world.min(axis=0),world.max(axis=0)))
    lo1,hi1 = bounds[0]; lo2,hi2 = bounds[1]
    return float(np.linalg.norm(np.maximum(0,np.maximum(lo1-hi2,lo2-hi1))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=Path, default=ROOT/'outputs/g1_hook_v2_mount_v2')
    args = parser.parse_args()
    m = mujoco.MjModel.from_xml_path(str(args.scene/'model.xml'))
    d = mujoco.MjData(m)
    d.qpos[:] = np.load(args.scene/'initial_qpos.npy')
    mujoco.mj_forward(m, d)
    assert (m.nq, m.nv, m.nu, m.neq) == (36, 35, 29, 0)
    rows = []
    for side, body in [('left', 'hook'), ('right', 'right_hook')]:
        bid = m.body(body).id
        hooks = [i for i in range(m.ngeom) if m.geom_bodyid[i] == bid and m.geom_group[i] == 3]
        others = [i for i in range(m.ngeom) if m.geom_bodyid[i] != bid
                  and (m.geom_contype[i] or m.geom_conaffinity[i])]
        pairs = []
        for a in hooks:
            for b in others:
                distance = mujoco.mj_geomDistance(m, d, a, b, .1, None)
                if distance < .1:
                    pairs.append(dict(hook=m.geom(a).name,other=m.geom(b).name or f'geom_{b}',
                                      body=m.body(m.geom_bodyid[b]).name,distance_m=distance,
                                      aabb_gap_lower_bound_m=mesh_aabb_gap(m,d,a,b) if abs(distance)<1e-9 else None))
        pairs.sort(key=lambda r:r['distance_m'])
        rows.append(dict(side=side,nearest=pairs[:8],
                         penetrations=[r for r in pairs if r['distance_m'] < -1e-5],
                         zero_distance_pairs=[r for r in pairs if abs(r['distance_m']) < 1e-9],
                         tail_wrist=[r for r in pairs if '_back_' in r['hook'] and 'wrist_yaw' in r['body']],
                         nearest_bar=next(r for r in pairs if r['other'] in ('bar','right_bar'))))
    report = dict(scope='Initial pose only; convex proxies, not exact CAD clearance or motion validation.',
                  zero_distance_caution='Zero distance is not proof of positive clearance.',
                  dims=[m.nq,m.nv,m.nu,m.neq],sides=rows,
                  active_contacts=[dict(a=m.geom(d.contact[i].geom1).name,
                                        b=m.geom(d.contact[i].geom2).name,distance_m=float(d.contact[i].dist))
                                   for i in range(d.ncon)])
    (args.scene/'clearance_check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(rows,indent=2))
    assert not any(r['penetrations'] for r in rows), 'Resolve overlap before showing final assembly'
    option = mujoco.MjvOption(); option.geomgroup[3] = 0; option.geomgroup[4] = 1
    m.vis.global_.offwidth = 1440; m.vis.global_.offheight = 1080
    with mujoco.Renderer(m,height=1080,width=1440) as renderer:
        for name, values in zip(('full','left','right'),VIEWS.values()):
            look, distance, azimuth, elevation = values
            camera = mujoco.MjvCamera(); camera.lookat[:] = look
            camera.distance = distance; camera.azimuth = azimuth; camera.elevation = elevation
            renderer.update_scene(d,camera=camera,scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            np.save(args.scene/(name+'.npy'),renderer.render())


if __name__ == '__main__':
    main()
