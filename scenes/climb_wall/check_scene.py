"""Check rung connectivity and discrete placement coverage, then render the scene.

Translated poses are geometry probes, not a motion trajectory or climbing result.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene',type=Path,default=ROOT/'outputs/climb_wall_2m_v3')
    args = parser.parse_args()
    manifest = json.loads((args.scene/'scene_manifest.json').read_text())
    p = manifest['parameters']
    m = mujoco.MjModel.from_xml_path(str(args.scene/'model.xml'));d = mujoco.MjData(m)
    initial = np.load(args.scene/'initial_qpos.npy');d.qpos[:] = initial
    mujoco.mj_forward(m,d)
    assert (m.nq,m.nv,m.nu,m.neq) == (36,35,29,0)
    assert all('ledge' not in m.geom(i).name for i in range(m.ngeom))
    pairs = []
    for row in manifest['rows']:
        name = row['name']
        for side in ('left','right'):
            for a,b in [(name,name+'_'+side+'_arm'),
                        (name+'_'+side+'_arm',side+'_mount_rail'),
                        (name+'_'+side+'_brace',name+'_'+side+'_arm'),
                        (name+'_'+side+'_brace',side+'_mount_rail'),
                        (side+'_mount_rail','wall')]:
                distance = mujoco.mj_geomDistance(m,d,m.geom(a).id,m.geom(b).id,1,None)
                assert distance <= 1e-5,(a,b,distance)
                pairs.append(dict(a=a,b=b,signed_distance_m=distance))
    probes = []
    steps = manifest['goal_foot_row']-manifest['initial_foot_row']
    for index in range(steps+1):
        d.qpos[:] = initial
        d.qpos[:3] += np.array(manifest['nominal_base_translation_m'])*index/steps
        mujoco.mj_forward(m,d)
        contacts = []
        for c in d.contact:
            names = [m.geom(int(g)).name or m.body(m.geom_bodyid[int(g)]).name for g in (c.geom1,c.geom2)]
            expected = any(n in ('left_sole_surface','right_sole_surface') for n in names) and f'rung_{index:02d}' in names
            contacts.append(dict(geoms=names,distance_m=float(c.dist),expected_sole_contact=expected))
            assert expected or c.dist >= -1e-5,(index,names,c.dist)
        assert {n for c in contacts for n in c['geoms'] if n.endswith('_sole_surface')} == {'left_sole_surface','right_sole_surface'}
        assert all(c['distance_m'] >= -.001 for c in contacts)
        probes.append(dict(vertical_offset_m=index*p['row_pitch_m'],contacts=contacts))
    report = dict(scope='Static geometry only. No transition trajectory, dynamics, stability or structural strength validation.',
                  dimensions=dict(nq=m.nq,nv=m.nv,nu=m.nu,neq=m.neq),
                  support_connection_checks=pairs,discrete_pose_probes=probes,
                  replaced_corner_spheres_with_flat_sole=True,
                  initial_sole_compression_m=p['initial_sole_compression_m'])
    (args.scene/'scene_check.json').write_text(json.dumps(report,indent=2))
    d.qpos[:] = initial;mujoco.mj_forward(m,d)
    config = yaml.safe_load((ROOT/'deploy_mujoco/config/mujoco_climb_wall_2m.yaml').read_text())
    opt = mujoco.MjvOption();opt.geomgroup[3]=0;opt.geomgroup[4]=1
    with mujoco.Renderer(m,height=1200,width=1600) as renderer:
        for name,view in zip(('overview','robot','hands','feet','side'),config['inspection_cameras']):
            cam=mujoco.MjvCamera();cam.lookat[:]=view['lookat'];cam.distance=view['distance']
            cam.azimuth=view['azimuth'];cam.elevation=view['elevation']
            renderer.update_scene(d,cam,scene_option=opt)
            np.save(args.scene/(name+'.npy'),renderer.render())
    print(json.dumps(dict(connection_checks=len(pairs),discrete_poses=len(probes),
                         rise_m=manifest['climb_rise_m'],result='PASS: static scene geometry'),indent=2))


if __name__ == '__main__':
    main()
