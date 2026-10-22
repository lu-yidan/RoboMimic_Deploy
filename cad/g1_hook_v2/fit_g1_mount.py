"""Space the provisional mount away from the wrist and preserve hook poses with IK."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT/'outputs/g1_hook_v2_mount_v1')
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/g1_hook_v2_mount_v2')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest = json.loads((args.source/'assembly_manifest.json').read_text())
    for rel, digest in manifest['files_sha256'].items():
        assert hashlib.sha256((args.source/rel).read_bytes()).hexdigest() == digest, rel
    old = mujoco.MjModel.from_xml_path(str(args.source/'model.xml'))
    od = mujoco.MjData(old)
    od.qpos[:] = np.load(args.source/'initial_qpos.npy')
    mujoco.mj_forward(old, od)
    targets = {name: (od.xpos[old.body(name).id].copy(), od.xquat[old.body(name).id].copy())
               for name in ('hook', 'right_hook')}
    args.out.mkdir(parents=True)
    shutil.copytree(args.source/'meshes', args.out/'meshes')
    shutil.copy2(args.source/'LICENSE', args.out/'LICENSE')
    tree = ET.parse(args.source/'model.xml')
    for name in targets:
        body = tree.find(f".//body[@name='{name}']")
        pos = np.fromstring(body.get('pos'), sep=' ')
        pos[0] += .012
        body.set('pos', ' '.join(map(str, pos)))
    tree.write(args.out/'model.xml', encoding='utf-8')
    m = mujoco.MjModel.from_xml_path(str(args.out/'model.xml'))
    d = mujoco.MjData(m)
    d.qpos[:] = od.qpos
    results = []
    for side, name in [('left', 'hook'), ('right', 'right_hook')]:
        joints = [j for j in range(m.njnt) if m.joint(j).name.startswith(side+'_')
                  and any(part in m.joint(j).name for part in ('shoulder', 'elbow', 'wrist'))]
        assert len(joints) == 7
        dofs = m.jnt_dofadr[joints]
        qadr = m.jnt_qposadr[joints]
        bid = m.body(name).id
        target_pos, target_quat = targets[name]
        for iteration in range(300):
            mujoco.mj_forward(m, d)
            rot = np.zeros(3)
            mujoco.mju_subQuat(rot, target_quat, d.xquat[bid])
            error = np.r_[target_pos-d.xpos[bid], rot]
            if np.linalg.norm(error) < 1e-8:
                break
            jp = np.zeros((3, m.nv)); jr = np.zeros_like(jp)
            mujoco.mj_jacBody(m, d, jp, jr, bid)
            jac = np.vstack((jp, jr))[:, dofs]
            delta = jac.T @ np.linalg.solve(jac@jac.T + 1e-7*np.eye(6), error)
            d.qpos[qadr] += np.clip(delta, -.04, .04)
            d.qpos[qadr] = np.clip(d.qpos[qadr], m.jnt_range[joints, 0], m.jnt_range[joints, 1])
        assert np.linalg.norm(error) < 1e-6, (side, error)
        results.append(dict(side=side,iterations=iteration,position_error_m=float(np.linalg.norm(error[:3])),
                            orientation_error_rad=float(np.linalg.norm(error[3:])),
                            max_joint_change_rad=float(np.max(abs(d.qpos[qadr]-od.qpos[qadr])))))
    np.save(args.out/'initial_qpos.npy', d.qpos)
    report = dict(source=str(args.source),wrist_parent_x_offset_m=.012,ik=results,
                  static_kinematics_only=True,physical_adapter_verified=False,
                  builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  files_sha256={str(p.relative_to(args.out)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in args.out.rglob('*') if p.is_file()})
    (args.out/'assembly_manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
