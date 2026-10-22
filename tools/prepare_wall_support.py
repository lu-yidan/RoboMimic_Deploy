"""Bind the continuous rung scene to a fresh observation/action contract."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from policy.climb.contract import ClimbContract

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scene',type=Path,default=ROOT/'outputs/climb_wall_2m_v3')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/wall_support_bundle_v2')
    p.add_argument('--kp-scale',type=float,default=1.)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Choose a fresh bundle directory')
    manifest=json.loads((a.scene/'scene_manifest.json').read_text())
    for rel,h in manifest['files_sha256'].items():
        path=(a.scene/rel).resolve()
        assert path.is_relative_to(a.scene.resolve()) and sha(path)==h,rel
    m=mujoco.MjModel.from_xml_path(str(a.scene/'model.xml'))
    initial=np.load(a.scene/'initial_qpos.npy')
    data=mujoco.MjData(m);data.qpos[:]=initial;mujoco.mj_forward(m,data)
    axis=data.geom_xmat[m.geom('rung_05').id].reshape(3,3)[:,2]
    if abs(axis[1])<.999999:raise ValueError('Expected a transverse rung')
    geometry=dict(target_geom='rung_05',target_axis_offset_m=-.2,target_axis_sign=float(np.sign(axis[1])))
    assert (m.nq,m.nv,m.nu,m.neq)==(36,35,29,0)
    config=json.loads((ROOT/'policy/climb/config/climb.json').read_text())
    if not np.isfinite(a.kp_scale) or not 0<a.kp_scale<=2:
        raise ValueError('This commissioning sweep is limited to 0 < kp scale <= 2')
    config['kps']=(np.asarray(config['kps'])*a.kp_scale).tolist()
    config['kds']=(np.asarray(config['kds'])*np.sqrt(a.kp_scale)).tolist()
    joints=np.array([m.joint(n).id for n in config['joint_names']])
    config.update(default_q=initial[m.jnt_qposadr[joints]].tolist(),
                  joint_lower=m.jnt_range[joints,0].tolist(),joint_upper=m.jnt_range[joints,1].tolist(),
                  scene_variant='continuous_rung_wall_v3_initial_support',
                  initial_pose_source_sha256=sha(a.scene/'initial_qpos.npy'),
                  scene_model_sha256=sha(a.scene/'model.xml'),
                  target_semantics='rung_05 axial offset -0.20 m, ideal simulation observation',
                  target_geometry=geometry,target_axis_convention='Positive world Y, independent of capsule mesh orientation',
                  gain_status='Nominal support commissioning; not a learned climbing controller',
                  kp_scale_from_deployment=a.kp_scale,kd_scale_from_deployment=float(np.sqrt(a.kp_scale)))
    contract=ClimbContract(config)
    a.out.mkdir(parents=True)
    shutil.copytree(a.scene/'meshes',a.out/'meshes')
    for name in ('model.xml','initial_qpos.npy','LICENSE'):shutil.copy2(a.scene/name,a.out/name)
    (a.out/'contract.json').write_text(json.dumps(config,indent=2))
    result=dict(source_scene=str(a.scene.resolve()),source_manifest_sha256=sha(a.scene/'scene_manifest.json'),
                importer_sha256=sha(__file__),contract_sha256=contract.fingerprint,
                task_geometry=geometry,
                physics_dt=float(m.opt.timestep),old_policy_compatible=False,
                task='Initial support only, no ascent or stepping',
                output_sha256={str(f.relative_to(a.out)).replace('\\','/'):sha(f) for f in a.out.rglob('*') if f.is_file()})
    (a.out/'manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(bundle=str(a.out),contract_sha256=contract.fingerprint),indent=2))


if __name__=='__main__':main()
