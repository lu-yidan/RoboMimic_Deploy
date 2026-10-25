"""Convert one actor from a GMR pair to the official ScaleBFM NPZ schema."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def convert(pair, actor, out, gmr_root, scale_root):
    metadata = json.loads((scale_root/'artifacts/scalebfm/humanoid_transformer_m/model_22200_tensorrt_metadata.json').read_text())
    source_model = mujoco.MjModel.from_xml_path(str(gmr_root/'assets/unitree_g1/g1_mocap_29dof.xml'))
    model = mujoco.MjModel.from_xml_path(str(scale_root/'ScaleBridge/scalebridge/data/robot/g1_29dof/g1_29dof.xml'))
    data = mujoco.MjData(model)
    with np.load(pair, allow_pickle=False) as archive:
        source = archive['qpos'][:,archive['actors'].tolist().index(actor)].copy()
        fps = float(archive['fps'])
    time = np.arange(len(source))/fps
    target_time = np.arange(int(np.floor(time[-1]*50))+1)/50
    root = np.column_stack([np.interp(target_time,time,source[:,j]) for j in range(3)])
    rotations = Slerp(time, Rotation.from_quat(source[:,[4,5,6,3]]))(target_time)
    root_quat = rotations.as_quat()[:,[3,0,1,2]]
    source_indices = [source_model.joint(n).qposadr[0] for n in metadata['joint_names']]
    target_indices = [model.joint(n).qposadr[0] for n in metadata['joint_names']]
    joint = np.column_stack([np.interp(target_time,time,source[:,j]) for j in source_indices])
    bodies = [model.body(n).id for n in metadata['body_names']]
    positions, quats = [], []
    for i in range(len(target_time)):
        data.qpos[:3], data.qpos[3:7] = root[i], root_quat[i]
        data.qpos[target_indices] = joint[i]
        mujoco.mj_forward(model,data)
        positions.append(data.xpos[bodies].copy())
        quats.append(data.xquat[bodies].copy())
    positions, quats = np.asarray(positions), np.asarray(quats)
    angular = np.zeros_like(positions)
    for b in range(len(bodies)):
        r = Rotation.from_quat(quats[:,b][:,[1,2,3,0]])
        delta = (r[1:]*r[:-1].inv()).as_rotvec()*50
        angular[0,b],angular[-1,b] = delta[0],delta[-1]
        angular[1:-1,b] = (delta[:-1]+delta[1:])/2
    arrays = dict(joint_pos=joint, joint_vel=np.gradient(joint,.02,axis=0),
                  body_pos_w=positions, body_quat_w=quats,
                  body_lin_vel_w=np.gradient(positions,.02,axis=0), body_ang_vel_w=angular)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as f:
        np.savez_compressed(f, **{k:v.astype(np.float32) for k,v in arrays.items()}, fps=50,
                            joint_names=np.asarray(metadata['joint_names']),body_names=np.asarray(metadata['body_names']))
    report = dict(source=str(pair.resolve()), source_sha256=hashlib.sha256(pair.read_bytes()).hexdigest(),
                  actor=actor,source_fps=fps,output_frames=len(joint),output_seconds=len(joint)/50,
                  resampling='linear root/joints; root quaternion SLERP; no slowing or smoothing',
                  fk_xml=str(scale_root/'ScaleBridge/scalebridge/data/robot/g1_29dof/g1_29dof.xml'),
                  root_xy_range_m=np.ptp(root[:,:2],axis=0).tolist())
    report['retarget'] = json.loads((pair.parent/'metadata.json').read_text())
    out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair',type=Path,required=True)
    p.add_argument('--actor',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gmr-root',type=Path,default=Path('/home/d080/workspace/GMR'))
    p.add_argument('--scale-root',type=Path,default=Path('/home/d080/workspace/ScaleBFM'))
    a=p.parse_args()
    print(json.dumps(convert(a.pair,a.actor,a.out,a.gmr_root,a.scale_root),indent=2))
