"""Bounded, unsupported single-G1 rollout through the deployment FSM, with video."""
from contextlib import nullcontext
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from common.ctrlcomp import StateAndCmd,PolicyOutput
from common.utils import FSMCommand,FSMStateName
from FSM.FSM import FSM
from deploy_mujoco.reset_sim import reset_to_fixed_pose
from deploy_mujoco.deploy_mujoco import pd_control


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--control-mode', type=int, choices=range(8), default=7)
    p.add_argument('--no-video',action='store_true',help='Batch metrics without rendering')
    a=p.parse_args()
    archive_header=dict(np.load(a.motion,allow_pickle=False))
    partial=bool(archive_header.get('partial_reference',False))
    if partial and a.control_mode not in (2,4):
        p.error('Partial pelvis/hands/feet archive requires control-mode 2 or 4')
    import torch
    torch.set_num_threads(1)
    a.out.mkdir(parents=True,exist_ok=False)
    from common.deployment_config import load_scalebfm_config
    cfg=OmegaConf.load(ROOT/'deploy_mujoco/config/mujoco.yaml')
    cfg.scalebfm = load_scalebfm_config(require_enabled=True)
    cfg.scalebfm.motion=str(a.motion.resolve())
    cfg.scalebfm.control_mode=a.control_mode
    model=mujoco.MjModel.from_xml_path(str(ROOT/'g1_description/scene.xml'))
    model.opt.timestep=cfg.simulation_dt
    data=mujoco.MjData(model)
    state,output=StateAndCmd(29),PolicyOutput(29)
    names=[model.joint(int(model.actuator_trnid[i,0])).name for i in range(29)]
    fsm=FSM(state,output,cfg.scalebfm,names)
    tracker=fsm.scalebfm_policy
    reset_to_fixed_pose(model,data,fsm,state,output)
    limits=np.asarray(cfg.tau_limit)
    renderer=None if a.no_video else mujoco.Renderer(model,height=480,width=640)
    cam=mujoco.MjvCamera()
    cam.distance,cam.azimuth,cam.elevation=3.2,135,-15
    trace,errors,heights,tilts,actual_wrists,ref_wrists=[],[],[],[],[],[]
    actual_bodies=[]
    aligned_references=[]
    selected=[model.body(n).id for n in tracker.agent.get_meta_data()['selected_body_names']]
    wrist_indices=[tracker.agent.get_meta_data()['selected_body_names'].index(n) for n in ['left_wrist_yaw_link','right_wrist_yaw_link']]

    def step(command=FSMCommand.INVALID):
        state.q,state.dq=data.qpos[7:36].copy(),data.qvel[6:35].copy()
        state.pelvis_pos_w=data.qpos[:3].astype(np.float32)
        state.pelvis_quat_w=data.qpos[3:7].astype(np.float32)
        state.root_ang_vel_b=data.qvel[3:6].astype(np.float32)
        state.skill_cmd=command
        fsm.run()
        for _ in range(cfg.control_decimation):
            assert np.all(data.xfrc_applied==0)
            data.ctrl[:]=np.clip(pd_control(output.actions,data.qpos[7:36],output.kps,np.zeros(29),data.qvel[6:35],output.kds),-limits,limits)
            mujoco.mj_step(model,data)
        mujoco.mj_forward(model,data)

    foot_ids=[model.body(n).id for n in ('left_ankle_roll_link','right_ankle_roll_link')]
    contact_speeds=[]
    contact_frames=0
    point_jacobian=np.zeros((3,model.nv))
    force=np.zeros(6)
    failure=None
    try:
        for _ in range(10): step()
        step(FSMCommand.CMD_SCALEBFM)
        with (nullcontext(None) if a.no_video else imageio.get_writer(a.out/'tracking.mp4',fps=50,codec='libx264')) as writer:
            for t in range(tracker.motion_len):
                step()
                actual=data.xpos[selected].copy()
                actual_bodies.append(actual)
                aligned_references.append(tracker.obs['body_pos_w_future'][0,0].cpu().numpy().copy())
                reference=tracker.obs['body_pos_w_future'][0,0].cpu().numpy().copy()
                errors.append(np.linalg.norm((actual-actual[0])-(reference-reference[0]),axis=1))
                actual_wrists.append(actual[wrist_indices]-actual[0])
                ref_wrists.append(reference[wrist_indices]-reference[0])
                trace.append(data.qpos.copy())
                heights.append(float(data.qpos[2]))
                tilt=float(np.degrees(np.arccos(np.clip(data.body('pelvis').xmat[8],-1,1))))
                tilts.append(tilt)
                # World velocity at load-bearing foot/floor contact points.
                frame_slip=[]
                for ci in range(data.ncon):
                    contact=data.contact[ci]
                    b1,b2=(int(model.geom_bodyid[g]) for g in (contact.geom1,contact.geom2))
                    foot=b1 if b1 in foot_ids and b2==0 else b2 if b2 in foot_ids and b1==0 else None
                    if foot is None:
                        continue
                    mujoco.mj_contactForce(model,data,ci,force)
                    if abs(force[0]) < 1.:
                        continue
                    mujoco.mj_jac(model,data,point_jacobian,None,contact.pos,foot)
                    point_vel=point_jacobian @ data.qvel
                    frame_slip.append(float(np.linalg.norm(point_vel[:2])))
                if frame_slip:
                    contact_frames+=1
                    contact_speeds.append(max(frame_slip))
                if writer is not None:
                    cam.lookat[:]=data.qpos[:3]
                    renderer.update_scene(data,cam)
                    valid_indices=([0]+wrist_indices if a.control_mode==2 else [i for i,n in enumerate(tracker.agent.get_meta_data()['selected_body_names']) if n in archive_header.get('valid_body_names',[])]) if partial else range(len(reference))
                    for pos in (reference-reference[0]+actual[0])[valid_indices]:
                        scene=renderer.scene
                        mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,.025),pos,np.eye(3).flatten(),np.array([.1,.9,.25,.8]))
                        scene.ngeom+=1
                    frame=Image.fromarray(renderer.render())
                    ImageDraw.Draw(frame).text((8,8),f'{a.motion.stem} | {t/50:.2f}s | PD physics, no support\nGreen: root-aligned reference',fill='white')
                    writer.append_data(np.asarray(frame))
                if tracker.failed or not np.isfinite(data.qpos).all():
                    failure='nonfinite_or_policy_failure'
                    break
                if heights[-1]<.3 or tilt>70:
                    failure='fall_threshold'
                    break
        # Full kinematic reference: deliberately separate from physical rollout.
        if not a.no_video and not partial:
            archive=np.load(a.motion,allow_pickle=False)
            indices=[model.joint(n).qposadr[0] for n in archive['joint_names']]
            with imageio.get_writer(a.out/'reference.mp4',fps=50,codec='libx264') as writer:
                for t in range(len(archive['joint_pos'])):
                    data.qpos[:3]=archive['body_pos_w'][t,0]
                    data.qpos[3:7]=archive['body_quat_w'][t,0]
                    data.qpos[indices]=archive['joint_pos'][t]
                    mujoco.mj_forward(model,data)
                    cam.lookat[:]=data.qpos[:3]
                    renderer.update_scene(data,cam)
                    frame=Image.fromarray(renderer.render())
                    ImageDraw.Draw(frame).text((8,8),f'{a.motion.stem} | {t/50:.2f}s\nKINEMATIC REFERENCE ONLY (not tracking)',fill='white')
                    writer.append_data(np.asarray(frame))
    finally:
        if renderer is not None:
            renderer.close()
    errors=np.asarray(errors)
    report=dict(motion=str(a.motion.resolve()),robot='g1_description/scene.xml',backend='deployment FSM + native official M checkpoint',
                reference_forcing=True,external_support=False,policy_hz=50,physics_hz=500,control_mode=a.control_mode,
                initialization='reset FixedPose, 0.2s settle, start reference at frame 0',
                requested_seconds=tracker.motion_len/50,completed_seconds=len(trace)/50,
                failure=failure,completed=not failure and len(trace)==tracker.motion_len,
                min_root_height_m=min(heights),max_pelvis_tilt_deg=max(tilts),
                mean_root_relative_body_error_m=None if partial else float(errors.mean()),
                mean_root_relative_wrist_error_m=float(errors[:,wrist_indices].mean()),
                actual_root_xy_range_m=np.ptp(np.asarray(trace)[:,:2],axis=0).tolist(),
                scope='One rollout; relative tracking; no opponent, no absolute-location assessment')
    report.update(contact_frames=contact_frames,
                  contact_frame_fraction=contact_frames/max(len(trace),1),
                  contact_slip_mean_m_s=float(np.mean(contact_speeds)) if contact_speeds else None,
                  contact_slip_p95_m_s=float(np.percentile(contact_speeds,95)) if contact_speeds else None,
                  contact_slip_definition='Per-frame max planar speed at foot/world contacts with normal load > 1 N; not ankle speed',
                  contact_slip_fraction_over_0_1_m_s=float(np.mean(np.asarray(contact_speeds)>.1)) if contact_speeds else None)
    if len(trace)>1:
        for label,values in [('actual',actual_wrists),('reference',ref_wrists)]:
            report[label+'_peak_root_relative_wrist_speed_m_s']=np.linalg.norm(np.diff(values,axis=0)*50,axis=-1).max(axis=0).tolist()
    if len(trace)>51:
        for label,values in [('actual',actual_wrists),('reference',ref_wrists)]:
            speed=np.linalg.norm(np.diff(np.asarray(values)[50:],axis=0)*50,axis=-1)
            report[label+'_wrist_speed_p95_after_1s_m_s']=np.percentile(speed,95,axis=0).tolist()
        actual_p95=np.asarray(report['actual_wrist_speed_p95_after_1s_m_s'])
        ref_p95=np.asarray(report['reference_wrist_speed_p95_after_1s_m_s'])
        report['wrist_speed_p95_ratio_after_1s']=(actual_p95/np.maximum(ref_p95,1e-6)).tolist()
    np.savez_compressed(a.out/'trace.npz' ,qpos=trace,actual_bodies=actual_bodies,aligned_references=aligned_references,selected_body_names=np.array(tracker.agent.get_meta_data()['selected_body_names']),body_error=errors,actual_wrists=actual_wrists,reference_wrists=ref_wrists,fps=50)
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
