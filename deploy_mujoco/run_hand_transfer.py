"""Free-base dynamics with scripted full-position IK references; no learned actor."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import imageio_ffmpeg
import mujoco
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from common.path_config import PROJECT_ROOT
from training.wall_support_env import WallSupportEnv
from training.hand_transfer import RightHookReference, smooth

ROOT = Path(PROJECT_ROOT)


def run_hand_transfer(cfg):
    out = ROOT/str(cfg.output_dir)
    if out.exists():
        raise FileExistsError('Choose a new hand_transfer.output_dir')
    out.mkdir(parents=True)
    env = WallSupportEnv(ROOT/str(cfg.bundle), horizon_s=18.)
    env.reset()
    m, d = env.model, env.data
    reference = RightHookReference(env, clearance=float(cfg.clearance), retreat=float(cfg.retreat), rise=float(cfg.rise), initial_clearance=float(cfg.initial_clearance))
    kp, kd = env.contract.kps.copy(), env.contract.kds.copy()
    kp[:22] *= float(cfg.support_gain)
    kd[:22] *= np.sqrt(float(cfg.support_gain))
    kp[22:] *= float(cfg.arm_gain)
    kd[22:] *= np.sqrt(float(cfg.arm_gain))
    commanded = env.contract.default_q.copy()
    load_offset = 0.
    rows, states, velocities, targets, contacts, video_times = [], [d.qpos.copy()], [d.qvel.copy()], [], [], []
    renderer = writer = None
    if cfg.video:
        m.vis.global_.offwidth, m.vis.global_.offheight = 1280,720
        renderer = mujoco.Renderer(m,height=720,width=1280)
        writer = imageio_ffmpeg.write_frames(str(out/'transfer_full_run.mp4'),(1280,720),fps=25,codec='libx264',quality=8,macro_block_size=1)
        writer.send(None)
        camera=mujoco.MjvCamera();camera.lookat[:]=[.30,0,1.13];camera.distance=2.6;camera.azimuth=-45;camera.elevation=-7
        opt=mujoco.MjvOption();opt.geomgroup[3]=0;opt.geomgroup[4]=1
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
    def frame(row):
        if renderer is None:return
        renderer.update_scene(d,camera,scene_option=opt)
        image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,1280,80),fill=(17,26,35))
        phase = 'initial' if row is None else row['phase']
        task = 'SAME-RUNG RELEASE / RESEAT' if float(cfg.rise)==0 else '25 CM RUNG TRANSFER ATTEMPT'
        draw.text((15,8),f'{task} | {d.time:05.2f} s | {phase}',font=font,fill='white')
        label='Free base, motor torques only; ideal simulator pose feedback; not a learned policy'
        if row:
            label=f'Hook load L/R: {row["left_hook_vertical_n"]:.1f}/{row["right_hook_vertical_n"]:.1f} N | root drift {row["root_error_m"]*1000:.1f} mm'
        draw.text((15,42),label,font=font,fill=(190,216,234))
        writer.send(np.asarray(image));video_times.append(float(d.time))
        image.save(out/('start.png' if row is None else 'finish.png'))
    frame(None)
    reason='complete';warning_start=d.warning.number.copy()
    physics_other_peak=0.;physics_root_peak=0.;physics_tilt_peak=0.
    try:
        for step in range(900):
            desired, phase=reference.target(d.time)
            if d.time >= 14. and float(cfg.load_goal_n)>0:
                measured = env.contact_metrics()['right_hook_vertical_n']
                load_offset = float(np.clip(load_offset+.0002*.02*(float(cfg.load_goal_n)-measured),0.,.025))
                desired[2] -= load_offset
            commanded[14] = env.contract.default_q[14]+float(cfg.waist_lean)*smooth((d.time-6.)/4.)
            lean = float(cfg.leg_lean)*smooth((d.time-6.)/4.)
            commanded[[0,6]] = env.contract.default_q[[0,6]]+lean
            commanded[[4,10]] = env.contract.default_q[[4,10]]-lean
            arm, pos_error, rot_error=reference.solve(d.qpos if cfg.base_feedback else env.initial,desired)
            # Full-position reference has its own explicit velocity bound. It is
            # deliberately not clipped into the old small residual actor range.
            change=np.clip(arm-commanded[22:],-float(cfg.joint_speed)*.02,float(cfg.joint_speed)*.02)
            commanded[22:] += change
            saturation=0
            for substep in range(env.substeps):
                torque=kp*(commanded-d.qpos[env.qadr])-kd*d.qvel[env.vadr]
                if cfg.arm_gravity:
                    torque[22:] += d.qfrc_bias[env.vadr[22:]]
                saturation += int(np.sum((torque<env.limits[:,0])|(torque>env.limits[:,1])))
                d.ctrl[env.motors]=np.clip(torque,env.limits[:,0],env.limits[:,1])
                assert not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
                mujoco.mj_step(m,d);mujoco.mj_kinematics(m,d)
                root,tilt,_=env._metrics()
                physics_root_peak=max(physics_root_peak,root);physics_tilt_peak=max(physics_tilt_peak,tilt)
                for ci,c in enumerate(d.contact):
                    ba,bb=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
                    ina,inb=ba in env.robot_bodies,bb in env.robot_bodies
                    if ina==inb:continue
                    body=ba if ina else bb
                    if body not in env.contact_groups:
                        wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
                        physics_other_peak=max(physics_other_peak,float(np.linalg.norm(wrench[:3])))
                if not np.isfinite(d.qpos).all() or np.any(d.warning.number>warning_start): reason='physics_failure'
                elif root>.2 or tilt>np.deg2rad(45):reason='support_loss'
                if reason!='complete':break
            force=env.contact_metrics();contact=force.pop('contacts');force.pop('force_sample')
            other_norm=sum(float(np.linalg.norm(c['force_on_robot_world_n'])) for c in contact if c['group']=='other_robot')
            right_norm=sum(float(np.linalg.norm(c['force_on_robot_world_n'])) for c in contact if c['group']=='right_hook')
            row=dict(time_s=float(d.time),phase=phase,root_error_m=root,tilt_rad=tilt,
                     ik_position_error_m=pos_error,ik_orientation_error_rad=rot_error,
                     tracking_error_m=float(np.linalg.norm(d.xpos[reference.body]-desired)),
                     right_hook_height_m=float(d.xpos[reference.body,2]),
                     right_hook_x_m=float(d.xpos[reference.body,0]),
                     right_hook_y_m=float(d.xpos[reference.body,1]),
                     other_contact_norm_sum_n=other_norm,right_contact_norm_sum_n=right_norm,
                     load_offset_m=load_offset,
                     saturation_fraction=saturation/(29*(substep+1)),**force)
            rows.append(row);states.append(d.qpos.copy());velocities.append(d.qvel.copy());targets.append(commanded.copy())
            contacts.append(dict(time_s=float(d.time),phase=phase,contacts=contact))
            if (step+1)%2==0 or reason!='complete':frame(row)
            if (step+1)%100==0:print(json.dumps(row),flush=True)
            if reason!='complete':break
    finally:
        if writer is not None:writer.close()
        if renderer is not None:renderer.close()
    with (out/'control_log.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (out/'contact_log.json').write_text(json.dumps(contacts,indent=2))
    np.savez_compressed(out/'trajectory.npz',qpos=states,qvel=velocities,targets=targets,
                        time_s=[0.]+[r['time_s'] for r in rows],video_times_s=video_times)
    tail=[r for r in rows if r['time_s']>=16.]
    unloaded=[r for r in rows if r['phase'] in ('retreat','rise','free_hold')]
    final_contacts=[c for record in contacts if record['time_s']>=16. for c in record['contacts']
                    if c['group']=='right_hook' and np.linalg.norm(c['force_on_robot_world_n'])>1.]
    checks=dict(duration=reason=='complete' and abs(d.time-18.)<1e-8,
                released=bool(unloaded) and sum(r['right_contact_norm_sum_n']<1 for r in unloaded)*.02>=1.,
                target_rung_loaded=bool(tail) and min(r['right_hook_vertical_n'] for r in tail)>10.,
                correct_rung=bool(final_contacts) and all(c['external_geom']==f'rung_{5+round(float(cfg.rise)/.25):02d}' for c in final_contacts),
                final_grasp_alignment=bool(tail) and max(abs(r['right_hook_x_m']-(reference.start[0]+float(cfg.rise)*np.tan(np.deg2rad(5.)))) for r in tail)<.01 and max(abs(r['right_hook_y_m']+.2) for r in tail)<.02,
                remaining_supports_loaded=bool(tail) and all(min(r[n] for r in tail)>10. for n in ['left_hook_vertical_n','left_foot_vertical_n','right_foot_vertical_n']),
                stable_base=physics_root_peak<.06 and physics_tilt_peak<np.deg2rad(15),
                ik_path_feasible=max(r['ik_position_error_m'] for r in rows)<.005,
                no_other_support=physics_other_peak<1.,
                no_saturation=max(r['saturation_fraction'] for r in rows)==0.)
    summary=dict(passed=bool(all(checks.values())),checks={k:bool(v) for k,v in checks.items()},reason=reason,
                 peak_root_error_m=physics_root_peak,peak_tilt_deg=float(np.rad2deg(physics_tilt_peak)),
                 physics_other_contact_peak_n=physics_other_peak,
                 task='same_rung_release_reseat' if float(cfg.rise)==0. else 'one_rung_transfer',
                 peak_ik_error_m=max(r['ik_position_error_m'] for r in rows),
                 final=rows[-1],last_2s_right_load_n=float(np.mean([r['right_hook_vertical_n'] for r in tail])) if tail else None,
                 trained_actor=False,whole_body_ascent=False,config=OmegaConf.to_container(cfg))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    paths=['training/hand_transfer.py','training/wall_support_env.py','training/climb_env.py','deploy_mujoco/run_hand_transfer.py',
           'deploy_mujoco/deploy_mujoco.py',
           'deploy_mujoco/config/mujoco_hand_transfer.yaml']
    for p in paths:
        dest=out/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dest)
    manifest=dict(controller='Scripted Cartesian reference, IK, joint PD, optional right-arm model bias compensation and integral load probing; separate full-position bridge',
                  state_source='Ideal simulator free-base/joint pose and right-hook contact-force truth; no perception or sensor noise',
                  bundle=str(env.bundle),bundle_manifest_sha256=hashlib.sha256((env.bundle/'manifest.json').read_bytes()).hexdigest(),
                  kp=kp.tolist(),kd=kd.tolist(),control_dt=.02,physics_dt=m.opt.timestep,
                  no_state_overwrite_after_reset=True,external_forces=False,
                  force_sampling='Last solver result sampled at 50 Hz, not transient peak loads',
                  source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(summary,indent=2),flush=True);env.close()
