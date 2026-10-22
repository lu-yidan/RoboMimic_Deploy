"""Nominal support commissioning through the Climb adapter, no learned actor.

Reuses the validated environment's physics loop and verifies deployment target
positions and gains before every environment action. The actor returns zero
residual, so this experiment is explicitly a fixed-target joint PD baseline.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import imageio_ffmpeg
import mujoco
import numpy as np
from omegaconf import OmegaConf
from PIL import Image,ImageDraw,ImageFont

from common.ctrlcomp import StateAndCmd,PolicyOutput
from common.path_config import PROJECT_ROOT
from common.utils import FSMStateName
from FSM.FSM import FSM,FSMMode
from training.wall_support_env import WallSupportEnv

ROOT=Path(PROJECT_ROOT)


class ZeroResidualActor:
    def __call__(self,observation):
        if observation.shape!=(100,) or not np.isfinite(observation).all():
            raise ValueError('Invalid actor observation')
        return np.zeros(29,np.float32)


def run_wall_support(cfg):
    out=ROOT/str(cfg.output_dir);bundle=ROOT/str(cfg.bundle)
    if out.exists():raise FileExistsError('Select a new wall_support.output_dir')
    out.mkdir(parents=True)
    env=WallSupportEnv(bundle,horizon_s=float(cfg.horizon_s))
    obs,_=env.reset(seed=20261009)
    state=StateAndCmd(29);output=PolicyOutput(29)
    fsm=FSM(state,output,profile='climb',climb_options=dict(config_path=bundle/'contract.json',actor=ZeroResidualActor()))
    fsm.get_next_policy(FSMStateName.SKILL_CLIMB);fsm.FSMmode=FSMMode.CHANGE
    source_paths=['training/climb_env.py','training/wall_support_env.py','policy/climb/Climb.py',
                  'policy/climb/contract.py','deploy_mujoco/run_wall_support.py','FSM/FSM.py']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths}
    for p in source_paths:
        dest=out/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dest)
    manifest=dict(task='Nominal support under constant joint targets',trained_actor=False,
                  actor='ZeroResidualActor; not the flat-ground stand-up policy and not a learned climb policy',
                  config=OmegaConf.to_container(cfg),contract_sha256=env.contract.fingerprint,
                  bundle_manifest_sha256=hashlib.sha256((bundle/'manifest.json').read_bytes()).hexdigest(),
                  source_sha256=hashes,physics_dt=env.model.opt.timestep,control_dt=env.contract.control_dt,
                  initialization='One reset to the declared support pose; directly activate Climb FSM state.',
                  contact_force_sampling='Last solver result at each 50 Hz control step, not high-frequency peak loads')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    shutil.copy2(bundle/'contract.json',out/'contract.json')
    renderer=writer=None;video_times=[];frame_count=0
    if cfg.video:
        env.model.vis.global_.offwidth=1280;env.model.vis.global_.offheight=720
        renderer=mujoco.Renderer(env.model,height=720,width=1280)
        writer=imageio_ffmpeg.write_frames(str(out/'support_full_run.mp4'),(1280,720),fps=25,codec='libx264',quality=8,macro_block_size=1)
        writer.send(None);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
        camera=mujoco.MjvCamera();camera.lookat[:]=[.30,0,1.02];camera.distance=2.6;camera.azimuth=-35;camera.elevation=-9
        opt=mujoco.MjvOption();opt.geomgroup[3]=0;opt.geomgroup[4]=1
    def frame(info):
        nonlocal frame_count
        if renderer is None:return
        renderer.update_scene(env.data,camera,scene_option=opt)
        image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,1280,80),fill=(17,26,35))
        draw.text((15,8),f'RUNG WALL | fixed-target PD (not learned) | simulation {env.data.time:05.2f} s',font=font,fill='white')
        label='Declared initial support pose; free base; no external support forces'
        if info is not None:
            label=f'Root drift {1000*info["root_error_m"]:.1f} mm | hook vertical forces {info["left_hook_vertical_n"]:.1f} / {info["right_hook_vertical_n"]:.1f} N'
        draw.text((15,42),label,font=font,fill=(190,216,234));writer.send(np.asarray(image))
        frame_count+=1;video_times.append(float(env.data.time))
        if info is None:image.save(out/'start.png')
        return image
    frame(None)
    rows=[];qpos=[env.data.qpos.copy()];qvel=[env.data.qvel.copy()];targets=[];observation_rows=[obs.copy()]
    contact_rows=[];last_image=None;max_adapter_error=0.
    try:
        for step in range(env.max_steps):
            state.q=env.data.qpos[env.qadr].copy();state.dq=env.data.qvel[env.vadr].copy()
            state.gravity_ori=obs[:3].copy();state.ang_vel=obs[3:6].copy()
            state.climb_target_pos_b=obs[93:96].copy();state.climb_target_axis_b=obs[96:99].copy();state.climb_target_valid=bool(obs[99])
            fsm.run();policy=fsm.climb_policy
            if policy.fault:raise RuntimeError(policy.fault)
            np.testing.assert_array_equal(obs,policy.last_observation)
            target,_=env.contract.decode_action(policy.previous_action)
            max_adapter_error=max(max_adapter_error,float(np.max(abs(output.actions-target))))
            np.testing.assert_array_equal(output.actions,target)
            np.testing.assert_array_equal(output.kps,env.contract.kps);np.testing.assert_array_equal(output.kds,env.contract.kds)
            targets.append(output.actions.copy())
            obs,reward,terminated,truncated,info=env.step(policy.previous_action)
            assert not np.any(env.data.qfrc_applied) and not np.any(env.data.xfrc_applied)
            record={k:v for k,v in info.items() if k not in ('contacts','force_sample')}
            record.update(policy_step=step,reward=reward,controller='fixed-target PD')
            rows.append(record);contact_rows.append(dict(time_s=float(env.data.time),contacts=info['contacts']))
            qpos.append(env.data.qpos.copy());qvel.append(env.data.qvel.copy());observation_rows.append(obs.copy())
            if (step+1)%2==0 or terminated or truncated:last_image=frame(info)
            if terminated or truncated:break
    finally:
        if writer is not None:writer.close()
        if renderer is not None:renderer.close()
    if last_image is not None:last_image.save(out/'finish.png')
    with (out/'control_log.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (out/'contact_log.json').write_text(json.dumps(contact_rows,indent=2))
    np.savez_compressed(out/'trajectory.npz',qpos=qpos,qvel=qvel,observations=observation_rows,
                        targets=targets,time_s=[0.]+[r['simulation_time_s'] for r in rows],video_times_s=video_times)
    tail=[r for r in rows if r['simulation_time_s']>=max(0.,rows[-1]['simulation_time_s']-2.)]
    force_names=['left_hook_vertical_n','right_hook_vertical_n','left_foot_vertical_n','right_foot_vertical_n','other_robot_vertical_n','total_contact_vertical_n']
    means={name:float(np.mean([r[name] for r in tail])) for name in force_names}
    weight=env.robot_weight_n
    # Geometric pose accuracy and contact loading are separate acceptance items.
    checks=dict(declared_duration=not terminated and abs(env.data.time-float(cfg.horizon_s))<1e-8,
                pose_hold=bool(info['pose_hold_success']),
                each_hook_loaded=all(means[n]>10. for n in force_names[:2]),
                each_foot_loaded=all(means[n]>10. for n in force_names[2:4]),
                no_other_support=abs(means['other_robot_vertical_n'])<1.,
                vertical_force_balance=abs(means['total_contact_vertical_n']-weight)/weight<.02,
                no_torque_saturation=max(r['motor_saturation_fraction'] for r in rows)==0.,
                adapter_equivalence=max_adapter_error==0.)
    summary=dict(passed=all(checks.values()),checks=checks,final=rows[-1],
                 last_two_seconds_mean_forces=means,robot_weight_n=weight,
                 hooks_fraction_of_weight=(means['left_hook_vertical_n']+means['right_hook_vertical_n'])/weight,
                 video_frames=frame_count,max_adapter_target_error_rad=max_adapter_error,
                 trained_actor=False,ascent_performed=False,
                 scope='One nominal support pose only; no disturbance, motion transition or mechanical strength validation.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));env.close()
    print(json.dumps(summary,indent=2),flush=True)
