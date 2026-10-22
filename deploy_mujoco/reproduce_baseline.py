"""Recorded stand-up deployment reproduction, using upstream scene and full FSM.

Only the initial state is placed at the first stand-up reference frame. Thereafter
all state changes come from MuJoCo stepping under the policy's joint PD torques.
"""
import contextlib
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

import mujoco
import numpy as np
import onnx
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.path_config import PROJECT_ROOT
from common.utils import FSMCommand, get_gravity_orientation
from FSM.FSM import FSM

ROOT = Path(PROJECT_ROOT)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True,encoding='utf-8').strip()


def update_state(m,d,s,torso):
    mujoco.mj_forward(m,d)
    s.q=d.qpos[7:36].copy();s.dq=d.qvel[6:35].copy()
    s.gravity_ori=get_gravity_orientation(d.qpos[3:7]);s.ang_vel=d.qvel[3:6].copy()
    rotation=d.xmat[m.body('pelvis').id].reshape(3,3)
    s.root_lin_vel_b=(rotation.T@d.qvel[:3]).astype(np.float32)
    s.root_ang_vel_b=d.qvel[3:6].astype(np.float32)
    s.torso_pos_w=d.xpos[torso].astype(np.float32);s.torso_quat_w=d.xquat[torso].astype(np.float32)
    s.pelvis_pos_w=d.qpos[:3].astype(np.float32);s.pelvis_quat_w=d.qpos[3:7].astype(np.float32)


def run_baseline(cfg):
    options=cfg.baseline
    out=ROOT/str(options.output_dir)
    if out.exists():
        raise FileExistsError(f'Preserve the previous run; choose a new baseline.output_dir: {out}')
    out.mkdir(parents=True)
    snapshots=out/'source_snapshot';snapshots.mkdir()
    paths=['g1_description/g1_liao.xml','deploy_mujoco/config/mujoco.yaml',
           'deploy_mujoco/config/mujoco_standup_baseline.yaml','deploy_mujoco/reproduce_baseline.py',
           'deploy_mujoco/deploy_mujoco.py','FSM/FSM.py','common/ctrlcomp.py','common/utils.py',
           'policy/beyondmimic_mj/BeyondMimicMJ.py','policy/beyondmimic_mj/config/standup_mj.yaml',
           'policy/beyondmimic_mj/model/beyondmimic_mj.onnx',
           'policy/beyondmimic_mj/model/fallAndGetUp2_subject2_mj.npz',
           'policy/fixedpose/FixedPose.py','policy/fixedpose/config/FixedPose.yaml',
           'policy/passive/PassiveMode.py','policy/passive/config/Passive.yaml']
    hashes={rel:sha(ROOT/rel) for rel in paths}
    for rel in paths:
        dest=snapshots/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    source_matches={}
    for rel in ['g1_description/g1_liao.xml','policy/beyondmimic_mj/BeyondMimicMJ.py',
                'policy/beyondmimic_mj/config/standup_mj.yaml',
                'policy/beyondmimic_mj/model/beyondmimic_mj.onnx',
                'policy/beyondmimic_mj/model/fallAndGetUp2_subject2_mj.npz']:
        source_matches[rel]=(git('hash-object',rel)==git('rev-parse','HEAD:'+rel))
    assert all(source_matches.values()),source_matches
    model_onnx=onnx.load(ROOT/'policy/beyondmimic_mj/model/beyondmimic_mj.onnx')
    provenance=dict(repository=git('remote','get-url','origin'),commit=git('rev-parse','HEAD'),
                    branch=git('branch','--show-current'),upstream_core_matches=source_matches,
                    source_sha256=hashes,onnx_metadata={x.key:x.value for x in model_onnx.metadata_props},
                    onnx_producer=dict(name=model_onnx.producer_name,version=model_onnx.producer_version),
                    packages={p:importlib.metadata.version(p) for p in ('mujoco','numpy','onnx','onnxruntime','torch','pillow','imageio-ffmpeg','hydra-core')},
                    python=sys.version,training_framework='mjlab (declared by upstream BeyondMimicMJ.py)',
                    training_configuration_status='Original training commit, PPO/reward/randomization config, seed and checkpoint iteration are not included in this deployment repository.',
                    training_reproduced=False,scene_modified=False,
                    initial_state='Reference clip frame 500 (10.0 s), zero velocities; initialization only, no later resets.',
                    resolved_config=OmegaConf.to_container(cfg,resolve=True))
    # Hash the meshes without duplicating an already version-controlled bundle.
    provenance['mesh_sha256']={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in (ROOT/'g1_description/meshes').glob('*') if p.is_file()}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    OmegaConf.save(cfg,out/'resolved_config.yaml')
    with (out/'runtime.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log):
        result=_simulate(cfg,out)
    print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:
        raise RuntimeError(f'Baseline acceptance failed; inspect {out}/metrics.json')


def _simulate(cfg,out):
    options=cfg.baseline
    m=mujoco.MjModel.from_xml_path(str(ROOT/cfg.xml_path));m.opt.timestep=float(cfg.simulation_dt)
    assert (m.nq,m.nv,m.nu,m.neq)==(43,41,29,0), 'Keep upstream robot and movable box'
    d=mujoco.MjData(m);s=StateAndCmd(29);o=PolicyOutput(29);fsm=FSM(s,o)
    policy=fsm.standup_mj_policy;torso=m.body('torso_link').id
    assert abs(float(cfg.simulation_dt)*int(cfg.control_decimation)-policy.control_dt)<1e-12
    d.qpos[:3]=policy.motion_body_pos[0,0];d.qpos[3:7]=policy.motion_body_quat[0,0]
    d.qpos[7:36]=policy.motion_joint_pos[0];d.qvel[:]=0
    update_state(m,d,s,torso);initial=d.qpos.copy();np.save(out/'initial_qpos.npy',initial)
    limits=np.asarray(cfg.tau_limit,dtype=float)
    warm=policy.WARMUP_STEPS;motion=policy.motion_total_steps
    hold=int(round(float(options.hold_seconds)/policy.control_dt))
    # 3 control intervals enter FixedPose and request the stand-up state.
    controls=3+warm+motion+hold
    controls+=controls%2  # End on a 25 Hz video boundary for 50 Hz control.
    renderer=None;writer=None
    video_times=[];frames=0
    if options.record_video:
        assert int(options.video_fps)==25 and abs(policy.control_dt-.02)<1e-12
        import imageio_ffmpeg
        width=int(options.video_width);height=int(options.video_height)
        m.vis.global_.offwidth=width;m.vis.global_.offheight=height
        renderer=mujoco.Renderer(m,height=height,width=width)
        writer=imageio_ffmpeg.write_frames(str(out/'full_run.mp4'),(width,height),fps=int(options.video_fps),
                                          codec='libx264',quality=8,pix_fmt_out='yuv420p',macro_block_size=1)
        writer.send(None)
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
        camera=mujoco.MjvCamera();camera.lookat[:]=[-.10,0,.55];camera.distance=2.5;camera.azimuth=135;camera.elevation=-16
        vopt=mujoco.MjvOption();vopt.geomgroup[3]=0
    def render(stage):
        nonlocal frames
        if renderer is None:return
        renderer.update_scene(d,camera,scene_option=vopt)
        image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,image.width,78),fill=(16,23,32))
        draw.text((15,9),f'UPSTREAM G1 | learned stand-up | simulation {d.time:05.2f} s | {stage}',font=font,fill='white')
        draw.text((15,42),'MuJoCo dynamics + full FSM + ONNX policy | no replay / no state reset after t=0',font=font,fill=(180,207,227))
        writer.send(np.asarray(image));video_times.append(float(d.time));frames+=1
        if frames==1:image.save(out/'start.png')
    render('reference start-state initialization')
    records=[];physics=[];control_qpos=[];control_qvel=[];actions=[];observations=[];events=[]
    previous=None
    try:
        for k in range(controls):
            update_state(m,d,s,torso)
            if k==0:s.skill_cmd=FSMCommand.POS_RESET
            elif k==2:s.skill_cmd=FSMCommand.CMD_STANDUP_MJ
            obs=(policy._build_obs().copy() if fsm.cur_policy is policy and policy.time_step>=warm
                 else np.full(154,np.nan))
            fsm.run()
            phase_step=policy.time_step-warm-1 if fsm.cur_policy is policy else -warm-1
            stage='FSM initialization' if k<3 else ('warmup' if phase_step<0 else ('motion' if phase_step<motion else 'final hold'))
            state_name=fsm.cur_policy.name_str
            if state_name!=previous:
                events.append(dict(time_s=float(d.time),state=state_name,control_step=k));previous=state_name
            for substep in range(int(cfg.control_decimation)):
                raw=o.kps*(o.actions-d.qpos[7:36])-o.kds*d.qvel[6:35]
                d.ctrl[:]=np.clip(raw,-limits,limits)
                assert not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
                mujoco.mj_step(m,d)
                physics.append(np.r_[d.time,d.qpos,d.qvel,d.ctrl])
                assert np.isfinite(physics[-1]).all(), 'Nonfinite simulation state'
            mujoco.mj_forward(m,d)
            ref=max(0,min(phase_step,motion-1))
            tilt=np.rad2deg(np.arccos(np.clip(-get_gravity_orientation(d.qpos[3:7])[2],-1,1)))
            err=float(np.sqrt(np.mean((d.qpos[7:36]-policy.motion_joint_pos[ref])**2)))
            floor_bodies=[]
            for c in d.contact:
                pair=[int(c.geom1),int(c.geom2)]
                if m.geom('floor').id in pair:
                    g=pair[1] if pair[0]==m.geom('floor').id else pair[0]
                    floor_bodies.append(m.body(m.geom_bodyid[g]).name)
            records.append(dict(time_s=float(d.time),stage=stage,state=state_name,reference_frame=ref,
                                root_x_m=d.qpos[0],root_y_m=d.qpos[1],root_height_m=d.qpos[2],
                                tilt_deg=float(tilt),joint_rmse_rad=err,max_abs_torque_nm=float(abs(d.ctrl).max()),
                                floor_contact_bodies=';'.join(sorted(set(floor_bodies)))))
            control_qpos.append(d.qpos.copy());control_qvel.append(d.qvel.copy());actions.append(o.actions.copy());observations.append(obs)
            if (k+1)%2==0:render(stage)
        if renderer is not None:
            Image.fromarray(renderer.render()).save(out/'finish.png')
    finally:
        if writer is not None:writer.close()
        if renderer is not None:renderer.close()
    with (out/'control_log.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
    np.savez_compressed(out/'trajectory.npz',physics=np.asarray(physics),qpos=np.asarray(control_qpos),
                        qvel=np.asarray(control_qvel),target_joint_positions=np.asarray(actions),
                        policy_observations=np.asarray(observations),
                        video_sim_times_s=video_times,initial_qpos=initial)
    (out/'fsm_events.json').write_text(json.dumps(events,indent=2))
    tail=[r for r in records if r['stage']=='final hold']
    tracked=[r for r in records if r['stage']=='motion']
    metrics=dict(duration_s=float(d.time),physics_steps=len(physics),control_steps=len(records),
                 motion_frames=motion,executed_motion_frames=len(tracked),video_frames=frames,
                 initial_root_height_m=float(initial[2]),final_root_height_m=float(d.qpos[2]),
                 final_hold_height_min_m=min(r['root_height_m'] for r in tail),
                 final_hold_height_mean_m=float(np.mean([r['root_height_m'] for r in tail])),
                 final_hold_tilt_max_deg=max(r['tilt_deg'] for r in tail),
                 motion_joint_rmse_rad=float(np.sqrt(np.mean([r['joint_rmse_rad']**2 for r in tracked]))),
                 no_state_resets_after_initialization=True,no_external_forces=True,
                 mujoco_warning_counts=[int(w.number) for w in d.warning],fsm_events=events)
    criteria=OmegaConf.to_container(options.acceptance)
    checks=dict(initially_lying=metrics['initial_root_height_m']<criteria['initial_root_height_max_m'],
                all_motion_frames=len(tracked)==motion,
                standing_height_min=metrics['final_hold_height_min_m']>criteria['final_hold_root_height_min_m'],
                standing_height_mean=metrics['final_hold_height_mean_m']>criteria['final_hold_root_height_mean_min_m'],
                standing_tilt=metrics['final_hold_tilt_max_deg']<criteria['final_hold_tilt_max_deg'],
                joint_tracking=metrics['motion_joint_rmse_rad']<criteria['motion_joint_rmse_max_rad'],
                no_simulation_warnings=not any(metrics['mujoco_warning_counts']))
    checks={name:bool(value) for name,value in checks.items()}
    metrics.update(criteria=criteria,checks=checks,passed=all(checks.values()))
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    return metrics
