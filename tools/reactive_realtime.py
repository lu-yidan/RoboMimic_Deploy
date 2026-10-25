"""Wall-clock R2R -> no-IK pelvis/hands -> ScaleBFM -> local MuJoCo.

No DDS or real robot. The prerecorded opponent is released by source timestamps.
Both learned models stay loaded. Only startup/reference/trace artifacts use files;
4-frame chunks travel over a local Unix socket. Upper policy has no G1 feedback.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import socket
import sys
import tempfile
import time
from multiprocessing.connection import Listener
from types import MethodType
os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from omegaconf import OmegaConf
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.utils import get_gravity_orientation
from FSM.FSM import FSM
from deploy_mujoco.reset_sim import reset_to_fixed_pose
from deploy_mujoco.deploy_mujoco import pd_control


class SparseReference:
    """Incremental 30->50Hz resampling; retain timestamps and a fixed world frame."""
    def __init__(self, metadata, config):
        self.metadata=metadata
        self.human_names=['Hips','LeftHand','RightHand']
        self.robot_names=['pelvis','left_wrist_yaw_link','right_wrist_yaw_link']
        self.human_ids=[0,8,12]
        table=config['ik_match_table1']
        entries=[table[n] for n in self.robot_names]
        self.offset=np.array([e[3] for e in entries])
        self.rot_offset=Rotation.from_quat(np.array([e[4] for e in entries])[:,[1,2,3,0]])
        self.scale=np.array([config['human_scale_table'][n] for n in self.human_names])[:,None]
        self.pos=[];self.quat=[];self.sample_pos=[];self.sample_quat=[]
        self.mapping_ms=[]
        self.align=Rotation.identity();self.origin=np.zeros(3);self.source_origin=np.zeros(3)

    def append(self, msg):
        start=time.perf_counter()
        assert msg['end']==len(self.pos)+len(msg['positions'])
        for p,r in zip(msg['positions'][:,0],msg['rotations'][:,0]):
            rr=Rotation.from_matrix(r[self.human_ids])*self.rot_offset
            pp=(p[self.human_ids]-p[0])*self.scale+p[0]*self.scale[0]+rr.apply(self.offset)
            self.pos.append(pp);self.quat.append(rr.as_quat())
        # Include the prior boundary frame for cross-chunk interpolation.
        first=max(0,len(self.pos)-len(msg['positions'])-1)
        source_t=np.arange(first,len(self.pos))/30
        times=np.arange(len(self.sample_pos),int((len(self.pos)-1)*50/30)+1)/50
        ps=np.asarray(self.pos[first:]);qs=np.asarray(self.quat[first:])
        if len(times):
            po=np.empty((len(times),3,3));qo=np.empty((len(times),3,4))
            for j in range(3):
                po[:,j]=np.column_stack([np.interp(times,source_t,ps[:,j,k]) for k in range(3)])
                qo[:,j]=Slerp(source_t,Rotation.from_quat(qs[:,j]))(times).as_quat()
            self.sample_pos.extend(po);self.sample_quat.extend(qo)
        self.mapping_ms.append((time.perf_counter()-start)*1000)

    def calibrate(self, current_pos, current_quat):
        def yaw(r):
            m=r.as_matrix();return np.arctan2(m[1,0],m[0,0])
        self.align=Rotation.from_euler('z',yaw(Rotation.from_quat(current_quat[[1,2,3,0]]))-yaw(Rotation.from_quat(self.quat[0][0])))
        self.origin=np.array(current_pos).copy();self.origin[2]=0
        self.source_origin=self.pos[0][0].copy();self.source_origin[2]=0

    def window(self, frame):
        idx=np.minimum(np.arange(frame,frame+6),len(self.sample_pos)-1)
        pos=np.array([self.sample_pos[i] for i in idx]);quat=np.array([self.sample_quat[i] for i in idx])
        pos=self.align.apply((pos-self.source_origin).reshape(-1,3)).reshape(6,3,3)+self.origin
        quat=(self.align*Rotation.from_quat(quat.reshape(-1,4))).as_quat().reshape(6,3,4)
        return pos,quat

    def bootstrap(self, path):
        b=len(self.metadata['body_names']);pos=np.zeros((6,b,3),np.float32);q=np.zeros((6,b,4),np.float32);q[:,:,0]=1
        pp,qq=self.window(0)
        for j,name in enumerate(self.robot_names):
            i=self.metadata['body_names'].index(name);pos[:,i]=pp[:,j];q[:,i]=qq[:,j,[3,0,1,2]]
        np.savez_compressed(path,body_pos_w=pos,body_quat_w=q,joint_pos=np.zeros((6,29),np.float32),joint_vel=np.zeros((6,29),np.float32),body_lin_vel_w=np.zeros_like(pos),body_ang_vel_w=np.zeros_like(pos),fps=50,body_names=self.metadata['body_names'],joint_names=self.metadata['joint_names'])


def stats(x):
    return dict(median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(np.max(x))) if len(x) else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--telemetry-port',type=int,default=0)
    p.add_argument('--opponent-mode',choices=['normal','freeze','delay'],default='normal')
    p.add_argument('--intervention-s',type=float,default=2)
    p.add_argument('--opponent-delay-ms',type=float,default=500)
    p.add_argument('--clip',type=int,default=3);p.add_argument('--frames',type=int,default=240)
    p.add_argument('--fixed',type=int,choices=[0,1],default=1);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--delay-ms',type=float,default=300);p.add_argument('--blend-ms',type=float,default=100)
    p.add_argument('--stall-at',type=int,default=-1);p.add_argument('--stall-ms',type=float,default=0)
    a=p.parse_args()
    if min(a.delay_ms,a.blend_ms,a.stall_ms)<0:p.error('Durations must be nonnegative')
    if a.frames<8:p.error('At least 8 source frames required')
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    ws=ROOT.parent
    meta=json.loads((ws/'ScaleBFM/artifacts/scalebfm/humanoid_transformer_m/model_22200_tensorrt_metadata.json').read_text())
    ik=json.loads((ws/'GMR/general_motion_retargeting/ik_configs/fbx_offline_to_g1.json').read_text())
    ref=SparseReference(meta,ik)
    chunks=[];all_pos=[];all_rot=[];worker=None;conn=None;trace=[];ticks=[];failure=None
    with tempfile.TemporaryDirectory(prefix='r2r-') as ipc, (a.out/'generator.log').open('w') as log:
        listener=Listener(str(Path(ipc)/'data.sock'),family='AF_UNIX')
        # Bound accept so a producer initialization error cannot hang the test.
        listener._listener._socket.settimeout(180)
        request=dict(clip=a.clip,start=0,frames=a.frames,fixed=a.fixed,seed=a.seed,socket=listener.address,torch_threads=1,stall_at=a.stall_at,stall_ms=a.stall_ms,opponent_mode=a.opponent_mode,intervention_s=a.intervention_s,opponent_delay_ms=a.opponent_delay_ms)
        (a.out/'request.json').write_text(json.dumps(request,indent=2))
        try:
            worker=subprocess.Popen([str(ws/'ready_to_react/.venv/bin/python'),str(ws/'ready_to_react/scripts/reactive_stream_worker.py'),'--request',str(a.out/'request.json')],cwd=ws/'ready_to_react',stdout=log,stderr=subprocess.STDOUT)
            conn=listener.accept()
            if not conn.poll(60):raise TimeoutError('No warmup from producer')
            first=conn.recv();ref.append(first);all_pos.extend(first['positions']);all_rot.extend(first['rotations'])
            total=first['total_frames'];ref.bootstrap(a.out/'bootstrap.npz')
            from common.deployment_config import load_scalebfm_config
            cfg=OmegaConf.load(ROOT/'deploy_mujoco/config/mujoco.yaml')
            cfg.scalebfm = load_scalebfm_config(require_enabled=True)
            cfg.scalebfm.motion=str(a.out/'bootstrap.npz');cfg.scalebfm.control_mode=2
            model=mujoco.MjModel.from_xml_path(str(ROOT/'g1_description/scene.xml'));model.opt.timestep=.002
            data=mujoco.MjData(model);state,output=StateAndCmd(29),PolicyOutput(29)
            names=[model.joint(int(model.actuator_trnid[i,0])).name for i in range(29)]
            fsm=FSM(state,output,cfg.scalebfm,names,enable_freekick=False)
            tracker=fsm.scalebfm_policy
            reset_to_fixed_pose(model,data,fsm,state,output)
            limits=np.array(cfg.tau_limit)
            def read_state():
                state.q=data.qpos[7:36].copy();state.dq=data.qvel[6:35].copy()
                state.pelvis_pos_w=data.qpos[:3].astype(np.float32);state.pelvis_quat_w=data.qpos[3:7].astype(np.float32)
                state.root_ang_vel_b=data.qvel[3:6].astype(np.float32);state.gravity_ori=get_gravity_orientation(data.qpos[3:7])
            def physics():
                assert not data.xfrc_applied.any()
                for _ in range(10):
                    data.ctrl[:]=np.clip(pd_control(output.actions,data.qpos[7:36],output.kps,np.zeros(29),data.qvel[6:35],output.kds),-limits,limits)
                    mujoco.mj_step(model,data)
                mujoco.mj_forward(model,data)
            for _ in range(10):read_state();fsm.run();physics()
            read_state();ref.calibrate(data.qpos[:3],data.qpos[3:7]);tracker.enter()
            tracker._show_progress=lambda: None
            tracker.motion_len=int((total-1)*50/30)+1
            # Always supply exactly the six time-indexed targets for this control tick.
            # Episode/history still advance normally in tracker.run().
            tracker.env._global_time_indices=MethodType(lambda self,offsets: offsets[None,:].clamp(0,5),tracker.env)
            telemetry=socket.socket(socket.AF_INET,socket.SOCK_DGRAM) if a.telemetry_port else None
            if telemetry:
                telemetry.setblocking(False)
                visual_model=mujoco.MjModel.from_xml_path(str(ws/'GMR/assets/unitree_g1/g1_mocap_29dof.xml'));visual_data=mujoco.MjData(visual_model)
                visual_ids=[visual_model.joint(n).qposadr[0] for n in names]
                raw_files=sorted((ws/'ready_to_react/data/DuoBox/fbx_export_joints').glob('*.npy'))
                raw=np.load(raw_files[a.clip],allow_pickle=True).item();who=first['actors'][1]
                live_opponent=np.asarray(raw[who+'_position'][4::4]).copy()
                # Display only: recorded self shares the generated skeleton's 30Hz index.
                # Never send these future recorded poses to the generator or tracker.
                recorded_self=np.asarray(raw[first['actors'][0]+'_position'][4::4])
                cut=int(a.intervention_s*30);ix=np.arange(len(live_opponent))
                if a.opponent_mode=='freeze':ix[cut:]=min(cut,len(ix)-1)
                elif a.opponent_mode=='delay':ix[cut:]=np.maximum(cut,ix[cut:]-int(a.opponent_delay_ms*.03))
                live_opponent=live_opponent[ix]
            selected=meta['selected_body_names'];ids=[selected.index(n) for n in ref.robot_names]
            bodypos=np.zeros((6,len(selected),3),np.float32);bodyquat=np.zeros((6,len(selected),4),np.float32);bodyquat[:,:,0]=1
            epoch=time.monotonic()+.1;conn.send({'epoch':epoch});start=epoch+a.delay_ms/1000
            steps=int((total-1)*50/30)+1;finished=False;was_starved=False;blend_start=None;blend_pos=None;blend_rot=None;previous=None;previous_target=None;previous_available=len(ref.sample_pos);targets=[]
            for t in range(steps+25):
                deadline=start+t*.02
                remaining=deadline-time.monotonic()
                if remaining>0:time.sleep(remaining)
                begun=time.monotonic()
                while not finished and conn.poll():
                    msg=conn.recv()
                    if msg.get('done'):finished=True;break
                    received=time.monotonic();ref.append(msg);all_pos.extend(msg['positions']);all_rot.extend(msg['rotations'])
                    chunks.append(dict(end=msg['end'],observed_until=msg['observed_until'],prediction_ms=msg['prediction_ms'],received_s=received-epoch,source_observation_s=msg['observed_until']/30,available_latency_ms=(received-epoch-msg['observed_until']/30)*1000,first_use_s=None))
                if worker.poll() not in (None,0):raise RuntimeError('Generator failed; see generator.log')
                starved=t<steps and t>=len(ref.sample_pos)
                preview_short=t<steps and min(t+5,steps-1)>=len(ref.sample_pos)
                pp,qq=ref.window(t)
                if was_starved and (not starved or len(ref.sample_pos)!=previous_available) and previous_target is not None:
                    blend_start=t;blend_pos=previous_target[0][0].copy();blend_rot=previous_target[1][0].copy()
                if blend_start is not None and a.blend_ms>0:
                    alpha=min(1,(t-blend_start+1)*20/a.blend_ms);alpha=alpha*alpha*(3-2*alpha)
                    pp=blend_pos[None]+alpha*(pp-blend_pos[None])
                    old=Rotation.from_quat(np.broadcast_to(blend_rot,(6,3,4)).reshape(-1,4));new=Rotation.from_quat(qq.reshape(-1,4))
                    qq=(Rotation.from_rotvec((new*old.inv()).as_rotvec()*alpha)*old).as_quat().reshape(6,3,4)
                    if alpha>=1:blend_start=None
                jump=0. if previous_target is None else float(np.max(np.linalg.norm(pp[0]-previous_target[0][0],axis=-1)))
                previous_target=(pp.copy(),qq.copy());was_starved=starved;previous_available=len(ref.sample_pos)
                targets.append(pp[0].copy())
                bodypos[:,ids]=pp;bodyquat[:,ids]=qq[:,:,[3,0,1,2]]
                tracker.env.body_pos_w=torch.as_tensor(bodypos.copy(),device='cuda');tracker.env.body_quat_w=torch.as_tensor(bodyquat.copy(),device='cuda')
                read_state();tracker.run();physics()
                done=time.monotonic();tilt=float(np.degrees(np.arccos(np.clip(data.body('pelvis').xmat[8],-1,1))))
                for c in chunks:
                    if (c['first_use_s'] is None
                        and (c['end']-4)/30<=min((t+5)/50,(len(ref.pos)-1)/30)
                        and (c['end']-1)/30>=min(t/50,(len(ref.pos)-1)/30)):
                        c['first_use_s']=done-epoch
                trace.append(data.qpos.copy())
                ticks.append(dict(frame=t,wall_s=done-epoch,starved=starved,preview_short=preview_short,available_frames=len(ref.sample_pos),start_late_ms=max(0,(begun-deadline)*1000),compute_ms=(done-begun)*1000,deadline_miss_ms=max(0,(done-deadline-.02)*1000),target_step_m=jump,joint_step_rad=0 if previous is None else float(np.max(np.abs(data.qpos[7:]-previous[7:]))),tilt_deg=tilt,root_height=float(data.qpos[2])))
                previous=data.qpos.copy()
                if telemetry and t%2==0:
                    visual_data.qpos[:7]=data.qpos[:7];visual_data.qpos[visual_ids]=data.qpos[7:36];mujoco.mj_forward(visual_model,visual_data)
                    oi=min(max(0,int((done-epoch)*30)),total-1,len(live_opponent)-1);gi=min(int(t*.6),len(all_pos)-1)
                    actual_targets=np.array([data.body(n).xpos.copy() for n in ref.robot_names])
                    packet=dict(run=a.out.name,frame=t,time_s=t/50,source_time_s=done-epoch,total_s=steps/50,opponent_mode=a.opponent_mode,
                                opponent=live_opponent[oi].round(5).tolist(),generated=np.asarray(all_pos[gi])[0].round(5).tolist(),
                                reference_frame=gi,recorded_self_frame=gi,
                                recorded_self=recorded_self[gi].round(5).tolist(),
                                targets=pp[0].round(5).tolist(),target_quats=qq[0].round(5).tolist(),actual_targets=actual_targets.round(5).tolist(),
                                robot=np.column_stack([visual_data.xpos,visual_data.xquat]).round(5).tolist(),
                                starved=starved,preview_short=preview_short,generation_ms=chunks[-1]['prediction_ms'] if chunks else 0,
                                control_ms=(done-begun)*1000,reference_age_ms=max(0,(t/50-(len(ref.pos)-1)/30)*1000),
                                hand_error_cm=(np.linalg.norm(actual_targets[1:]-pp[0,1:],axis=1)*100).tolist(),
                                hand_relative_error_cm=(np.linalg.norm((actual_targets[1:]-actual_targets[0])-(pp[0,1:]-pp[0,0]),axis=1)*100).tolist())
                    try:telemetry.sendto(json.dumps(packet,separators=(',',':')).encode(),('127.0.0.1',a.telemetry_port))
                    except (BlockingIOError,OSError):pass
                # Include telemetry packing in control deadline accounting. Browser/network never waits here.
                cycle_end=time.monotonic()
                ticks[-1]['compute_ms']=(cycle_end-begun)*1000
                ticks[-1]['deadline_miss_ms']=max(0,(cycle_end-deadline-.02)*1000)
                if tracker.failed or not np.isfinite(data.qpos).all() or data.qpos[2]<.3 or tilt>70:
                    failure='fall_or_policy_failure';break
                if t%50==0:print(f'REALTIME {t/50:.1f}s / {steps/50:.1f}s | available {len(ref.sample_pos)/50:.2f}s | hold={starved}',flush=True)
            wall=time.monotonic()-start
            active=ticks[:steps]
            report=dict(protocol='wall-clock opponent release; persistent R2R + ScaleBFM; Unix socket; no IK; mode 2; 50Hz/500Hz MuJoCo; FixedPose + 0.2s settle',
                        upper_actual_state_feedback=False,reference_forcing=bool(cfg.scalebfm.reference_forcing),external_support=False,
                        opponent_mode=a.opponent_mode,intervention_s=a.intervention_s,opponent_delay_ms=a.opponent_delay_ms,
                        delay_ms=a.delay_ms,blend_ms=a.blend_ms,stall_at=a.stall_at,stall_ms=a.stall_ms,total_source_frames=total,
                        failure=failure,simulation_completed=failure is None and len(trace)==steps+25,
                        completed=failure is None and len(trace)==steps+25 and len(ref.pos)==total,
                        received_source_frames=len(ref.pos),producer_done=finished,
                        world_alignment=dict(quat_xyzw=ref.align.as_quat().tolist(),origin=ref.origin.tolist(),source_origin=ref.source_origin.tolist()),
                        actual_realtime_factor=len(trace)*.02/wall,current_target_hold_frames=sum(x['starved'] for x in active),preview_short_frames=sum(x['preview_short'] for x in active),active_frames=len(active),
                        control_compute_ms=stats([x['compute_ms'] for x in ticks]),deadline_misses=sum(x['deadline_miss_ms']>0 for x in ticks),deadline_miss_ms=stats([x['deadline_miss_ms'] for x in ticks]),
                        generation_ms=stats([x['prediction_ms'] for x in chunks]),observation_to_available_ms=stats([x['available_latency_ms'] for x in chunks]),
                        observation_to_first_policy_use_ms=stats([(x['first_use_s']-x['source_observation_s'])*1000 for x in chunks if x['first_use_s'] is not None]),
                        mapping_resampling_chunk_ms=stats(ref.mapping_ms),max_target_step_m=max(x['target_step_m'] for x in ticks),max_joint_step_rad=max(x['joint_step_rad'] for x in ticks),max_tilt_deg=max(x['tilt_deg'] for x in ticks),
                        latency_scope='Observed opponent sample timestamp to chunk receipt / first policy command using it in current or future preview. Not measured physical response latency. Startup excluded.',chunks=chunks,ticks=ticks)
            (a.out/'report.json').write_text(json.dumps(report,indent=2))
            np.savez_compressed(a.out/'trace.npz',qpos=np.array(trace),target_positions=np.array(targets),fps=50)
            np.savez_compressed(a.out/'human.npz',positions=all_pos,rotations=all_rot,actors=first['actors'],fps=30)
            print(json.dumps({k:v for k,v in report.items() if k not in ('chunks','ticks')},indent=2))
        finally:
            if 'telemetry' in locals() and telemetry:telemetry.close()
            if conn is not None:conn.close()
            listener.close()
            if worker is not None:
                if worker.poll() is None:worker.terminate()
                try:worker.wait(timeout=10)
                except subprocess.TimeoutExpired:worker.kill();worker.wait()

if __name__=='__main__':main()
