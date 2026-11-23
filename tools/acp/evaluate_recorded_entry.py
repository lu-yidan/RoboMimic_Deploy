#!/usr/bin/env python3
"""Replay recorded standing poses through the real FixedPose/FSM/ACP guard in MuJoCo.

No transport constructors run. Unrecorded root translation/velocity and foot
forces are explicitly simulation assumptions, never called measured telemetry.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.deployment.replay_branches import NoDDS
_no_dds = NoDDS()
sys.meta_path.insert(0,_no_dds)
from bridge.python.policy_runtime import PolicyRuntime
from common.ctrlcomp import StateAndCmd,PolicyOutput
from common.deployment_config import load_acp_config
from common.utils import FSMCommand,FSMStateName
from FSM.FSM import FSM,FSMMode
from policy.acp.ACP import ACP
from policy.acp.observations import rotation,yaw
from policy.acp.terrain import MujocoTerrain
from policy.fixedpose.FixedPose import FixedPose
from policy.passive.PassiveMode import PassiveMode
from tools.acp.evaluate_mujoco import final_hold_result,final_pose_metrics
sys.meta_path.remove(_no_dds)

PINNED_POLICY_SHA256 = '2a85ea0312bbdff8b681dabe659c8f9c68b1d55221c4f9ec41284dd674e310e8'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_log(meta):
    m = json.loads(meta.read_text())
    path = meta.with_suffix('.bin')
    if m.get('dtype') != 'float32' or path.stat().st_size % (4*m['record_dim']):
        raise ValueError('Incomplete or unsupported recorded log: '+str(path))
    a = np.memmap(path,dtype=np.float32,mode='r').reshape(-1,m['record_dim'])
    values,offset = {},0
    for key,width in m['fields'].items():
        values[key],offset = a[:,offset:offset+width],offset+width
    if offset != m['record_dim']:raise ValueError('Logger width mismatch')
    return m,values


def standing_windows(meta, minimum_frames=50):
    m,d = read_log(meta)
    if not str(m.get('git_commit','')).startswith('6a25875'):return []
    q = d['pelvis_quat_w']
    tilt = np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2),-1,1))
    dq = np.sqrt((d['dq']**2).mean(1))
    gyro = np.linalg.norm(d['ang_vel'],axis=1)
    good = ((d['executed_fsm_state'][:,0]==16)&(tilt<.15)&(dq<.08)&(gyro<.15)
            &(np.linalg.norm(d['vel_cmd'],axis=1)<.02))
    good &= np.isfinite(np.c_[d['q'],d['dq'],q,d['ang_vel']]).all(1)
    good &= np.abs(np.linalg.norm(q,axis=1)-1)<.01
    good &= ~np.r_[False,(np.diff(d['time_s'][:,0])<=0)|(np.diff(d['time_s'][:,0])>.05)]
    edges = np.flatnonzero(np.diff(np.r_[False,good,False]))
    windows = []
    for start,stop in zip(edges[::2],edges[1::2]):
        if stop-start<minimum_frames or np.ptp(d['q'][start:stop],axis=0).max()>.06:continue
        windows.append(dict(source=str(meta),start=int(start),stop=int(stop),frames=int(stop-start),
            dq_rms_mean=float(dq[start:stop].mean()),gyro_max=float(gyro[start:stop].max()),
            tilt_max=float(tilt[start:stop].max()),q_span_max=float(np.ptp(d['q'][start:stop],axis=0).max())))
    return windows


def select_poses(logs, per_window=3):
    windows = []
    for meta in sorted(logs.glob('*/*.json')):
        m = json.loads(meta.read_text())
        if not str(m.get('git_commit','')).startswith('6a25875') or not meta.with_suffix('.bin').exists():continue
        if meta.with_suffix('.bin').stat().st_size==0:continue
        windows.extend(standing_windows(meta))
    windows.sort(key=lambda w:(-w['frames'],w['dq_rms_mean']))
    chosen,sources = [],set()
    for window in windows:
        if window['source'] in sources:continue
        chosen.append(window);sources.add(window['source'])
        if len(chosen)==2:break
    if len(chosen)<2:raise ValueError('Need two complete, qualified recorded standing intervals')
    poses = []
    for window in chosen:
        path = Path(window['source']);m,d = read_log(path)
        for frame in np.linspace(window['start'],window['stop']-1,per_window+2,dtype=int)[1:-1]:
            pose = {key:np.asarray(d[key][frame]).copy() for key in ('q','dq','pelvis_quat_w','ang_vel','actions','kps','kds')}
            pose['provenance'] = dict(metadata=str(path),metadata_sha256=sha(path),source_bin_sha256=sha(path.with_suffix('.bin')),
                frame=int(frame),time_s=float(d['time_s'][frame,0]),git_commit=m['git_commit'],git_dirty=m['git_dirty'],
                joint_order=m['joint_order'],standing_window=window)
            poses.append(pose)
    return poses,windows


def collision_foot_geoms(model):
    result = []
    for side in ('left','right'):
        body = model.body(side+'_ankle_roll_link').id
        ids = np.flatnonzero((model.geom_bodyid==body)&((model.geom_contype|model.geom_conaffinity)!=0))
        if not len(ids):raise ValueError('No collision foot geometry: '+side)
        result.append(ids)
    return result


def lowest_geom_z(model,data,geom):
    kind,size = model.geom_type[geom],model.geom_size[geom]
    if kind==mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom];first,count = model.mesh_vertadr[mesh],model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[first:first+count]
        return float((vertices@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom])[:,2].min())
    if kind==mujoco.mjtGeom.mjGEOM_SPHERE:return float(data.geom_xpos[geom,2]-size[0])
    if kind==mujoco.mjtGeom.mjGEOM_BOX:
        return float(data.geom_xpos[geom,2]-np.abs(data.geom_xmat[geom].reshape(3,3)[2])@size)
    if kind==mujoco.mjtGeom.mjGEOM_CAPSULE:
        return float(data.geom_xpos[geom,2]-abs(data.geom_xmat[geom].reshape(3,3)[2,2])*size[1]-size[0])
    raise ValueError('Unsupported foot geometry')


def foot_sample(model,data,feet):
    loads = np.zeros(2)
    wrench = np.zeros(6)
    sets = [set(ids.tolist()) for ids in feet]
    for index in range(data.ncon):
        contact = data.contact[index]
        for side,ids in enumerate(sets):
            other = contact.geom2 if contact.geom1 in ids else (contact.geom1 if contact.geom2 in ids else -1)
            if other>=0 and model.body_weldid[model.geom_bodyid[other]]==0:
                mujoco.mj_contactForce(model,data,index,wrench)
                loads[side] += max(0.,wrench[0])
    heights = np.array([min(lowest_geom_z(model,data,int(g)) for g in ids) for ids in feet])
    return loads,heights


def initialize_recorded_pose(model,data,pose):
    """Preserve measured joint/IMU state; fit only unmeasured root translation."""
    data.qpos[:3] = [0.,0.,0.]
    data.qpos[3:7],data.qpos[7:] = pose['pelvis_quat_w'],pose['q']
    data.qvel[:3],data.qvel[3:6],data.qvel[6:] = 0.,pose['ang_vel'],pose['dq']
    feet = collision_foot_geoms(model)
    mujoco.mj_forward(model,data)
    data.qpos[2] = .002-min(lowest_geom_z(model,data,int(g)) for ids in feet for g in ids)
    mujoco.mj_forward(model,data)
    return data.qpos[:7].copy(),feet


def run_case(bundle,pose,scenario,support,seconds=15.):
    model = mujoco.MjModel.from_xml_path(str(ROOT/'g1_description/scene.xml'))
    model.opt.timestep = .005
    data = mujoco.MjData(model)
    names = [model.joint(int(model.actuator_trnid[i,0])).name for i in range(model.nu)]
    recorded_names = [n if n.endswith('_joint') else n+'_joint' for n in pose['provenance']['joint_order']]
    if recorded_names != names:raise ValueError('Recorded 29-joint order differs from simulation motor order')
    state,out = StateAndCmd(29),PolicyOutput(29)
    initial_heading = yaw(pose['pelvis_quat_w'])
    offset,heading = {'stop':([0.,0.],initial_heading),'forward':([np.cos(initial_heading),np.sin(initial_heading)],initial_heading),
                     'turn':([-np.sin(initial_heading),np.cos(initial_heading)],initial_heading+np.pi/2)}[scenario]
    config = dict(load_acp_config(),enabled=True,model_dir=str(bundle),target_position=offset,target_heading=heading)
    policy = ACP(state,out,config,names)
    fixed = FixedPose(state,out);fixed.set_runtime_target(policy.default,policy.kps,policy.kds)
    fsm = FSM.__new__(FSM)
    fsm.state_cmd,fsm.policy_output,fsm.FSMmode,fsm.sim_counter = state,out,FSMMode.NORMAL,0
    fsm.acp_policy,fsm.fixed_pose_1 = policy,fixed
    fsm.passive_mode = PassiveMode(state,out)
    fsm.scalebfm_policy,fsm.php_policy = None,None
    fsm.cur_policy = fixed
    runtime = PolicyRuntime.__new__(PolicyRuntime)
    runtime.state_cmd,runtime.policy_output,runtime.FSM_controller = state,out,fsm
    runtime._logger,runtime._cmd_seq,runtime._acp_effort_limits = None,0,policy.effort_limits
    root,feet = initialize_recorded_pose(model,data,pose)
    state.q[:] = pose['q'];fixed.enter()
    terrain = MujocoTerrain(model)
    records,trajectory = [],[]
    error,entry,first = None,None,None
    previous = pose['actions'].copy()
    raw,bounded,limited,torque,clipped = [np.zeros(29) for _ in range(5)]
    obs = np.full(939,np.nan)
    for tick in range(round((2.+seconds)/.005)):
        if tick%4==0:
            state.q[:],state.dq[:] = data.qpos[7:],data.qvel[6:]
            terrain.capture(data,state)
            if tick==400:
                reason = policy.entry_error()
                state.skill_cmd = FSMCommand.CMD_ACP
            fsm.run()
            raw = out.actions.copy()
            if tick==400 and fsm.cur_policy is not policy:
                error = 'ACP entry rejected: '+str(reason);break
            if runtime._guard_acp_output(was_active=entry is not None):
                error = policy.last_error or 'Runtime guard requested damping';break
            bounded = out.actions.copy();limited = np.abs(raw-bounded)>1e-7
            loads,heights = foot_sample(model,data,feet)
            if fsm.cur_policy is policy and entry is None:
                entry = float(data.time)
                first = dict(time_s=entry,q_before=state.q.tolist(),dq_before=state.dq.tolist(),pelvis_pose=data.qpos[:7].tolist(),
                    previous_fixedpose_target=previous.tolist(),raw_target=raw.tolist(),bounded_target=bounded.tolist(),
                    raw_target_jump=(raw-previous).tolist(),bounded_target_jump=(bounded-previous).tolist(),
                    max_raw_target_jump=float(np.abs(raw-previous).max()),max_bounded_target_jump=float(np.abs(bounded-previous).max()),
                    limited_joint_indices=np.flatnonzero(limited).tolist(),kp=out.kps.tolist(),kd=out.kds.tolist(),
                    foot_normal_force=loads.tolist(),foot_min_z=heights.tolist(),support_released_before_first_acp_physics_step=bool(support))
            obs = policy.last_observation[0].copy() if fsm.cur_policy is policy else np.full(939,np.nan)
            records.append(dict(time_s=float(data.time),phase=1 if entry is not None else 0,support=bool(support and entry is None),
                qpos=data.qpos.copy(),qvel=data.qvel.copy(),raw_target=raw.copy(),q_des=bounded.copy(),kp=out.kps.copy(),kd=out.kds.copy(),
                previous_q_des=previous.copy(),target_limited=limited.copy(),foot_normal_force=loads,foot_min_z=heights,
                pd_torque=np.full(29,np.nan),applied_torque=np.full(29,np.nan),effort_clipped=np.zeros(29,dtype=bool),
                planner_obs=obs,tilt=float(np.arccos(np.clip(rotation(data.qpos[3:7])[2,2],-1,1)))))
            previous = bounded.copy()
            runtime._cmd_seq += 1
        torque = out.kps*(out.actions-data.qpos[7:])-out.kds*data.qvel[6:]
        clipped = np.clip(torque,-policy.effort_limits,policy.effort_limits)
        if not np.isfinite(np.r_[torque,clipped]).all():error='Nonfinite control';break
        data.ctrl[:] = clipped
        if tick%4==0:
            records[-1].update(pd_torque=torque.copy(),applied_torque=clipped.copy(),effort_clipped=np.abs(torque-clipped)>1e-6)
        mujoco.mj_step(model,data)
        if support and entry is None:
            data.qpos[:7],data.qvel[:6],data.qacc_warmstart[:6] = root,0.,0.
            mujoco.mj_forward(model,data)
        if not np.isfinite(np.r_[data.qpos,data.qvel]).all() or data.qpos[2]<.3:
            error='Robot fell or nonfinite state';break
        if entry is not None and (tick+1)%4==0:
            trajectory.append(np.r_[data.time,data.qpos[:3],data.qvel[:3],yaw(data.qpos[3:7]),rotation(data.qpos[3:7])[2,2]])
    arrays = {key:np.asarray([row[key] for row in records]) for key in records[0]} if records else {}
    hold = final_hold_result(trajectory,offset,heading)
    result = dict(scenario=scenario,support_during_fixedpose=bool(support),support_during_acp=False,source=pose['provenance'],
        initial_root_pose=root.tolist(),initial_root_linear_velocity_assumed=[0.,0.,0.],
        root_height_source='simulation foot collision minimum fitted 2mm above plane z=0; not recorded localization',
        foot_forces_source='MuJoCo contacts, not measured real forces',
        entry_time_s=entry,first_acp_frame=first,error=error,sim_seconds=float(data.time),unassisted_acp_seconds=len(trajectory)*.02,
        max_tilt_after_entry=float(arrays['tilt'][arrays['phase']==1].max()) if entry is not None else None,
        target_limited_frames=int(arrays['target_limited'].any(1).sum()) if arrays else 0,
        effort_clipped_frames=int(arrays['effort_clipped'].any(1).sum()) if arrays else 0,
        **final_pose_metrics(data.qpos,offset,heading),**hold,
        passed=error is None and len(trajectory)>=round(seconds/.02) and hold['held_all_criteria_last_second'])
    return result,arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--logs-root',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--poses-per-window',type=int,default=3)
    args = p.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):p.error('Output directory must be empty')
    if not 1<=args.poses_per_window<=10:p.error('poses-per-window must be 1..10')
    bundle = args.bundle.resolve(strict=True)
    if sha(bundle/'policy.onnx')!=PINNED_POLICY_SHA256:raise ValueError('Requires the fixed bundle_stage3_1000 bytes')
    poses,windows = select_poses(args.logs_root,args.poses_per_window)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'selected_poses.json').write_text(json.dumps([dict(provenance=x['provenance'],**{k:v.tolist() for k,v in x.items() if k!='provenance'}) for x in poses],indent=2)+'\n')
    results = []
    for support in (False,True):
        for index,pose in enumerate(poses):
            for scenario in ('stop','forward','turn'):
                row,arrays = run_case(bundle,pose,scenario,support)
                name = '{}_pose{}_{}'.format('supported' if support else 'free',index,scenario)
                np.savez_compressed(args.output_dir/(name+'.npz'),**arrays)
                row['trace_file'] = name+'.npz'
                row['trace_sha256'] = sha(args.output_dir/(name+'.npz'))
                results.append(row)
                (args.output_dir/'results.json').write_text(json.dumps(dict(policy_sha256=PINNED_POLICY_SHA256,windows=windows,results=results,
                    manifest_sha256=sha(bundle/'manifest.json'),
                    scene_sha256=sha(ROOT/'g1_description/scene.xml'),
                    robot_model_sha256=sha(ROOT/'g1_description/g1_29dof_rev_1_0.xml'),
                    transport='No DDS constructors/publishers; actual FSM.run and PolicyRuntime._guard_acp_output in MuJoCo',
                    assumptions=['Root XY canonicalized to zero','Root Z fitted using simulation foot geometry','Unmeasured root linear velocity assumed zero',
                                 'Recorded quaternion, angular velocity, q and dq retained','Support holds initial recorded root pose until ACP first physics step']),indent=2,allow_nan=False)+'\n')
                print(json.dumps({k:row[k] for k in ('scenario','support_during_fixedpose','entry_time_s','error','passed','unassisted_acp_seconds')}),flush=True)
    if NoDDS.attempts:raise RuntimeError('Transport constructor attempted')


if __name__=='__main__':main()
