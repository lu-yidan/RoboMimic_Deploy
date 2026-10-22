"""Recorded-prefix diagnostic: bounded foot search, load confirmation, then ramp.

No live-state replacement after initialization. Actual contact identity/force is
privileged. Online foot references use kinematics, not displaced rod geometry.
This segment is not a full-cycle acceptance result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.climb_cycle import LimbIK
from training.climb_robustness import apply_rung_offsets,ReleaseInterlock
from training.probing_contact_frame_controller import ProbingContactFrameController


def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--search-speed',type=float,default=.02)
    p.add_argument('--maximum-depth',type=float,default=.015)
    p.add_argument('--baseline',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.prefix/'manifest.json').read_text())
    plan=np.load(manifest['config']['plan']);e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2')
    variant=apply_rung_offsets(e,json.loads((a.prefix/'scene_variant.json').read_text())['offsets'])
    saved=np.load(a.prefix/'trajectory.npz');i=int(np.argmin(abs(saved['time_s']-79.)))
    assert abs(saved['time_s'][i]-79.)<1e-7
    e.data.qpos[:]=saved['qpos'][i];e.data.qvel[:]=saved['qvel'][i];e.data.time=79.
    e.initial=plan['qpos'][0].copy();mujoco.mj_forward(e.model,e.data)
    c=ProbingContactFrameController(e,torque_weight=10000.);ik=LimbIK(e)
    guard=ReleaseInterlock(e);guard.expected=['rung_01','rung_01','rung_05','rung_05']
    sources={}
    for rel in ['tools/probe_contact_landing.py','training/probing_contact_frame_controller.py',
                'training/climb_cycle.py','training/climb_robustness.py','training/wall_support_env.py','training/climb_env.py']:
        target=a.out/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
        sources[rel]=hashlib.sha256(target.read_bytes()).hexdigest()
    (a.out/'manifest.json').write_text(json.dumps(dict(prefix=str(a.prefix.resolve()),
        prefix_sha256=hashlib.sha256((a.prefix/'trajectory.npz').read_bytes()).hexdigest(),
        plan_sha256=manifest['plan_sha256'],sources=sources,scene=variant,args=vars(a)|{'prefix':str(a.prefix),'out':str(a.out)},
        contract='79..84 s recorded-prefix diagnostic, 200 Hz commands, no online state writes; exact contact topology and forces; LF search <=15 mm then 250 ms confirmation and 500 ms cap ramp'),indent=2))
    depth=0.;previous=np.zeros(6);mode='waiting';confirmed_at=None;reason='complete'
    states=[e.data.qpos.copy()];velocities=[e.data.qvel.copy()];times=[79.]
    rows=[];qrefs=[];vrefs=[];torques=[];active_rows=[]
    other_peak=self_peak=base_peak=torque_peak=0.;minimum_margin=float('inf')
    while e.data.time<84.-1e-8:
        d=e.data;m=e.model;t=float(d.time)
        j=min(max(np.searchsorted(plan['time'],t,side='right')-1,0),len(plan['time'])-2)
        dt=plan['time'][j+1]-plan['time'][j];u=np.clip((t-plan['time'][j])/dt,0,1)
        q=(1-u)*plan['qpos'][j]+u*plan['qpos'][j+1];v=np.zeros(m.nv)
        v[:3]=(plan['qpos'][j+1,:3]-plan['qpos'][j,:3])/dt
        v[e.vadr]=(plan['qpos'][j+1,e.qadr]-plan['qpos'][j,e.qadr])/dt
        active=plan['active'][j+1].astype(bool).copy()
        guard.sample(t);metrics=e.contact_metrics()
        contacts=[x for x in metrics['contacts'] if x['group']=='left_foot' and x['external_geom']=='rung_01']
        has_contact=bool(contacts)
        if not a.baseline and t>=79.72-1e-8:
            if mode=='waiting':mode='searching'
            if mode=='searching':
                if has_contact:mode='confirming'
                else:depth=min(a.maximum_depth,depth+a.search_speed*.005)
            if mode=='confirming':
                if not has_contact:mode='searching'
                elif guard.history and t-guard.history[0][0]>=.25-1e-7 and all(r[1][0] for r in guard.history):
                    mode='ramping';confirmed_at=t
            if mode=='ramping' and t-confirmed_at>=.5:mode='loaded'
            active[0]=has_contact
            cap=10. if confirmed_at is None else 10.+340.*min(1.,(t-confirmed_at)/.5)
            c.normal_caps={'lf':cap}
            if t>=81.72-1e-8 and mode!='loaded':reason='landing_not_confirmed_before_body_rise';break
        if depth:
            ik.d.qpos[:]=q;mujoco.mj_kinematics(m,ik.d);bid=ik.bodies['lf']
            target=ik.d.xpos[bid].copy();target[2]-=depth;quat=ik.d.xquat[bid].copy()
            changed,errors=ik.pose(q,{'lf':target},orientations={'lf':quat})
            if errors['lf']>.002:reason='online_ik_error';break
            correction=changed[e.qadr[:6]]-q[e.qadr[:6]]
            q=changed;v[e.vadr[:6]]+=(correction-previous)/.005;previous=correction
        try:torque,info=c.command(q,v,active)
        except RuntimeError as exc:reason=str(exc);break
        qrefs.append(q.copy());vrefs.append(v.copy());torques.append(torque.copy());active_rows.append(active.copy())
        rows.append(dict(time_s=t,mode=mode,depth_m=depth,lf_contact=has_contact,lf_force_n=metrics['left_foot_vertical_n'],
            lf_normal_cap_n=c.normal_caps.get('lf',350.),torque_ratio=info['torque_ratio']))
        torque_peak=max(torque_peak,info['torque_ratio']);d.ctrl[e.motors]=torque
        for _ in range(10):
            assert not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
            mujoco.mj_step(m,d);mujoco.mj_kinematics(m,d)
            for ci,contact in enumerate(d.contact):
                ba,bb=int(m.geom_bodyid[contact.geom1]),int(m.geom_bodyid[contact.geom2]);ia=ba in e.robot_bodies;ib=bb in e.robot_bodies
                if not (ia or ib):continue
                force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force);magnitude=float(np.linalg.norm(force[:3]))
                if ia and ib:self_peak=max(self_peak,magnitude)
                elif (ba if ia else bb) not in e.contact_groups:other_peak=max(other_peak,magnitude)
            base_peak=max(base_peak,float(np.linalg.norm(d.qpos[:3]-q[:3])))
            minimum_margin=min(minimum_margin,float(np.min(np.minimum(d.qpos[e.qadr]-m.jnt_range[e.joints,0],m.jnt_range[e.joints,1]-d.qpos[e.qadr]))))
        states.append(d.qpos.copy());velocities.append(d.qvel.copy());times.append(float(d.time))
        if max(other_peak,self_peak)>1 or base_peak>.03 or minimum_margin<0 or np.any(d.warning.number):reason='physical_check_failed';break
    np.savez_compressed(a.out/'trajectory.npz',qpos=states,qvel=velocities,time_s=times,qref=qrefs,vref=vrefs,torques=torques,active=active_rows)
    (a.out/'records.json').write_text(json.dumps(rows))
    summary=dict(reason=reason,time_s=float(e.data.time),mode=mode,depth_m=depth,confirmed_at_s=confirmed_at,
        other_peak_n=other_peak,self_peak_n=self_peak,base_peak_m=base_peak,minimum_native_joint_margin_rad=minimum_margin,
        torque_ratio=torque_peak,success=reason=='complete' and (a.baseline or mode=='loaded'),
        scope='Recorded 79 s prefix; controller solver cache reset; segment only, not full cycle or deployment')
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
