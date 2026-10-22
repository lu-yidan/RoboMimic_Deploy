"""Offline failure-state experiment: can bounded torques hold the current pose?

This is a restart probe, not an online recovery demonstration. It retains only
scheduled supports with >10 N instantaneous measured load; it adds no contact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.climb_cycle import CycleTorqueController
from training.climb_robustness import apply_rung_offsets, GROUPS


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--brake-seconds',type=float,default=0.)
    a=p.parse_args()
    if not np.isfinite(a.brake_seconds) or not 0<=a.brake_seconds<=.5:raise ValueError('Brake duration must be 0..0.5 s')
    a.out.mkdir(parents=True,exist_ok=False)
    e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset()
    source=a.run/'failure_state.npz';f=np.load(source)
    offsets=json.loads((a.run/'scene_variant.json').read_text())['offsets']
    apply_rung_offsets(e,offsets);d=e.data;m=e.model
    d.qpos[:]=f['qpos'];d.qvel[:]=f['qvel'];d.time=float(f['time_s']);mujoco.mj_forward(m,d)
    initial=d.qpos.copy();initial_velocity=d.qvel.copy();start=float(d.time)
    metrics=e.contact_metrics();loaded=np.array([metrics[g+'_vertical_n']>10 for g in GROUPS])
    active=f['active'].astype(bool)&loaded
    controller=CycleTorqueController(e,torque_weight=10000.)
    states=[d.qpos.copy()];velocities=[d.qvel.copy()];times=[float(d.time)];commands=[];records=[]
    reason='complete';peak=0.;ratio=0.;min_forces=np.full(4,np.inf)
    for step in range(100):
        target=initial.copy();velocity=np.zeros(m.nv)
        if a.brake_seconds:
            elapsed=min(float(d.time-start),a.brake_seconds)
            mujoco.mj_integratePos(m,target,initial_velocity,elapsed-elapsed*elapsed/(2*a.brake_seconds))
            velocity=initial_velocity*(1-elapsed/a.brake_seconds)
        try:torque,info=controller.command(target,velocity,active)
        except RuntimeError as ex:reason=str(ex);break
        ratio=max(ratio,info['torque_ratio']);d.ctrl[e.motors]=torque
        for _ in range(10):
            if np.any(d.qfrc_applied) or np.any(d.xfrc_applied):raise AssertionError('External force forbidden')
            mujoco.mj_step(m,d);mujoco.mj_kinematics(m,d)
            for ci,c in enumerate(d.contact):
                ba,bb=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
                ia,ib=ba in e.robot_bodies,bb in e.robot_bodies
                if not (ia or ib):continue
                if ia and ib or (ba if ia else bb) not in e.contact_groups:
                    wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
                    peak=max(peak,float(np.linalg.norm(wrench[:3])))
            if not np.isfinite(d.qpos).all() or np.any(d.warning.number):reason='physics_failure';break
        force=e.contact_metrics();min_forces=np.minimum(min_forces,[force[g+'_vertical_n'] for g in GROUPS])
        records.append(dict(time_s=float(d.time),forces=[force[g+'_vertical_n'] for g in GROUPS]))
        commands.append(torque);states.append(d.qpos.copy());velocities.append(d.qvel.copy());times.append(float(d.time))
        if np.linalg.norm(d.qpos[:3]-initial[:3])>.03:reason='base_drift';break
        if reason!='complete':break
    drift=max(float(np.linalg.norm(q[:3]-initial[:3])) for q in states)
    complete=reason=='complete' and abs(d.time-start-.5)<1e-7
    result=dict(scope='Offline current-pose hold from recorded failure state; not online recovery',
                source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                offsets=offsets,brake_seconds=a.brake_seconds,active=active.tolist(),initial_loaded=loaded.tolist(),start_s=start,
                duration_s=float(d.time-start),reason=reason,complete=complete,
                no_unintended_contact=peak<1.,unintended_contact_peak_n=peak,torque_peak_ratio=ratio,
                base_drift_peak_m=drift,minimum_vertical_n=[float(x) if np.isfinite(x) else None for x in min_forces],
                all_retained_loaded=bool(records and np.all(min_forces[active]>10.)))
    (a.out/'summary.json').write_text(json.dumps(result,indent=2))
    (a.out/'force_log.json').write_text(json.dumps(records))
    np.savez_compressed(a.out/'trajectory.npz',qpos=states,qvel=velocities,time_s=times,torques=commands)
    print(json.dumps(result,indent=2));e.close()


if __name__=='__main__':main()
