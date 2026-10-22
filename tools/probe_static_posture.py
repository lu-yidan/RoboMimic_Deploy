"""Test a scanned static pose with motor torques; initialization is not recovery."""
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
    p=argparse.ArgumentParser();p.add_argument('--scan',type=Path,required=True)
    p.add_argument('--index',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();manifest=json.loads((a.scan/'manifest.json').read_text())
    records=json.loads((a.scan/'results.json').read_text());record=records[a.index]
    if not record['feasible']:raise ValueError('Static feasibility required')
    run=Path(manifest['source']);a.out.mkdir(parents=True,exist_ok=False)
    e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset()
    apply_rung_offsets(e,json.loads((run/'scene_variant.json').read_text())['offsets'])
    q=np.load(a.scan/'candidates.npz')['qpos'][a.index];active=np.load(run/'failure_state.npz')['active']
    d=e.data;m=e.model;d.qpos[:]=q;d.qvel[:]=0;d.time=0;mujoco.mj_forward(m,d)
    native=e.limits.copy();e.limits=native*manifest['torque_fraction']
    controller=CycleTorqueController(e,torque_weight=10000.)
    states=[d.qpos.copy()];velocities=[d.qvel.copy()];times=[0.];torques=[];forces=[]
    reason='complete';peak=0.;drift=0.;ratio=0.
    for _ in range(400):
        try:torque,info=controller.command(q,np.zeros(m.nv),active)
        except RuntimeError as exc:reason=str(exc);break
        ratio=max(ratio,float(np.max(abs(torque)/np.max(abs(native),axis=1))));d.ctrl[e.motors]=torque
        for _ in range(10):
            assert not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied)
            mujoco.mj_step(m,d);mujoco.mj_kinematics(m,d)
            for ci,c in enumerate(d.contact):
                ba,bb=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
                ia,ib=ba in e.robot_bodies,bb in e.robot_bodies
                if not (ia or ib):continue
                if ia==ib or (ba if ia else bb) not in e.contact_groups:
                    w=np.zeros(6);mujoco.mj_contactForce(m,d,ci,w);peak=max(peak,float(np.linalg.norm(w[:3])))
            if not np.isfinite(d.qpos).all() or np.any(d.warning.number):reason='physics_failure';break
        drift=max(drift,float(np.linalg.norm(d.qpos[:3]-q[:3])))
        metrics=e.contact_metrics();forces.append(dict(time_s=float(d.time),vertical_n=[metrics[g+'_vertical_n'] for g in GROUPS]))
        states.append(d.qpos.copy());velocities.append(d.qvel.copy());times.append(float(d.time));torques.append(torque)
        if drift>.03:reason='base_drift'
        if peak>=1:reason='unintended_contact'
        if reason!='complete':break
    tail=np.array([f['vertical_n'] for f in forces if f['time_s']>1.5])
    supported=bool(len(tail)>=99 and np.all(tail[:,np.asarray(active,dtype=bool)]>10) and np.max(abs(tail[:,0]))<1)
    complete=reason=='complete' and abs(d.time-2)<1e-7
    result=dict(passed=complete and supported and peak<1 and drift<.03,
        complete=complete,reason=reason,time_s=float(d.time),scan=str(a.scan.resolve()),index=a.index,
        candidate=record,torque_peak_ratio_native=ratio,base_drift_peak_m=drift,unintended_contact_peak_n=peak,
        terminal_supports_confirmed=supported,
        terminal_minimum_vertical_n=tail.min(axis=0).tolist() if len(tail) else None,
        candidate_file_sha256=hashlib.sha256((a.scan/'candidates.npz').read_bytes()).hexdigest(),
        scope='Initialized at candidate qpos with zero velocity; 2 s motor-only test. Does not validate a path from the failure state or online recovery.')
    np.savez_compressed(a.out/'trajectory.npz',qpos=states,qvel=velocities,time_s=times,torques=torques)
    (a.out/'force_log.json').write_text(json.dumps(forces));(a.out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['passed','reason','time_s','base_drift_peak_m','terminal_supports_confirmed']}));e.close()


if __name__=='__main__':main()
