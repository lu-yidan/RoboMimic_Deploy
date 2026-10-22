"""Replay failed frozen-policy trajectories, then inspect same-state allocation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from training.contact_load_transfer_residual_env import LoadTransferResidualEnv
from training.biased_contact_frame_controller import BiasedContactFrameController
from tools.diagnose_axis_allocation import feasibility

def trial(root,out,angle):
    run=root/f'ppo__normal_{angle:+g}deg';manifest=json.loads((run/'manifest.json').read_text());case='row01_z_-10mm'
    env=LoadTransferResidualEnv(manifest['suite'],[case]);obs,_=env.reset(seed=20261017,options={'case':case})
    env.controller=BiasedContactFrameController(env.env,torque_weight=10000.,normal_bias_y_deg=angle)
    model=PPO.load(run/'policy.zip',device='cpu');saved={};original=env.controller.command
    def command(q,v,active):
        try:return original(q,v,active)
        except RuntimeError:
            saved.update(qref=q.copy(),vref=v.copy(),active=np.asarray(active).copy());raise
    env.controller.command=command;e=env.env;states=[e.data.qpos.copy()];velocities=[e.data.qvel.copy()]
    for _ in range(env.max_steps):
        action=model.predict(obs,deterministic=True)[0];obs,_,term,trunc,info=env.step(action)
        states.append(e.data.qpos.copy());velocities.append(e.data.qvel.copy())
        if term or trunc:break
    old=dict(np.load(run/(case+'.npz')));qd=float(np.max(abs(np.asarray(states)-old['qpos'])));vd=float(np.max(abs(np.asarray(velocities)-old['qvel'])))
    assert max(qd,vd)<1e-9 and saved
    folder=out/f'normal_{angle:+g}deg';folder.mkdir()
    np.savez_compressed(folder/'failure_state.npz',qpos=e.data.qpos.copy(),qvel=e.data.qvel.copy(),time_s=float(e.data.time),**saved)
    gaps={}
    for bodyname,rodname in [('hook','rung_05'),('right_hook','rung_05'),('right_ankle_roll_link','rung_01')]:
        body=e.model.body(bodyname).id;rod=e.model.geom(rodname).id
        geoms=[g for g in range(e.model.ngeom) if e.model.geom_bodyid[g]==body and e.model.geom_contype[g]]
        gaps[bodyname]=min(float(mujoco.mj_geomDistance(e.model,e.data,g,rod,.1,None)) for g in geoms)
    captured={}
    class Capture:
        def setup(self,**kwargs):captured.update(kwargs);raise RuntimeError('Captured allocation')
    controller=BiasedContactFrameController(e,torque_weight=10000.,normal_bias_y_deg=angle)
    with patch('training.biased_contact_frame_controller.osqp.OSQP',Capture):
        try:controller.command(saved['qref'],saved['vref'],saved['active'])
        except RuntimeError as exc:capture_reason=str(exc)
    checks={}
    if captured:
        A=captured['A'].toarray();lo=captured['l'];hi=captured['u'];checks['all_constraints'],_=feasibility(A,lo,hi)
        keep=np.r_[np.arange(6),np.arange(6+len(e.vadr),len(A))];checks['without_motor_bounds'],_=feasibility(A[keep],lo[keep],hi[keep])
        names=[e.model.joint(j).name for j in e.joints]
        wrist=[6+i for i,name in enumerate(names) if 'wrist_pitch' in name or 'wrist_yaw' in name]
        keep=np.setdiff1d(np.arange(len(A)),wrist);checks['without_wrist_pitch_yaw_bounds'],_=feasibility(A[keep],lo[keep],hi[keep])
        np.savez_compressed(folder/'allocation.npz',A=A,lower=lo,upper=hi)
    result=dict(angle=angle,case=case,time_s=float(e.data.time),reason=info['reason'],qpos_replay_difference=qd,qvel_replay_difference=vd,minimum_geom_gaps_m=gaps,capture_reason=capture_reason,linear_feasibility=checks,trajectory_sha256=hashlib.sha256((run/(case+'.npz')).read_bytes()).hexdigest(),scope='Exact-state replay; signed geometric distances at failure. LP constraint removals are offline diagnosis, never used for control. No proof of dynamic recovery.')
    (folder/'summary.json').write_text(json.dumps(result,indent=2));env.close();return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    results=[trial(ROOT/'outputs/untrained_geometry_comparison_v1',a.out,angle) for angle in [-3.,3.]]
    (a.out/'summary.json').write_text(json.dumps(dict(results=results,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2));print(json.dumps(results,indent=2))

if __name__=='__main__':main()
