"""Exercise the actual Amp controller in the existing deployment MuJoCo scene."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
import yaml
from common.ctrlcomp import StateAndCmd,PolicyOutput
from common.utils import get_gravity_orientation
from policy.amp.Amp import Amp


def run_case(model, cfg, sample, seconds, profile):
    data=mujoco.MjData(model)
    data.qpos[:7]=sample['root']
    names=sample['joint_names']
    for name,value in zip(names,sample['joints']):data.qpos[model.joint(name).qposadr[0]]=value
    mujoco.mj_forward(model,data)
    initial_penetration=min([float(c.dist) for c in data.contact]+[0.])
    state=StateAndCmd(29);out=PolicyOutput(29)
    def state_update():
        state.q=data.qpos[7:36].astype(np.float32).copy()
        state.dq=data.qvel[6:35].astype(np.float32).copy()
        state.ang_vel=data.qvel[3:6].astype(np.float32).copy()
        state.gravity_ori=get_gravity_orientation(data.qpos[3:7]).astype(np.float32)
    state_update()
    policy=Amp(state,out,profile=profile)
    policy.enter();hold=0;first=None;tail=[];vx_error=[];max_tau=0.;finite=True
    sim_cfg=yaml.safe_load((ROOT/'deploy_mujoco/config/mujoco.yaml').read_text())
    tau_limit=np.asarray(sim_cfg['tau_limit'])
    model.opt.timestep=float(sim_cfg['simulation_dt']);decimation=int(sim_cfg['control_decimation'])
    for step in range(int(seconds/.02)):
        target=[0.,.5,1.,2.][min(step//250,3)] if sample['label']=='walking' else 0.
        state.vel_cmd[0]=.05+.95*target/cfg['cmd_range_fast']['lin_vel_x'][1] if target else 0.
        policy.run()
        for _ in range(decimation):
            tau=(out.actions-data.qpos[7:36])*out.kps-data.qvel[6:35]*out.kds
            data.ctrl[:]=np.clip(tau,-tau_limit,tau_limit)
            max_tau=max(max_tau,float(np.abs(data.ctrl).max()))
            mujoco.mj_step(model,data)
        state_update()
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():finite=False;break
        stable=data.qpos[2]>.65 and state.gravity_ori[2]<-.9
        hold=hold+1 if stable else 0
        if hold>=50 and first is None:first=(step+1)*.02
        if step>=int(seconds/.02)-250:tail.append(float(stable))
        vx_error.append(abs(float(data.qvel[0])-target))
    return {'label':sample['label'],'success_1s':first is not None,'first_success_s':first,
        'last5s_upright_percent':100*float(np.mean(tail)) if tail else None,
        'finite':finite,'final_root_height':float(data.qpos[2]),'max_abs_torque':max_tau,
        'initial_min_contact_distance':initial_penetration,
        'vx_mae_by_5s':[float(np.mean(vx_error[i:i+250])) for i in range(0,len(vx_error),250)] if sample['label']=='walking' else None}


def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',default='stratified_guidance_60000')
    p.add_argument('--xml',default='g1_description/scene.xml');p.add_argument('--seconds',type=float,default=20.)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    from common.deployment_config import resolve_amp_config
    cfg, _ = resolve_amp_config(yaml.safe_load((ROOT/'policy/amp/config/amp.yaml').read_text()), args.profile)
    samples=json.loads((ROOT/'tests/fixtures/amp_guidance_poses.json').read_text())
    model=mujoco.MjModel.from_xml_path(str(ROOT/args.xml))
    manifest=json.loads((ROOT/'policy/amp/model/amp_stratified_guidance_60000.manifest.json').read_text())
    assert [model.joint(int(model.actuator_trnid[i,0])).name for i in range(model.nu)]==manifest['deployment_motor_names']
    report={'model':cfg['model_path'], 'profile':args.profile,'xml':args.xml,'duration_s':args.seconds,'control_hz':50,
        'protocol':'Existing Amp.py controller and deployment PD/torque limits; no resets or pushes; fixed first held-out pose per direction, plus standing velocity ramp.',
        'scope':'Five nominal examples, not a statistical benchmark or hardware test.',
        'cases':[run_case(model,cfg,s,args.seconds,args.profile) for s in samples]}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
