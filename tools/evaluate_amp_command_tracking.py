"""Diagnose the released 60k AMP actor with joystick limits versus direct commands.

Does not change configuration or connect to hardware. Each case starts standing,
settles 2s, receives a command for 10s, and reports the last 5s torso velocity.
"""
import sys,json
from pathlib import Path
from unittest.mock import patch
import numpy as np,mujoco,yaml
import argparse
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
from common.ctrlcomp import StateAndCmd,PolicyOutput
from common.utils import get_gravity_orientation
from policy.amp.Amp import Amp
cfg=yaml.safe_load((root/'policy/amp/config/amp.yaml').read_text());cfg['model_path']='amp_stratified_guidance_60000.onnx'
sim=yaml.safe_load((root/'deploy_mujoco/config/mujoco.yaml').read_text());limit=np.array(sim['tau_limit'])
m=mujoco.MjModel.from_xml_path(str(root/sim['xml_path']));m.opt.timestep=.002
cases={'forward05':[.5,0,0],'forward1':[1,0,0],'backward05':[-.5,0,0],'left05':[0,.5,0],'right05':[0,-.5,0],'yaw_left':[0,0,1],'yaw_right':[0,0,-1]};out={}
for mode in ['joystick','direct']:
 out[mode]={}
 for label,target in cases.items():
  target=np.array(target);d=mujoco.MjData(m);d.qpos[:7]=[0,0,.78,1,0,0,0];d.qpos[7:36]=cfg['default_dof_pos'];mujoco.mj_forward(m,d)
  s=StateAndCmd(29);o=PolicyOutput(29)
  def update():
   s.q=d.qpos[7:36].copy();s.dq=d.qvel[6:35].copy();s.ang_vel=d.qvel[3:6].copy();s.gravity_ori=get_gravity_orientation(d.qpos[3:7])
  update()
  with patch('policy.amp.Amp.yaml.safe_load',return_value=cfg):p=Amp(s,o)
  p.enter();vals=[];commands=[]
  if mode=='direct':p._update_cmd=lambda:None
  torso=m.body('torso_link').id
  for step in range(600):
   cmd=np.zeros(3) if step<100 else target
   if mode=='direct':p._cmd_smooth_val[:]=cmd
   else:
    for j,ran in enumerate(p._range_fast):
     vmax=ran[1] if cmd[j]>=0 else -ran[0]
     s.vel_cmd[j]=np.sign(cmd[j])*min(1.,.05+.95*abs(cmd[j])/vmax) if cmd[j] else 0.
   p.run()
   for _ in range(10):
    d.ctrl[:]=np.clip((o.actions-d.qpos[7:36])*o.kps-d.qvel[6:35]*o.kds,-limit,limit);mujoco.mj_step(m,d)
   update()
   if step>=350:
    vel=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,torso,vel,0)
    rot=d.xmat[torso].reshape(3,3);yaw=np.arctan2(rot[1,0],rot[0,0]);c=np.cos(yaw);ss=np.sin(yaw)
    vals.append([c*vel[3]+ss*vel[4],-ss*vel[3]+c*vel[4],vel[2]])
    commands.append(p._cmd_smooth_val.copy())
  out[mode][label]={'requested':target.tolist(),'policy_command':np.mean(commands,axis=0).tolist(),'actual':np.mean(vals,axis=0).tolist()}
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
