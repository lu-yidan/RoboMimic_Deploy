"""CPU MuJoCo validation using the actual SmpRecovery deployment controller.
No DDS, robot connection, joystick, FSM transitions, or hardware actions.
"""
import argparse,json,os,sys,hashlib,copy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def make_model():
 import mujoco
 tree=ET.parse(ROOT/'g1_description/g1_liao.xml');r=tree.getroot()
 r.find('compiler').set('meshdir',str(ROOT/'g1_description/meshes'))
 ET.SubElement(r.find(".//body[@name='torso_link']"),'site',name='audit_head',pos='0 0 .43',size='.005',rgba='0 0 0 0')
 m=mujoco.MjModel.from_xml_string(ET.tostring(r,encoding='unicode'));m.opt.timestep=.002
 return m

def build_policy(profile):
 import onnxruntime as ort
 from unittest.mock import patch
 from common.ctrlcomp import StateAndCmd,PolicyOutput
 from policy.smp_recovery.SmpRecovery import SmpRecovery
 options=ort.SessionOptions
 def single_thread():
  o=options();o.intra_op_num_threads=1;o.inter_op_num_threads=1;return o
 os.environ['SMP_RECOVERY_PROFILE']=profile
 state=StateAndCmd(29);output=PolicyOutput(29)
 with patch.object(ort,'SessionOptions',single_thread):policy=SmpRecovery(state,output)
 return policy,state,output

def run_case(case,m,policy,state,output,seconds=20,record=False):
 import mujoco
 from common.utils import get_gravity_orientation
 d=mujoco.MjData(m);d.qpos[:36]=case['qpos'];d.qvel[:35]=case.get('qvel',np.zeros(35));mujoco.mj_forward(m,d)
 robot_geoms={g.get('name') for g in ET.parse(ROOT/'g1_description/g1_liao.xml').getroot().find(".//body[@name='pelvis']").iter('geom') if g.get('class')=='collision'}
 head=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'audit_head');floor=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'floor')
 feet=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in ['left_ankle_roll_link','right_ankle_roll_link']]
 names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,int(j)) for j in m.actuator_trnid[:,0]]
 assert [m.jnt_qposadr[int(j)] for j in m.actuator_trnid[:,0]]==list(range(7,36))
 knees=[names.index('left_knee_joint'),names.index('right_knee_joint')]
 def sync():
  state.q=d.qpos[7:36].astype(np.float32).copy();state.dq=d.qvel[6:35].astype(np.float32).copy()
  state.root_ang_vel_b=d.qvel[3:6].astype(np.float32).copy();state.gravity_ori=get_gravity_orientation(d.qpos[3:7]).astype(np.float32)
 sync();policy.enter();target=state.q.copy()
 jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);contact=np.zeros(6)
 peak_tau=np.zeros(29);peak_dq=np.zeros(29);peak_power=np.zeros(29);peak_head_v=0.;peak_head_force=0.;hold=0;best=0;first=-1.;maxheight=0.;states=[];obs=[];checks=np.zeros(10);ncheck=0
 initial_min=min([c.dist for c in d.contact],default=0.);unsafe=None
 for step in range(round(seconds/.002)):
  tau=np.clip((target-d.qpos[7:36])*policy.kps-d.qvel[6:35]*policy.kds,-policy.tau_limit,policy.tau_limit)
  d.ctrl[:]=tau;mujoco.mj_step(m,d)
  if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():unsafe='nonfinite';break
  qtau=d.qfrc_actuator[6:35];dq=d.qvel[6:35]
  peak_tau=np.maximum(peak_tau,np.abs(qtau));peak_dq=np.maximum(peak_dq,np.abs(dq));peak_power=np.maximum(peak_power,np.abs(qtau*dq))
  mujoco.mj_jacSite(m,d,jp,jr,head);peak_head_v=max(peak_head_v,abs(float((jp@d.qvel)[2])))
  # Ground contacts only, vertical force per foot / non-foot as in validation.
  loads=np.zeros(2);other=0.;headforce=0.
  for ci,c in enumerate(d.contact):
   if c.geom1!=floor and c.geom2!=floor:continue
   g=int(c.geom2 if c.geom1==floor else c.geom1);name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or ''
   if name not in robot_geoms:continue
   mujoco.mj_contactForce(m,d,ci,contact);world=c.frame.reshape(3,3).T@contact[:3];fz=abs(world[2])
   if name.startswith('left_foot'):loads[0]+=fz
   elif name.startswith('right_foot'):loads[1]+=fz
   else:other+=fz
   if 'head' in name:headforce+=np.linalg.norm(world)
  peak_head_force=max(peak_head_force,float(headforce))
  if (step+1)%10:continue
  sync();z=float(d.site_xpos[head,2]);upr=float(-state.gravity_ori[2]);maxheight=max(maxheight,z)
  if first<0 and z>=1.15 and upr>=.93:first=(step+1)*.002
  speeds=[]
  for b in feet:
   mujoco.mj_jacBody(m,d,jp,jr,b);speeds.append(np.linalg.norm(jp@d.qvel))
  width=np.linalg.norm(d.xpos[feet[0],:2]-d.xpos[feet[1],:2])
  conditions=np.array([z>=1.15,upr>=.93,np.abs(state.q[knees]).max()<.8,np.linalg.norm(d.qvel[:3])<.15,np.linalg.norm(d.qvel[3:6])<.3,np.sqrt(np.mean(state.dq**2))<.5,max(speeds)<.1,min(loads)>20,other<20,.12<width<.45])
  hold=hold+.02 if conditions.all() else 0.;best=max(best,hold)
  if step>=5000:checks+=~conditions;ncheck+=1
  obs_input=policy._build_obs().copy();policy.run();target=output.actions.copy()
  if record:
   states.append(np.r_[d.qpos.copy(),d.qvel.copy(),z,upr,hold,loads,other,width]);obs.append(obs_input)
 result={'name':case['name'],'direction':case['direction'],'source':case.get('source','procedural'),'success_10s':best>=10-1e-6,'best_hold_s':best,'first_upright_s':first,'max_head_height':maxheight,'peak_tau':peak_tau.tolist(),'peak_dq':peak_dq.tolist(),'peak_power':peak_power.tolist(),'head_vz_peak':peak_head_v,'head_force_peak':peak_head_force,'initial_min_contact':initial_min,'failure_last10s':(checks/max(ncheck,1)).tolist(),'unstable':unsafe,'joint_names':names}
 return result,np.array(states),np.array(obs)

def worker(args):
 cases,profile,out=args;out=Path(out);m=make_model();policy,state,output=build_policy(profile);results=[]
 for case in cases:
  rec=case.get('record',False);r,trace,obs=run_case(case,m,policy,state,output,record=rec);results.append(r)
  if rec:np.savez_compressed(out/(case['name']+'.npz'),state=trace,obs=obs,qpos_initial=np.array(case['qpos']),qvel_initial=np.array(case.get('qvel',np.zeros(35))),nq=m.nq,nv=m.nv)
  (out/(case['name']+'.json')).write_text(json.dumps(r,indent=2));print(case['name'],r['success_10s'],round(r['first_upright_s'],2),flush=True)
 return results

def main():
 p=argparse.ArgumentParser();p.add_argument('--profile',default='ft12k_sim');p.add_argument('--out',type=Path,required=True);p.add_argument('--bank',type=Path);p.add_argument('--per-direction',type=int,default=16);p.add_argument('--workers',type=int,default=4);p.add_argument('--cases',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if a.cases:cases=json.loads(a.cases.read_text())
 else:
  bank=np.load(a.bank);cases=[]
  for label in ['supine','prone','left_side_down','right_side_down']:
   ids=np.flatnonzero(bank['labels']==label)[:a.per_direction]
   for j,k in enumerate(ids):cases.append({'name':f'{label}_{j:02d}','direction':label,'qpos':bank['qpos'][k].tolist(),'record':j==0})
 (a.out/'cases.json').write_text(json.dumps(cases))
 with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
  results=sum(list(pool.map(worker,[(cases[i::a.workers],a.profile,str(a.out)) for i in range(a.workers)])),[])
 import mujoco
 summary={'mujoco':mujoco.__version__,'profile':a.profile,'xml_sha256':hashlib.sha256((ROOT/'g1_description/g1_liao.xml').read_bytes()).hexdigest(),'physics_dt':.002,'control_dt':.02,'episode_seconds':20,'n':len(results),'success':sum(r['success_10s'] for r in results),'per_direction':{k:{'n':sum(r['direction']==k for r in results),'success':sum(r['success_10s'] for r in results if r['direction']==k)} for k in sorted(set(r['direction'] for r in results))},'peak_speed_p95':float(np.percentile([max(r['peak_dq']) for r in results],95)),'peak_power_p95':float(np.percentile([max(r['peak_power']) for r in results],95)),'head_vz_peak_p95':float(np.percentile([r['head_vz_peak'] for r in results],95))}
 (a.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
