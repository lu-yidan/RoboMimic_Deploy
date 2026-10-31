"""Reconstruct endpoint contact geometry from audited poses; never infer force."""
import argparse
import csv
import json
from pathlib import Path
import mujoco
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.compare_audited_closure_candidates import load_run,sha


def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 verified=load_run(a.experiment);manifest=json.loads((a.experiment/'manifest.json').read_text())
 env=FullCycleResidualEnv(plan=manifest['reference'],cases=manifest['evaluation_cases']);rows=[];summaries=[]
 try:
  for case in manifest['evaluation_cases']:
   env.reset(options={'case':case});e=env.env;m=e.model;d=mujoco.MjData(m)
   with np.load(a.experiment/f'after_{case}.npz') as z:
    mask=z['time_s']>111.66;poses=z['qpos'][mask];times=z['time_s'][mask]
   for q,t in zip(poses,times):
    d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d)
    for contact in d.contact:
     ga,gb=int(contact.geom1),int(contact.geom2);ba,bb=int(m.geom_bodyid[ga]),int(m.geom_bodyid[gb])
     ia,ib=ba in e.robot_bodies,bb in e.robot_bodies
     if ia==ib:continue
     body,robot,external=(ba,ga,gb) if ia else (bb,gb,ga)
     group=e.contact_groups.get(body,'other_robot')
     local=d.xmat[body].reshape(3,3).T@(contact.pos-d.xpos[body])
     rows.append(dict(case=case,time_s=float(t),group=group,robot_geom=m.geom(robot).name,external_geom=m.geom(external).name,
      local_x_m=float(local[0]),local_y_m=float(local[1]),local_z_m=float(local[2]),distance_m=float(contact.dist)))
   groups={}
   for group in e.contact_groups.values():
    selected=[r for r in rows if r['case']==case and r['group']==group]
    counts=[sum(r['time_s']==float(t) for r in selected) for t in times]
    groups[group]=dict(contact_count_range=[min(counts),max(counts)],
     local_x_range_m=[min(r['local_x_m'] for r in selected),max(r['local_x_m'] for r in selected)] if selected else None,
     separation_range_m=[min(r['distance_m'] for r in selected),max(r['distance_m'] for r in selected)] if selected else None,
     pairs=sorted({(r['robot_geom'],r['external_geom']) for r in selected}))
   summaries.append(dict(case=case,success=verified['rows'][case]['success'],samples=len(times),groups=groups))
 finally:env.close()
 a.output.mkdir(exist_ok=False)
 with (a.output/'contact_geometry.csv').open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 result=dict(rows=summaries,audit_sha256=verified['audit_sha256'],source_sha256=sha(__file__),
  scope='MuJoCo collision geometry reconstructed at saved 50 Hz endpoint poses. Not original solver contact forces, center of pressure, continuous contact proof, or causal diagnosis.')
 (a.output/'summary.json').write_text(json.dumps(result,indent=2));(a.output/'diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
 print(json.dumps(summaries))

if __name__=='__main__':main()
