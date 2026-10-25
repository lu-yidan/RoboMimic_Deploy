"""Compare all policies against common human-mapped targets, not each own target."""
import argparse,json,os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--video',action='store_true');a=p.parse_args()
out=a.manifest.parent;cases=json.loads(a.manifest.read_text());rows=[]
def yaw(q):
 r=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
 return np.unwrap(np.arctan2(r[:,1,0],r[:,0,0]))
def local(p,y):
 r=Rotation.from_euler('z',-y).as_matrix()
 return np.einsum('tij,tbj->tbi',r,p)
for c in cases:
 direct=dict(np.load(c['direct']));names=direct['body_names'].tolist();bi=[names.index(n) for n in ['left_wrist_yaw_link','right_wrist_yaw_link','left_ankle_roll_link','right_ankle_roll_link']]
 source=direct['body_pos_w'][:,bi]-direct['body_pos_w'][:,0,None]
 sy=yaw(direct['body_quat_w'][:,0]);source_local=local(source,sy)
 for method in ['direct3','direct5','gmr14']:
  folder=out/(c['name']+'_'+method);r=json.loads((folder/'report.json').read_text());t=dict(np.load(folder/'trace.npz'));n=len(t['qpos'])
  selected=t['selected_body_names'].tolist();ix=[selected.index(names[i]) for i in bi]
  actual=t['actual_bodies'][:,ix]-t['actual_bodies'][:,0,None]
  ay=yaw(t['qpos'][:,3:7]);actual_local=local(actual,ay)
  start=min(50,n-1);e=np.linalg.norm(actual_local-source_local[:n],axis=-1)[start:]
  actual_speed=np.linalg.norm(np.diff(actual,axis=0)*50,axis=-1)[start:,:2]
  source_speed=np.linalg.norm(np.diff(source[:n],axis=0)*50,axis=-1)[start:,:2]
  asp=np.percentile(actual_speed,95,axis=0);ssp=np.percentile(source_speed,95,axis=0)
  yaw_error=(ay-ay[0])-(sy[:n]-sy[0])
  source_path=direct['body_pos_w'][:n,0]-direct['body_pos_w'][0,0]
  actual_path=t['qpos'][:,:3]-t['qpos'][0,:3]
  source_path=Rotation.from_euler('z',ay[0]-sy[0]).apply(source_path)
  path_error=np.linalg.norm(actual_path[:,:2]-source_path[:,:2],axis=1)
  row=dict(case=c['name'],method=method,completed=r['completed'],seconds=r['completed_seconds'],requested_seconds=c['seconds'],
   root_path_xy_rmse_m=float(np.sqrt(np.mean(path_error[start:]**2))),
   common_hand_error_m=float(e[:,:2].mean()),common_foot_error_m=float(e[:,2:].mean()),
   common_hand_speed_p95_ratio=(asp/np.maximum(ssp,1e-6)).tolist(),
   relative_yaw_rmse_deg=float(np.degrees(np.sqrt(np.mean(yaw_error[start:]**2)))),
   source_yaw_range_deg=float(np.degrees(np.ptp(sy))),actual_yaw_range_deg=float(np.degrees(np.ptp(ay))),
   contact_slip_mean_m_s=r['contact_slip_mean_m_s'],contact_slip_p95_m_s=r['contact_slip_p95_m_s'],max_tilt_deg=r['max_pelvis_tilt_deg'])
  rows.append(row)
summary=dict(protocol='Same checkpoint, FixedPose initialization, 50 Hz tracker / 500 Hz physics, no support. Full prerecorded target preview. One deterministic rollout per case/method; no realtime deadlines tested.',
 metrics='Common targets: identical scaled human pelvis/hands/ankles for all methods. Root-relative positions expressed in each pelvis heading frame. Discard first 1s for errors and speed. Yaw compares change from first recorded sample. Actual samples are after each 20ms PD interval and compared with the target requested at that interval start. Slip is max loaded foot-ground contact planar speed per frame.',
 rows=rows)
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(rows,indent=2))
if a.video:
 os.environ.setdefault('MUJOCO_GL','egl')
 import mujoco,imageio.v2 as imageio
 from PIL import Image,ImageDraw
 model=mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[2]/'g1_description/scene.xml'))
 data=mujoco.MjData(model);renderer=mujoco.Renderer(model,height=400,width=400)
 cam=mujoco.MjvCamera();cam.distance=3.1;cam.azimuth=135;cam.elevation=-18
 try:
  for name in ['007_ma_rear_hook','021_ma_circle','reactive_004']:
   traces=[dict(np.load(out/(name+'_'+m)/'trace.npz'))['qpos'] for m in ['direct3','direct5','gmr14']]
   with imageio.get_writer(out/(name+'_comparison.mp4'),fps=15,codec='libx264') as writer:
    for ti in np.arange(0,max(map(len,traces))/50,1/15):
     frames=[]
     for label,trace in zip(['No IK: 3 bodies','No IK: 5 bodies','GMR v10: 14 bodies'],traces):
      idx=min(round(ti*50),len(trace)-1);data.qpos[:]=trace[idx];mujoco.mj_forward(model,data)
      cam.lookat[:]=data.qpos[:3];renderer.update_scene(data,cam)
      im=Image.fromarray(renderer.render());ImageDraw.Draw(im).text((8,8),f'{label}\n{name} | {ti:.2f}s'+(' | stopped' if ti*50>=len(trace) else ''),fill='white');frames.append(np.asarray(im))
     writer.append_data(np.concatenate(frames,axis=1))
 finally:renderer.close()
