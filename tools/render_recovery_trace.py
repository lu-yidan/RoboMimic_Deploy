"""Render saved deployment-simulation states, without resimulating dynamics."""
import argparse,sys,subprocess
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate_ft12k_sim import make_model
p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('output',type=Path);p.add_argument('--duration',type=float,default=20);p.add_argument('--slow',type=float,default=1.);p.add_argument('--azimuth',type=float,default=90);a=p.parse_args()
f=np.load(a.trace);states=f['state'];m=make_model();d=mujoco.MjData(m);m.vis.global_.offwidth=960;m.vis.global_.offheight=720;renderer=mujoco.Renderer(m,height=720,width=960)
cam=mujoco.MjvCamera();cam.lookat[:]=[0,0,.55];cam.distance=2.4;cam.azimuth=a.azimuth;cam.elevation=-12
nq=int(f['nq']);nv=int(f['nv']);a.output.parent.mkdir(parents=True,exist_ok=True)
with subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size','960x720','-framerate',str(25*a.slow),'-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(a.output)],stdin=subprocess.PIPE) as writer:
 for i in range(0,min(len(states),round(a.duration/.02)),2):
  s=states[i];d.qpos[:]=s[:nq];d.qvel[:]=s[nq:nq+nv];mujoco.mj_forward(m,d)
  renderer.update_scene(d,camera=cam);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,960,44),fill='black');draw.text((12,8),f'FT 12000 | RoboMimic CPU MuJoCo {mujoco.__version__} | t={(i+1)*.02:.2f}s | {a.slow}x | head={s[nq+nv]:.2f}m | hold={s[nq+nv+2]:.2f}s',fill='white');writer.stdin.write(np.asarray(im).tobytes())
 writer.stdin.close();code=writer.wait();assert code==0,code
renderer.close()
