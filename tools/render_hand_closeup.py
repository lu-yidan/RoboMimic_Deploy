"""Render recorded dynamic states from a new camera, without resimulation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from training.wall_support_env import WallSupportEnv


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    output=a.run/'recorded_right_hook_closeup_v2.mp4'
    if output.exists():raise FileExistsError(output)
    manifest=json.loads((a.run/'manifest.json').read_text());summary=json.loads((a.run/'summary.json').read_text())
    e=WallSupportEnv(manifest['bundle']);e.reset();z=np.load(a.run/'trajectory.npz')
    r=mujoco.Renderer(e.model,height=600,width=800);c=mujoco.MjvCamera()
    c.lookat[:]=[.33,-.2,1.58];c.distance=.65;c.azimuth=0;c.elevation=-10
    o=mujoco.MjvOption();o.geomgroup[3]=0;o.geomgroup[4]=1
    w=imageio_ffmpeg.write_frames(str(output),(800,600),fps=25,codec='libx264',quality=8,macro_block_size=1);w.send(None)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    try:
        for i in range(0,len(z['time_s']),2):
            e.data.qpos[:]=z['qpos'][i];mujoco.mj_forward(e.model,e.data);r.update_scene(e.data,c,scene_option=o)
            im=Image.fromarray(r.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,800,55),fill=(17,26,35))
            draw.text((12,7),f'Recorded dynamics: right hook close-up | {z["time_s"][i]:.2f} s',font=font,fill='white')
            draw.text((12,30),summary['task']+' | new camera, same recorded trajectory',font=font,fill=(190,216,234))
            w.send(np.asarray(im))
    finally:w.close();r.close();e.close()


if __name__=='__main__':main()
