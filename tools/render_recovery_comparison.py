"""Render matched deployment traces: columns=profiles, rows=four directions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.validate_ft12k_sim import make_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--scenario', default='nominal')
    p.add_argument('--profiles', nargs='+', default=['ft_r2_9000', 'path_a6_9999', 'm2_9999'])
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    m = make_model(); d = mujoco.MjData(m)
    w, h = 480, 320
    renderer = mujoco.Renderer(m, height=h, width=w)
    cam = mujoco.MjvCamera(); cam.distance=2.9; cam.azimuth=130; cam.elevation=-22
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    labels = ['supine','prone','left_side_down','right_side_down']
    traces = {}
    for row, label in enumerate(labels):
        for col, profile in enumerate(a.profiles):
            file = a.run / a.scenario / profile / (label+'_00.npz')
            traces[row,col] = (np.load(file), json.loads(file.with_suffix('.json').read_text()))
    cmd = ['ffmpeg','-nostdin','-n','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w*len(a.profiles)}x{h*4}',
           '-framerate','25','-i','pipe:0','-an','-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(a.out)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as writer:
        for frame in range(500):
            canvas = Image.new('RGB',(w*len(a.profiles),h*4))
            for (row,col),(trace,report) in traces.items():
                idx=min(frame*2,len(trace['state'])-1);s=trace['state'][idx]
                nq,nv=int(trace['nq']),int(trace['nv']);d.qpos[:]=s[:nq];d.qvel[:]=s[nq:nq+nv]
                mujoco.mj_forward(m,d);cam.lookat[:]=[d.qpos[0],d.qpos[1],.65]
                renderer.update_scene(d,camera=cam);tile=Image.fromarray(renderer.render());draw=ImageDraw.Draw(tile)
                draw.rectangle((0,0,w,40),fill='#101d28')
                draw.text((6,2),f'{a.profiles[col]} | {labels[row]}',font=font,fill='white')
                draw.text((6,21),f'{a.scenario} | {(frame*2+1)*.02:.2f}s | hold {s[nq+nv+2]:.1f}s | 1x',font=font,fill='#f0dda0')
                if report['unstable'] and (frame*2+1)*.02>report['completed_sim_s']:
                    draw.rectangle((0,130,w,190),fill='black');draw.text((8,150),'NUMERICAL FAILURE: stopped',font=font,fill='red')
                canvas.paste(tile,(col*w,row*h))
            writer.stdin.write(np.asarray(canvas).tobytes())
            if frame in (0,125,250,499):canvas.save(a.out.with_name(a.out.stem+f'_{frame:03d}.jpg'))
        writer.stdin.close();assert writer.wait()==0
    renderer.close()


if __name__=='__main__':main()
