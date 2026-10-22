"""Independently validate saved timing, video coverage and repeat-run agreement."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=ROOT/'outputs/standup_baseline_v2')
    p.add_argument('--repeat',type=Path,default=ROOT/'outputs/standup_baseline_repeat_v2')
    args=p.parse_args()
    a=np.load(args.run/'trajectory.npz');b=np.load(args.repeat/'trajectory.npz')
    metrics=json.loads((args.run/'metrics.json').read_text());repeat=json.loads((args.repeat/'metrics.json').read_text())
    assert metrics['passed'] and repeat['passed']
    difference={k:float(np.max(abs(a[k]-b[k]))) for k in ('physics','qpos','qvel','target_joint_positions')}
    assert max(difference.values())<1e-10,difference
    assert np.isfinite(a['physics']).all()
    assert np.allclose(np.diff(a['physics'][:,0]),.002,atol=1e-12)
    rows=list(csv.DictReader((args.run/'control_log.csv').open()))
    indices=[int(r['reference_frame']) for r in rows if r['stage']=='motion']
    assert indices==list(range(450))
    limits=np.asarray(json.loads((args.run/'provenance.json').read_text())['resolved_config']['tau_limit'])
    assert np.all(abs(a['physics'][:,-29:])<=limits+1e-9)
    reader=imageio_ffmpeg.read_frames(str(args.run/'full_run.mp4'));metadata=next(reader)
    wanted=[0,50,100,150,200,282];selected=[];count=0
    for i,frame in enumerate(reader):
        if i in wanted:
            image=Image.frombytes('RGB',metadata['size'],frame)
            image.save(args.run/f'video_frame_{i:03d}.png');selected.append((i,image))
        count+=1
    assert count==metrics['video_frames']==len(a['video_sim_times_s'])
    assert abs(metadata['duration']-count/metadata['fps'])<.01
    assert abs(a['video_sim_times_s'][-1]-metrics['duration_s'])<1e-9
    canvas=Image.new('RGB',(1440,1205),'#edf1f4');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24)
    draw.text((20,12),'原场景 · 学习策略完成躺下站起 · 实际仿真录像取帧',font=font,fill='#233448')
    for j,(index,image) in enumerate(selected):
        image=image.resize((720,405));x=(j%2)*720;y=55+(j//2)*380
        # Remove the video header in this contact sheet only; full video is unchanged.
        image=image.crop((0,48,720,405));canvas.paste(image,(x,y))
        draw.text((x+10,y+328),f't = {a["video_sim_times_s"][index]:.2f} s',font=font,fill='#ffffff',stroke_width=1,stroke_fill='#172b40')
    canvas.save(args.run/'storyboard.png')
    audit=dict(passed=True,independent_repeat_max_absolute_difference=difference,
               decoded_video=metadata,decoded_frames=count,physics_step_s=.002,
               complete_reference_indices=True,torque_limits_respected=True,
               video_last_simulation_time_s=float(a['video_sim_times_s'][-1]),
               video_extra_final_frame_hold_s=1/metadata['fps'])
    (args.run/'audit.json').write_text(json.dumps(audit,indent=2))
    hashes={str(f.relative_to(args.run)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in args.run.rglob('*') if f.is_file() and f.name!='artifact_sha256.json'}
    (args.run/'artifact_sha256.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
