"""Compare accepted recorded dynamics at the same elapsed simulation time."""
import argparse
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--optimized',type=Path,required=True)
    a=p.parse_args()
    for run in [a.baseline,a.optimized]:
        for name in ['summary.json','independent_audit.json']:
            if not json.loads((run/name).read_text())['passed']:
                raise ValueError(f'Only accepted dynamics may be compared: {run/name}')
    style=[json.loads((a.optimized/name).read_text()) for name in ['baseline_style.json','style.json']]
    overlay=Image.new('RGBA',(1280,656),(16,24,32,255));d=ImageDraw.Draw(overlay)
    d.rectangle((0,86,1279,565),fill=(0,0,0,0))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    for x,label,s in zip([18,658],['原动作','优化动作'],style):
        d.text((x,10),label+' · MuJoCo 实际轨迹',font=font,fill='white')
        d.text((x,48),f'最大脚偏航 {s["foot_yaw_peak_deg"]:.1f}° / 膝间距 {100*s["knee_width_peak_m"]:.1f} cm',font=small,fill=(163,208,229))
    d.text((18,580),'同一仿真时刻对照 · 4 倍速 · 较早完成侧保持末帧',font=font,fill='white')
    d.text((18,617),'两侧均完成 25 cm 单周期与独立验收；固定轨迹控制，尚非学习策略。',font=small,fill=(163,208,229))
    png=a.optimized/'comparison_overlay.png';overlay.save(png)
    output=a.optimized/'comparison_4x.mp4'
    filters=('[0:v]crop=640:480:0:74,setpts=PTS/4,tpad=stop_mode=clone:stop_duration=60[a];'
             '[1:v]crop=640:480:0:74,setpts=PTS/4,tpad=stop_mode=clone:stop_duration=60[b];'
             '[a][b]hstack,pad=1280:656:0:86:color=0x101820[base];'
             '[base][2:v]overlay=0:0[v]')
    duration=max(json.loads((r/'summary.json').read_text())['simulation_time_s'] for r in [a.baseline,a.optimized])/4
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-n',
        '-i',str(a.baseline/'cycle_recorded_1x.mp4'),'-i',str(a.optimized/'cycle_recorded_1x.mp4'),
        '-i',str(png),'-filter_complex',filters,'-map','[v]','-t',str(duration),
        '-r','25','-an','-c:v','libx264','-crf','18',str(output)],check=True)
    print(output)


if __name__=='__main__':main()
