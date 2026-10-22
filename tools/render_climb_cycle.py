"""Render saved physics states, with contact logs; no ideal-reference animation."""
import argparse
import csv
import json
import hashlib
from pathlib import Path
import sys
import subprocess
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from training.wall_support_env import WallSupportEnv
from training.climb_robustness import apply_rung_offsets


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--speed',type=int,default=1);p.add_argument('--preview-only',action='store_true')
    p.add_argument('--follow-right',action='store_true',help='Keep overall wall view left, follow the robot on the right')
    a=p.parse_args()
    output=a.run/f'cycle_recorded_{a.speed}x.mp4'
    original=a.run/'cycle_recorded_1x.mp4'
    inputs={name:hashlib.sha256((a.run/name).read_bytes()).hexdigest() if (a.run/name).exists() else None for name in ['trajectory.npz','summary.json','independent_audit.json','joint_limit_audit.json','landing_audit.json','policy_provenance.json','policy_reference_audit.json','metadata_erratum.json']}
    renderer_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    original_meta_path=a.run/'render_manifest_1x.json'
    original_meta=json.loads(original_meta_path.read_text()) if original_meta_path.exists() else {}
    if not a.preview_only and output.exists():raise FileExistsError(output)
    if not a.preview_only and a.speed>1 and original.exists() and original_meta.get('inputs')==inputs and original_meta.get('renderer_sha256')==renderer_hash and original_meta.get('follow_right')==a.follow_right:
        badge=Image.new('RGB',(220,30),(16,24,32));badge_draw=ImageDraw.Draw(badge)
        badge_draw.text((8,2),f'播放 {a.speed} 倍速',font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18),fill=(163,208,229))
        badge_path=a.run/f'playback_{a.speed}x.png';badge.save(badge_path)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-n','-i',str(original),
                        '-i',str(badge_path),'-filter_complex',f'[0:v]setpts=PTS/{a.speed}[v];[v][1:v]overlay=1050:41',
                        '-r','25','-an','-c:v','libx264','-crf','18',str(output)],check=True)
        (a.run/f'render_manifest_{a.speed}x.json').write_text(json.dumps(dict(original_meta,playback_speed=a.speed),indent=2))
        print(output);return
    manifest=json.loads((a.run/'manifest.json').read_text())
    summary=json.loads((a.run/'summary.json').read_text());z=dict(np.load(a.run/'trajectory.npz'))
    independent=json.loads((a.run/'independent_audit.json').read_text()) if inputs['independent_audit.json'] else {}
    joint_audit=json.loads((a.run/'joint_limit_audit.json').read_text()) if inputs['joint_limit_audit.json'] else {}
    landing_audit=json.loads((a.run/'landing_audit.json').read_text()) if inputs['landing_audit.json'] else {}
    accepted=bool(summary.get('passed') and independent.get('passed') and joint_audit.get('passed') and (not manifest['config'].get('contact_landing') or landing_audit.get('passed')))
    local_policy=bool(inputs['policy_provenance.json'])
    if local_policy:
        provenance=json.loads((a.run/'policy_provenance.json').read_text())
        if hashlib.sha256((a.run/'policy.zip').read_bytes()).hexdigest()!=provenance['sha256']:raise ValueError('Policy provenance mismatch')
        policy_audit=json.loads((a.run/'policy_reference_audit.json').read_text()) if inputs['policy_reference_audit.json'] else {}
        accepted=bool(accepted and policy_audit.get('passed'))
    controller_label='电机驱动；仅49–61.56s使用训练残差' if local_policy else '仅关节电机驱动；尚非学习策略'
    ref=dict(np.load(a.run/'reference.npz'));cycles=len(ref.get('cycle_ends_s',[ref['time'][-1]]))
    rows=list(csv.DictReader((a.run/'control_log.csv').open(encoding='utf-8')))
    e=WallSupportEnv(manifest['bundle']);e.reset()
    variant_path=a.run/'scene_variant.json'
    if variant_path.exists():apply_rung_offsets(e,json.loads(variant_path.read_text())['offsets'])
    e.model.vis.quality.offsamples=0  # Rendering only: reduce software GL cost.
    renderer=mujoco.Renderer(e.model,height=480,width=640)
    cameras=[]
    for azimuth in [0,65]:
        c=mujoco.MjvCamera();c.lookat[:]=[.26,0,1.13+.125*(cycles-1)];c.distance=2.5+.35*(cycles-1);c.azimuth=azimuth;c.elevation=-8
        cameras.append(c)
    opt=mujoco.MjvOption();opt.geomgroup[3]=0;opt.geomgroup[4]=1
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    initial_z=z['qpos'][0,2]
    def frame(i):
        e.data.qpos[:]=z['qpos'][i];mujoco.mj_forward(e.model,e.data)
        if a.follow_right:
            cameras[1].lookat[:]=e.data.xpos[e.pelvis]+np.array([0.,0.,.05]);cameras[1].distance=2.5
        im=Image.new('RGB',(1280,656),(16,24,32));draw=ImageDraw.Draw(im)
        for j,c in enumerate(cameras):
            renderer.update_scene(e.data,c,scene_option=opt)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
            im.paste(Image.fromarray(renderer.render()),(640*j,74))
        row=rows[max(0,i-1)];t=float(z['time_s'][i]);rise=100*(z['qpos'][i,2]-initial_z)
        draw.text((20,8),f'G1 钩爪攀爬：连续 {cycles} 周期 / 已记录的 MuJoCo 动力学',font=font,fill='white')
        status='验收通过' if accepted else ('诊断录像：关节位置限位未通过' if joint_audit and not joint_audit['passed'] else ('待完成附加审计' if not joint_audit else '诊断录像：未通过完整验收'))
        draw.text((20,43),f'{status}  |  {t:.2f} s  |  阶段 {row["phase"]}',font=small,fill=(163,208,229))
        draw.text((1058,43),f'播放 {a.speed} 倍速',font=small,fill=(163,208,229))
        draw.text((20,564),f'身体抬升 {rise:.2f} cm     跟踪误差 {float(row["base_error_m"])*1000:.2f} mm',font=font,fill='white')
        force='   '.join(f'{label} {float(row[key+"_vertical_n"]):.1f} N' for key,label in
                        [('left_hook','左钩'),('right_hook','右钩'),('left_foot','左脚'),('right_foot','右脚')])
        draw.text((20,603),force+'  |  '+controller_label,font=small,fill=(163,208,229))
        return im
    preview=a.run/'cycle_preview.png';frame(len(z['time_s'])-1).save(preview)
    (a.run/f'render_manifest_{a.speed}x.json').write_text(json.dumps(dict(inputs=inputs,renderer_sha256=renderer_hash,follow_right=a.follow_right,playback_speed=a.speed,preview_only=a.preview_only,extended_acceptance=accepted),indent=2))
    if not a.preview_only:
        writer=imageio_ffmpeg.write_frames(str(output),(1280,656),fps=10,codec='libx264',quality=8,macro_block_size=1)
        writer.send(None)
        try:
            for i in range(0,len(z['time_s']),5*a.speed):writer.send(np.asarray(frame(i)))
            writer.send(np.asarray(frame(len(z['time_s'])-1)))
        finally:writer.close()
        print(output)
    print(preview);renderer.close();e.close()


if __name__=='__main__':main()
