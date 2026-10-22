"""Render recorded learned-residual deployment states, without running a policy."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from training.wall_support_env import WallSupportEnv
from training.climb_robustness import apply_rung_offsets


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    output=a.run/'policy_recorded_1x.mp4'
    if output.exists():raise FileExistsError(output)
    z=np.load(a.run/'trajectory.npz');summary=json.loads((a.run/'summary.json').read_text())
    e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset()
    apply_rung_offsets(e,json.loads((a.run/'scene_variant.json').read_text())['offsets'])
    e.model.vis.quality.offsamples=0;renderer=mujoco.Renderer(e.model,height=480,width=640)
    opt=mujoco.MjvOption();opt.geomgroup[3]=0;opt.geomgroup[4]=1
    cameras=[]
    for angle in [0,65]:
        c=mujoco.MjvCamera();c.lookat[:]=[.26,0,1.15];c.distance=2.45;c.azimuth=angle;c.elevation=-8;cameras.append(c)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    def frame(i):
        e.data.qpos[:]=z['qpos'][i];mujoco.mj_forward(e.model,e.data)
        im=Image.new('RGB',(1280,620),(16,24,32));draw=ImageDraw.Draw(im)
        draw.text((20,8),'G1 / PPO 残差试训：右钩单次换杆',font=font,fill='white')
        draw.text((20,43),f'部署入口实跑记录 | {z["time_s"][i]:.2f} s | 原速 | 最终结果：{summary["final"]["reason"]}',font=small,fill=(163,208,229))
        for j,c in enumerate(cameras):
            renderer.update_scene(e.data,c,scene_option=opt);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
            im.paste(Image.fromarray(renderer.render()),(640*j,74))
        draw.text((20,568),'名义力矩控制器 + 右臂小幅修正；2,048 步试训；尚非完整周期学习策略',font=small,fill=(163,208,229))
        return im
    frame(len(z['time_s'])-1).save(a.run/'policy_preview.png')
    writer=imageio_ffmpeg.write_frames(str(output),(1280,620),fps=10,codec='libx264',quality=8,macro_block_size=1);writer.send(None)
    try:
        for i in range(0,len(z['time_s']),5):writer.send(np.asarray(frame(i)))
    finally:writer.close();renderer.close();e.close()
    print(output)


if __name__=='__main__':main()
