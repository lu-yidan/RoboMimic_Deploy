"""Render audited fixed-reference before/after trajectories."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from training.full_cycle_residual_env import FullCycleResidualEnv
from tools.compare_audited_closure_candidates import load_run


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--after',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',default='foot_z_plus15')
    parser.add_argument('--speed',type=int,default=4)
    args=parser.parse_args();assert args.speed>=1
    runs=[load_run(args.before),load_run(args.after)]
    assert runs[0]['reference_sha256']==runs[1]['reference_sha256']
    assert runs[0]['cases'][args.case]==runs[1]['cases'][args.case]
    root=args.output;root.mkdir(exist_ok=False);output=root/'comparison.mp4'
    envs,data,records,renderers=[],[],[],[]
    for source,run in zip((args.before,args.after),runs):
        manifest=json.loads((source/'manifest.json').read_text())
        path=source/f'after_{args.case}.npz'
        record=dict(**run['rows'][args.case],trajectory_sha256=sha(path),independently_accepted=run['rows'][args.case]['success'])
        with np.load(path) as z:data.append({key:z[key].copy() for key in ('qpos','time_s')})
        env=FullCycleResidualEnv(plan=manifest['reference'],cases={args.case:manifest['evaluation_cases'][args.case]})
        assert sha(env.env.bundle/'manifest.json')==manifest['bundle_manifest_sha256']
        env.reset(options={'case':args.case});env.env.model.vis.quality.offsamples=0
        envs.append(env);records.append(record)
        renderers.append(mujoco.Renderer(env.env.model,height=480,width=640))
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.26, 0., 1.25]
    camera.distance, camera.azimuth, camera.elevation = 2.6, 65., -8.
    options = mujoco.MjvOption()
    options.geomgroup[3], options.geomgroup[4] = 0, 1
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 23)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
    horizon = max(z['time_s'][-1] for z in data)

    def frame(t):
        image = Image.new('RGB', (1280, 624), (16, 24, 32))
        draw = ImageDraw.Draw(image)
        for side, (env, z, record, renderer) in enumerate(zip(envs, data, records, renderers)):
            i = min(int(np.searchsorted(z['time_s'], t, side='left')), len(z['time_s'])-1)
            e = env.env
            e.data.qpos[:] = z['qpos'][i]
            mujoco.mj_kinematics(e.model, e.data)
            renderer.update_scene(e.data, camera, scene_option=options)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            image.paste(Image.fromarray(renderer.render()), (640*side, 68))
            name = '原固定参考候选' if side == 0 else '最新固定参考候选（7/8）'
            verdict = '独立验收通过' if record['independently_accepted'] else '未通过独立验收'
            color = (126, 222, 172) if record['independently_accepted'] else (255, 170, 130)
            draw.text((18+640*side, 8), name, font=font, fill='white')
            draw.text((18+640*side, 39), f"{verdict} | 记录时间 {z['time_s'][i]:.2f}s", font=small, fill=color)
            if t > z['time_s'][-1]+.02:
                draw.rectangle((640*side+14, 493, 640*side+600, 539), fill=(65, 24, 24))
                draw.text((640*side+25, 501), '该实验已终止；此侧保留最后一帧', font=small, fill='white')
            rise = (z['qpos'][i, 2]-z['qpos'][0, 2])*100
            draw.text((18+640*side, 557), f"抬升 {rise:.2f} cm | 收尾误差 {record['closure_mrad']:.3f} mrad", font=small, fill='white')
        draw.text((18, 590), f'{args.case} | {args.speed} 倍速 | 记录状态回放；单周期约 0.25 米，脚本参考 + 残差策略', font=small, fill=(163, 208, 229))
        return image

    preview=root/'preview.png';frame(horizon).save(preview)
    writer=imageio_ffmpeg.write_frames(str(output),(1280,624),fps=10,codec='libx264',quality=8,macro_block_size=1);writer.send(None)
    try:
        for t in np.r_[np.arange(0.,horizon,.1*args.speed),horizon]:writer.send(np.asarray(frame(float(t))))
    finally:writer.close()
    result=dict(case=args.case,speed=args.speed,fps=10,source_sha256=sha(__file__),before=runs[0],after=runs[1],
        input_trajectory_sha256=[r['trajectory_sha256'] for r in records],video_sha256=sha(output),preview_sha256=sha(preview),
        scope='Rendering audited recorded states, no new dynamics. One scripted-reference cycle only; Same actor; added fixed reference commands improved eight-case success from 5/8 to 7/8.')
    (root/'manifest.json').write_text(json.dumps(result,indent=2));(root/'render_source.py').write_bytes(Path(__file__).read_bytes())
    for renderer,env in zip(renderers,envs):renderer.close();env.close()
    print(output)


if __name__=='__main__':main()
