"""Render audited saved zero/learned trajectories, with explicit failure labels."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from training.full_cycle_residual_env import FullCycleResidualEnv


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=Path)
    parser.add_argument('--case', default='nominal')
    parser.add_argument('--speed', type=int, default=8)
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    if args.speed < 1:
        raise ValueError('Playback speed must be positive')
    root = args.experiment
    manifest = json.loads((root/'manifest.json').read_text())
    audit = json.loads((root/'independent_training_audit.json').read_text())
    if not audit['passed'] or sha(root/'manifest.json') != audit['experiment_manifest_sha256']:
        raise ValueError('Complete matching integrity audit required')
    if sha(root/'summary.json') != audit['experiment_summary_sha256'] or sha(root/'policy.zip') != audit['checkpoint_sha256']:
        raise ValueError('Experiment changed after audit')
    if sha(manifest['reference']) != manifest['reference_sha256']:
        raise ValueError('Reference changed after experiment')
    output = root/f'comparison_{args.case}_{args.speed}x.mp4'
    if output.exists() and not args.preview_only:
        raise FileExistsError(output)
    envs, data, records, renderers = [], [], [], []
    for side in (0,1):
        source_root=Path(manifest['parent_experiment']) if side==0 else root
        source_audit=json.loads((source_root/'independent_training_audit.json').read_text())
        assert source_audit['passed'] and source_audit['experiment_summary_sha256']==sha(source_root/'summary.json')
        label='after'
        record = next(row for row in source_audit['trajectories'] if row['label'] == label and row['case'] == args.case)
        path = source_root/f'{label}_{args.case}.npz'
        if sha(path) != record['trajectory_sha256']:
            raise ValueError('Saved trajectory changed after audit')
        with np.load(path) as z:
            data.append({key: z[key].copy() for key in ('qpos', 'time_s')})
        env = FullCycleResidualEnv(plan=manifest['reference'], cases={args.case: manifest['evaluation_cases'][args.case]})
        if sha(env.env.bundle/'manifest.json') != manifest['bundle_manifest_sha256']:
            env.close()
            raise ValueError('Scene bundle changed after experiment')
        env.reset(options={'case': args.case})
        env.env.model.vis.quality.offsamples = 0
        envs.append(env)
        renderers.append(mujoco.Renderer(env.env.model, height=480, width=640))
        records.append(record)
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
            name = '原选定策略' if side == 0 else '搜索偏置＋腰部及机身修正'
            verdict = '独立验收通过' if record['independently_accepted'] else '未通过独立验收'
            color = (126, 222, 172) if record['independently_accepted'] else (255, 170, 130)
            draw.text((18+640*side, 8), name, font=font, fill='white')
            draw.text((18+640*side, 39), f"{verdict} | 记录时间 {z['time_s'][i]:.2f}s", font=small, fill=color)
            if t > z['time_s'][-1]+.02:
                draw.rectangle((640*side+14, 493, 640*side+600, 539), fill=(65, 24, 24))
                draw.text((640*side+25, 501), '该实验已终止；此侧保留最后一帧', font=small, fill='white')
            rise = (z['qpos'][i, 2]-z['qpos'][0, 2])*100
            draw.text((18+640*side, 557), f'身体抬升 {rise:.2f} cm', font=small, fill='white')
        draw.text((18, 590), f'{args.case} | {args.speed} 倍速 | 完整动力学记录；脚本参考＋搜索补偿，非新 PPO 训练收益', font=small, fill=(163, 208, 229))
        return image

    preview = root/f'comparison_{args.case}_preview.png'
    frame(horizon).save(preview)
    if not args.preview_only:
        writer = imageio_ffmpeg.write_frames(str(output), (1280, 624), fps=10, codec='libx264', quality=8, macro_block_size=1)
        writer.send(None)
        try:
            for t in np.arange(0., horizon, .1*args.speed):
                writer.send(np.asarray(frame(float(t))))
            writer.send(np.asarray(frame(horizon)))
        finally:
            writer.close()
    manifest_suffix = 'render_preview' if args.preview_only else 'render'
    (root/f'comparison_{args.case}_{manifest_suffix}.json').write_text(json.dumps(dict(
        audit_sha256=sha(root/'independent_training_audit.json'), renderer_sha256=sha(__file__),
        case=args.case, speed=args.speed, preview_only=args.preview_only,
        input_end_time_s=float(horizon), fps=10,
        frame_count=0 if args.preview_only else len(np.arange(0., horizon, .1*args.speed))+1,
        preview_sha256=sha(preview),
        input_trajectories={row['label']: row['trajectory_sha256'] for row in records},
        accepted={row['label']: row['independently_accepted'] for row in records},
        scope='Kinematic rendering of previously recorded physics states; no new simulation or interpolated climb trajectory.'), indent=2))
    for renderer, env in zip(renderers, envs):
        renderer.close()
        env.close()
    print(preview)
    if not args.preview_only:
        print(output)


if __name__ == '__main__':
    main()
