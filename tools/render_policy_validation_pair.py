"""Render saved zero/actor validation trajectories with audited endpoint labels."""
import argparse
import json
from pathlib import Path
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    p = args.experiment
    manifest = json.loads((p/'manifest.json').read_text())
    audit = json.loads((p/'independent_training_audit.json').read_text())
    assert audit['passed'] and audit['experiment_manifest_sha256'] == sha(p/'manifest.json')
    assert audit['checkpoint_sha256'] == sha(p/'policy.zip')
    args.output.mkdir(exist_ok=False)
    envs, renderers, records = [], [], []
    sources = {str(p/'independent_training_audit.json'): sha(p/'independent_training_audit.json')}
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 22)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.26, 0., 1.25]
    camera.distance, camera.azimuth, camera.elevation = 2.6, 65., -8.
    options = mujoco.MjvOption()
    options.geomgroup[3], options.geomgroup[4] = 0, 1
    try:
        for label in ('zero', 'after'):
            row = next(r for r in audit['trajectories'] if r['label'] == label and r['case'] == args.case)
            trajectory = p/f'{label}_{args.case}.npz'
            assert row['passed'] and row['trajectory_sha256'] == sha(trajectory)
            env = FullCycleResidualEnv(plan=manifest['reference'], cases={args.case: manifest['evaluation_cases'][args.case]})
            env.reset(options={'case': args.case})
            assert sha(env.env.bundle/'manifest.json') == manifest['bundle_manifest_sha256']
            env.env.model.vis.quality.offsamples = 0
            for i in range(env.env.model.ngeom):
                if (env.env.model.geom(i).name or '').startswith('rung_'):
                    env.env.model.geom_rgba[i] = [.55, .63, .68, 1.]
            envs.append(env)
            renderers.append(mujoco.Renderer(env.env.model, height=480, width=640))
            with np.load(trajectory) as z:
                records.append(dict(q=z['qpos'].copy(), t=z['time_s'].copy(), audit=row))
            sources[str(trajectory)] = sha(trajectory)

        def frame(time):
            canvas = Image.new('RGB', (1280, 656), (16, 24, 32))
            draw = ImageDraw.Draw(canvas)
            for side, (env, renderer, record) in enumerate(zip(envs, renderers, records)):
                index = min(int(np.searchsorted(record['t'], time)), len(record['t'])-1)
                e = env.env
                e.data.qpos[:] = record['q'][index]
                mujoco.mj_kinematics(e.model, e.data)
                renderer.update_scene(e.data, camera, scene_option=options)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                canvas.paste(Image.fromarray(renderer.render()), (640*side, 68))
                passed = record['audit']['independently_accepted']
                title = '零残差参考' if side == 0 else '同一参考 + 已有学习策略'
                draw.text((18+640*side, 8), title, font=font, fill='white')
                draw.text((18+640*side, 39), f"记录时间 {record['t'][index]:.2f}s | 最终判定：{'通过' if passed else '失败'}", font=small, fill=(163, 208, 229))
                endpoint = record['audit']['independent_endpoints'][0]
                closure = endpoint['joint_closure_rad']*1000
                draw.text((18+640*side, 552), f'末2秒最大关节闭合误差 {closure:.3f} mrad / 上限40', font=small, fill='white')
                rise = 100*(record['q'][index, 2]-record['q'][0, 2])
                draw.text((18+640*side, 580), f'当前实际机身抬升 {rise:.2f} cm', font=small, fill='white')
            draw.text((18, 621), '实际杆高 +13.75 mm、x偏移0 | 8倍速 | 已记录轨迹重放；已有策略迁移，无新增训练', font=small, fill=(163, 208, 229))
            return canvas

        # This explanatory caption is specific to the audited newly repaired case.
        assert args.case == 'foot_x+0_z13.75'
        assert records[0]['t'][-1] == records[1]['t'][-1]
        horizon = float(records[1]['t'][-1])
        times = list(np.arange(0., horizon, .8)) + [horizon]
        for label, timestamp in [('start', 0.), ('transfer', 60.), ('final', horizon)]:
            frame(timestamp).save(args.output/f'{label}.png')
        video = args.output/'policy_transfer_8x.mp4'
        writer = imageio_ffmpeg.write_frames(str(video), (1280, 656), fps=10, codec='libx264', quality=8, macro_block_size=1)
        writer.send(None)
        try:
            for timestamp in times:
                writer.send(np.asarray(frame(timestamp)))
        finally:
            writer.close()
        reader = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
        metadata = next(reader)
        count = sum(1 for _ in reader)
        assert count == len(times) and tuple(metadata['size']) == (1280, 656)
        result = dict(source_hashes=sources, source_sha256=sha(__file__), video_sha256=sha(video), frames=count,
                      fps=10, speed=8, simulation_horizon_s=horizon, decoded_size=metadata['size'],
                      scope='Audited saved-state visualization only. No synthesized motion, resimulation, new learning, camera perception or hardware validation.')
        (args.output/'provenance.json').write_text(json.dumps(result, indent=2))
        (args.output/'renderer_source.py').write_bytes(Path(__file__).read_bytes())
        print(json.dumps(result, indent=2))
    finally:
        for renderer in renderers:
            renderer.close()
        for env in envs:
            env.close()


if __name__ == '__main__':
    main()
