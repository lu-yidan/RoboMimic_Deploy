"""Render audited high-rod handoff, distinguishing learned and classical control."""
import argparse
import json
from pathlib import Path
import subprocess
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from deploy_mujoco.run_trained_full_cycle import load_verified, sha
from training.strict_full_cycle_residual_env import FullCycleResidualEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--speed', type=int, default=8)
    args = parser.parse_args()
    if args.speed < 1:
        raise ValueError('Positive playback speed required')
    provenance = json.loads((args.run/'manifest.json').read_text())
    summary = json.loads((args.run/'summary.json').read_text())
    audit = json.loads((args.run/'independent_audit.json').read_text())
    assert sha(args.run/'trajectory.npz') == summary['trajectory_sha256'] == audit['trajectory_sha256']
    assert sha(args.run/'reference.npz') == provenance['plan_sha256']
    assert sha(args.plan) == provenance['plan_sha256']
    assert all(sha(args.run/'source_snapshot'/name) == digest for name, digest in provenance['sources'].items())
    manifest = json.loads((Path(provenance['parent_experiment'])/'manifest.json').read_text())
    env = FullCycleResidualEnv(plan=args.plan, cases={provenance['case']: provenance['offsets']},
        normal_bias_y_deg=manifest['normal_bias_y_deg'], memory_enabled=manifest['force_history_memory'])
    env.reset(options={'case': provenance['case']})
    with np.load(args.run/'trajectory.npz') as z:
        q, times = z['qpos'].copy(), z['time_s'].copy()
    mode = 'handoff'
    cycles = len(env.ends)
    output = args.run/f'{mode}_{cycles}_cycles_{args.speed}x.mp4'
    if output.exists():
        raise FileExistsError(output)
    e = env.env
    e.model.vis.quality.offsamples = 0
    # Rendering only: remove stale orange highlights of the initial rods.
    # Physics, geometry and recorded state arrays are unchanged.
    for geom in range(e.model.ngeom):
        if (e.model.geom(geom).name or '').startswith('rung_'):
            e.model.geom_rgba[geom] = [.55, .63, .68, 1.]
    renderer = mujoco.Renderer(e.model, height=480, width=640)
    cameras = []
    for azimuth in (0, 65):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.26, 0., 1.13+.125*(cycles-1)]
        camera.distance, camera.azimuth, camera.elevation = 2.5+.35*(cycles-1), azimuth, -8
        cameras.append(camera)
    options = mujoco.MjvOption()
    options.geomgroup[3], options.geomgroup[4] = 0, 1
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 23)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)

    def frame(t):
        i = min(int(np.searchsorted(times, t)), len(times)-1)
        e.data.qpos[:] = q[i]
        mujoco.mj_kinematics(e.model, e.data)
        cameras[1].lookat[:] = e.data.xpos[e.pelvis]+np.array([0., 0., .05])
        cameras[1].distance = 2.5
        canvas = Image.new('RGB', (1280, 656), (16, 24, 32))
        draw = ImageDraw.Draw(canvas)
        for side, camera in enumerate(cameras):
            renderer.update_scene(e.data, camera, scene_option=options)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            canvas.paste(Image.fromarray(renderer.render()), (side*640, 74))
        cycle = min(int(np.searchsorted(env.ends, times[i], side='right'))+1, len(env.ends))
        phase = env.reference(float(times[i]))[3]
        verdict = '运行时与末端验收通过' if (summary['passed'] and audit['passed']) else '未通过；保留真实终止状态'
        label = '高杆适配后衔接经典控制'
        draw.text((20, 8), f'G1 黑色钩爪：{label} / 目标连续 {cycles} 个周期', font=font, fill='white')
        draw.text((20, 43), f'{verdict} | 记录 {times[i]:.2f} s | 周期 {cycle}/{cycles} | {args.speed} 倍速', font=small, fill=(163, 208, 229))
        rise = float(q[i, 2]-q[0, 2])
        draw.text((20, 564), f'根部实际抬升 {rise*100:.2f} cm | 阶段 {phase}', font=font, fill='white')
        detail = '首周期学习残差与固定参考，次周期衰减后经典控制'
        draw.text((20, 605), f'{detail}；周期间无物理重置；脚本参考', font=small, fill=(163, 208, 229))
        return canvas

    preview = args.run/f'{mode}_{cycles}_cycles_preview.png'
    frame(float(times[-1])).save(preview)
    render_times = list(np.arange(0., times[-1], .1*args.speed))+[float(times[-1])]
    writer = imageio_ffmpeg.write_frames(str(output), (1280, 656), fps=10, codec='libx264', quality=8, macro_block_size=1)
    writer.send(None)
    try:
        for t in render_times:
            writer.send(np.asarray(frame(t)))
    finally:
        writer.close()
        renderer.close()
        env.close()
    decode = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(output), '-f', 'null', '-'], capture_output=True, text=True)
    evidence = dict(input_trajectory_sha256=summary['trajectory_sha256'], summary_sha256=sha(args.run/'summary.json'),
        provenance_sha256=sha(args.run/'manifest.json'), source_sha256=sha(__file__), output_sha256=sha(output),
        preview_sha256=sha(preview), frame_count=len(render_times), fps=10, playback_speed=args.speed,
        rendering_only_rod_colors_normalized=True,
        decode_passed=decode.returncode == 0, decode_errors=decode.stderr,
        recorded_horizon_s=float(times[-1]), measured_root_rise_m=float(q[-1, 2]-q[0, 2]),
        scope='Rendering saved physics states of a mixed learned/classical handoff. Not a learned two-cycle policy or a newly simulated run.', independent_audit_sha256=sha(args.run/'independent_audit.json'))
    (args.run/'render_manifest.json').write_text(json.dumps(evidence, indent=2))
    (args.run/'render_source.py').write_bytes(Path(__file__).read_bytes())
    if decode.returncode:
        raise RuntimeError('Video decoding failed: '+decode.stderr)
    print(json.dumps(evidence), flush=True)


if __name__ == '__main__':
    main()
