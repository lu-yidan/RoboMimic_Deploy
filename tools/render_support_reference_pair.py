"""Render audited before/after reference trajectories; freeze failures visibly."""
import json
from pathlib import Path
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha


def main():
    out=Path('outputs/support_retarget_calibration_v1/media12.5');out.mkdir(exist_ok=False)
    directories=[Path('outputs')/name/'true_height_12.5mm' for name in ('highrod_height_mismatch_v1','support_retarget_calibration_v1')]
    envs=[];renderers=[];records=[];sources={};speed=8;fps=10
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    options=mujoco.MjvOption();options.geomgroup[3],options.geomgroup[4]=0,1
    camera=mujoco.MjvCamera();camera.lookat[:]=[.26,0.,1.25];camera.distance=2.6;camera.azimuth=65.;camera.elevation=-8.
    try:
        for p in directories:
            m=json.loads((p/'manifest.json').read_text());a=json.loads((p/'independent_audit.json').read_text())
            assert a['integrity_passed'] and a['trajectory_sha256']==sha(p/'trajectory.npz')
            env=FullCycleResidualEnv(plan=m['reference'],cases={m['case']:m['offsets']});env.reset(options={'case':m['case']})
            assert sha(env.env.bundle/'manifest.json')==m['bundle_manifest_sha256']
            env.env.model.vis.quality.offsamples=0
            for i in range(env.env.model.ngeom):
                if (env.env.model.geom(i).name or '').startswith('rung_'):env.env.model.geom_rgba[i]=[.55,.63,.68,1.]
            envs.append(env);renderers.append(mujoco.Renderer(env.env.model,height=480,width=640))
            with np.load(p/'trajectory.npz') as z:records.append(dict(q=z['qpos'].copy(),t=z['time_s'].copy(),accepted=a['independently_accepted']))
            sources[str(p/'trajectory.npz')]=sha(p/'trajectory.npz');sources[str(p/'independent_audit.json')]=sha(p/'independent_audit.json')
        def frame(time):
            canvas=Image.new('RGB',(1280,624),(16,24,32));draw=ImageDraw.Draw(canvas)
            for side,(env,renderer,record) in enumerate(zip(envs,renderers,records)):
                index=min(int(np.searchsorted(record['t'],time)),len(record['t'])-1);e=env.env;e.data.qpos[:]=record['q'][index];mujoco.mj_kinematics(e.model,e.data)
                renderer.update_scene(e.data,camera,scene_option=options);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False;renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
                canvas.paste(Image.fromarray(renderer.render()),(640*side,68))
                label='原参考：60.26秒限位失败' if side==0 else '支撑重定向：113.66秒完整通过'
                draw.text((18+640*side,8),label,font=font,fill='white')
                draw.text((18+640*side,39),f"记录时间 {record['t'][index]:.2f}s | 零策略残差",font=small,fill=(163,208,229))
                if time>record['t'][-1]+.02:
                    draw.rectangle((640*side+15,492,640*side+610,540),fill=(65,24,24));draw.text((640*side+24,503),'该实验已终止；左侧保留最后一帧',font=small,fill='white')
                rise=100*(record['q'][index,2]-record['q'][0,2]);draw.text((18+640*side,556),f'机身实际抬升 {rise:.2f} cm',font=small,fill='white')
            draw.text((18,591),'实际杆高 +12.5 mm | 8倍速 | 已记录动力学重放；参考规划改善，不是新策略训练',font=small,fill=(163,208,229))
            return canvas
        horizon=float(records[1]['t'][-1]);times=list(np.arange(0.,horizon,speed/fps))+[horizon]
        for label,time in [('start',0.),('transfer',60.),('final',horizon)]:frame(time).save(out/f'{label}.png')
        video=out/'reference_repair_8x.mp4';writer=imageio_ffmpeg.write_frames(str(video),(1280,624),fps=fps,codec='libx264',quality=8,macro_block_size=1);writer.send(None)
        try:
            for time in times:writer.send(np.asarray(frame(time)))
        finally:writer.close()
        reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');metadata=next(reader);count=sum(1 for _ in reader)
        assert count==len(times) and tuple(metadata['size'])==(1280,624)
        result=dict(source_hashes=sources,source_sha256=sha(__file__),video_sha256=sha(video),frames=count,fps=fps,speed=speed,decoded_size=metadata['size'],
            simulation_horizon_s=horizon,scope='Saved-state rendering, no resimulation or synthesized continuation. Failed left trajectory visibly freezes at its actual termination.')
        (out/'provenance.json').write_text(json.dumps(result,indent=2));(out/'renderer_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result,indent=2))
    finally:
        for renderer in renderers:renderer.close()
        for env in envs:env.close()


if __name__=='__main__':main()
