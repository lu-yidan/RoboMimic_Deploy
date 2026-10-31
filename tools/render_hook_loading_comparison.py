"""Synchronized recorded-pose comparison with independently sampled force plots."""
import argparse
from pathlib import Path
import json
import subprocess
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.validate_swing_reference import read,write,sha


def load(run,audit_name):
    m=read(run/'manifest.json');a=read(run/audit_name)
    assert a['passed'] and a['trajectory_sha256']==sha(run/'trajectory.npz')
    if 'manifest_sha256' in a:assert a['manifest_sha256']==sha(run/'manifest.json')
    assert a['telemetry_sha256']==sha(run/'physics_telemetry.npz')
    final=read(run/'v2_multicycle_audit.json')
    assert final['passed'] and final['trajectory_sha256']==a['trajectory_sha256']
    assert final['manifest_sha256']==sha(run/'manifest.json')
    assert sha(run/'reference.npz')==m['plan_sha256']
    assert not m.get('scene_compiler') and not m.get('timed_loading')
    with np.load(run/'trajectory.npz') as f:trajectory={k:f[k] for k in ('qpos','time_s')}
    with np.load(run/'physics_telemetry.npz') as f:telemetry={k:f[k] for k in ('time_s','group_force_world_n')}
    telemetry['groups']=a['groups'] if 'groups' in a else read(run/'hook_admission_record.json')['groups']
    return m,trajectory,telemetry


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();b,x,z=load(a.baseline,'telemetry_audit.json');c,y,w=load(a.candidate,'hook_admission_audit_v2.json')
    assert not b.get('contact_admission') and c.get('contact_admission')
    for key in ('case','offsets','plan_sha256','checkpoint_sha256','parent_experiment'):assert b[key]==c[key]
    assert b['cycles']==c['cycles']==2 and b['case']=='training_12'
    parent=read(Path(b['parent_experiment'])/'manifest.json')
    env=FullCycleResidualEnv(plan=Path(b['plan_path']),cases={b['case']:b['offsets']},
        normal_bias_y_deg=parent['normal_bias_y_deg'],memory_enabled=parent['force_history_memory'])
    env.reset(options={'case':b['case']});e=env.env
    a.output.mkdir(parents=True,exist_ok=False)
    e.model.vis.quality.offsamples=0
    for i in range(e.model.ngeom):
        if (e.model.geom(i).name or '').startswith('rung_'):e.model.geom_rgba[i]=[.55,.63,.68,1.]
    renderer=mujoco.Renderer(e.model,height=400,width=640)
    camera=mujoco.MjvCamera();camera.distance=2.3;camera.azimuth=65;camera.elevation=-8
    options=mujoco.MjvOption();options.geomgroup[3]=0;options.geomgroup[4]=1
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    start,end,fps=126.66,143.16,20
    colors=((255,140,100),(92,196,245));plots=[];peak_records=[]
    for run,telemetry in ((a.baseline,z),(a.candidate,w)):
        groups=telemetry['groups']
        selected=(telemetry['time_s']>=start)&(telemetry['time_s']<=end)
        t=telemetry['time_s'][selected];forces=telemetry['group_force_world_n'][selected]
        plot=Image.new('RGB',(640,170),(22,31,42));draw=ImageDraw.Draw(plot)
        for value in (0,150,300,450):
            yy=140-value/450*110;draw.line((45,yy,620,yy),fill=(65,78,91))
            draw.text((4,yy-10),str(value),font=small,fill='white')
        peaks={}
        for group,color in zip(('left_hook','right_hook'),colors):
            values=np.linalg.norm(forces[:,groups.index(group)],axis=1);peaks[group]=float(values.max())
            points=list(zip(45+(t-start)/(end-start)*575,140-values/450*110))
            draw.line(points,fill=color,width=2)
        draw.text((45,0),'钩爪净接触力（N）：左手 / 右手；2000 Hz',font=small,fill='white')
        draw.text((45,145),f'{start:.2f} s',font=small,fill='white');draw.text((540,145),f'{end:.2f} s',font=small,fill='white')
        plots.append(plot);peak_records.append(peaks)
    def frame(t):
        canvas=Image.new('RGB',(1280,752),(16,24,32));draw=ImageDraw.Draw(canvas)
        draw.text((18,8),'T12 第二周期挂接：同参考、同冻结策略；记录对照（1 倍速）',font=font,fill='white')
        draw.text((18,43),'已采用 v2',font=font,fill='white')
        draw.text((660,43),'接触确认候选（未采用）',font=font,fill='white')
        for column,trajectory in enumerate((x,y)):
            i=max(0,min(int(np.searchsorted(trajectory['time_s'],t+1e-8,side='right'))-1,len(trajectory['time_s'])-1))
            e.data.qpos[:]=trajectory['qpos'][i];mujoco.mj_kinematics(e.model,e.data)
            camera.lookat[:]=e.data.xpos[e.pelvis]+np.array([0,0,.1])
            renderer.update_scene(e.data,camera,scene_option=options)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
            canvas.paste(Image.fromarray(renderer.render()),(column*640,78))
            canvas.paste(plots[column],(column*640,480))
            draw=ImageDraw.Draw(canvas);cursor=column*640+45+(t-start)/(end-start)*575
            draw.line((cursor,506,cursor,620),fill='white',width=2)
            peaks=peak_records[column]
            draw.text((column*640+18,655),f'本片段峰值：左 {peaks["left_hook"]:.1f} N / 右 {peaks["right_hook"]:.1f} N',font=small,fill='white')
        draw.text((18,684),f'记录时刻 {t:.2f} s | 50 Hz 实测姿态回放；力曲线独立保存于 2000 Hz；非实时控制',font=small,fill=(163,208,229))
        draw.text((18,716),'代价同时保留：第二周期闭合误差 25.66 / 31.48 mrad；理想反馈，本轮未重训',font=small,fill=(255,190,120))
        return canvas
    output=a.output/'hook_loading_paired_1x.mp4';times=np.arange(start,end+1e-8,1/fps)
    writer=imageio_ffmpeg.write_frames(str(output),(1280,752),fps=fps,codec='libx264',quality=8,macro_block_size=1)
    writer.send(None)
    try:
        frame(140.72).save(a.output/'preview.png')
        for t in times:writer.send(np.asarray(frame(float(t))))
    finally:writer.close();renderer.close();env.close()
    decode=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(output),'-f','null','-'],capture_output=True,text=True)
    report=dict(decode_passed=decode.returncode==0,decode_errors=decode.stderr,
        baseline_run=str(a.baseline.resolve()),candidate_run=str(a.candidate.resolve()),
        baseline_trajectory_sha256=sha(a.baseline/'trajectory.npz'),candidate_trajectory_sha256=sha(a.candidate/'trajectory.npz'),
        baseline_telemetry_sha256=sha(a.baseline/'physics_telemetry.npz'),candidate_telemetry_sha256=sha(a.candidate/'physics_telemetry.npz'),
        source_sha256=sha(__file__),video_sha256=sha(output),preview_sha256=sha(a.output/'preview.png'),
        frames=len(times),fps=fps,speed=1,recorded_window_s=[start,end],window_force_peaks_n=peak_records,
        scope='Recorded poses only; no simulation stepped or controller changed. Poses held from nearest prior 50 Hz sample, not physics-resolution video. Full-rate force curves cover the entire displayed window. Experimental candidate remains unadopted.')
    write(a.output/'render_manifest.json',report);(a.output/'render_source.py').write_bytes(Path(__file__).read_bytes())
    assert decode.returncode==0,decode.stderr
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
