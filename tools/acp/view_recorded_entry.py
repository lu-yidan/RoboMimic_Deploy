#!/usr/bin/env python3
"""View a recorded entry trace without inference, DDS, or robot control.

Blue pelvis marker: initialization support. Orange: released/ACP. Green/red
foot markers: recorded simulation normal force above/below 5 N. SPACE pauses,
R restarts. --video renders the same trajectory headlessly through ffmpeg.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load_trace(path):
    report = json.loads((path.parent/'results.json').read_text())
    row = next(r for r in report['results'] if r['trace_file']==path.name)
    if row.get('trace_sha256') and hashlib.sha256(path.read_bytes()).hexdigest()!=row['trace_sha256']:
        raise ValueError('Trace hash mismatch')
    with np.load(path,allow_pickle=False) as archive:
        trace = {name:archive[name].copy() for name in archive.files}
    if not np.isfinite(trace['qpos']).all() or np.any(np.diff(trace['time_s'])<=0):
        raise ValueError('Trace pose or time is invalid')
    return row,trace


def verify_model(report):
    for key,relative in (('scene_sha256','g1_description/scene.xml'),
                         ('robot_model_sha256','g1_description/g1_29dof_rev_1_0.xml')):
        if report.get(key) and hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=report[key]:
            raise ValueError('Replay model differs from the evaluated model: '+relative)


def markers(scene,model,data,trace,index,row):
    start = scene.ngeom
    def sphere(position,color,radius=.025):
        nonlocal start
        mujoco.mjv_initGeom(scene.geoms[start],mujoco.mjtGeom.mjGEOM_SPHERE,np.full(3,radius),
            np.asarray(position,dtype=float),np.eye(3).ravel(),np.asarray(color,dtype=np.float32))
        start+=1
    # Put the phase marker above the head so the torso does not occlude it.
    sphere(data.qpos[:3]+[0,0,.65], [0.1,.4,1.,.85] if trace['support'][index] else [1.,.55,.05,.85],.04)
    for side,name in enumerate(('left','right')):
        pos = data.xpos[model.body(name+'_ankle_roll_link').id].copy();pos[2]=.02
        sphere(pos,[.1,.9,.2,1.] if trace['foot_normal_force'][index,side]>5. else [1.,.15,.1,1.])
    # Goal is one metre forward/sideways from the canonical initial root.
    q = np.asarray(row['initial_root_pose'][3:7]);w,x,y,z = q
    psi = np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    offset = {'stop':[0.,0.],'forward':[np.cos(psi),np.sin(psi)],'turn':[-np.sin(psi),np.cos(psi)]}[row['scenario']]
    sphere([*offset,.03],[.9,.05,.8,.8],.04)
    scene.ngeom = start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',required=True,type=Path)
    p.add_argument('--speed',type=float,default=1.)
    p.add_argument('--loop',action='store_true')
    p.add_argument('--video',type=Path)
    p.add_argument('--fps',type=int,default=25)
    p.add_argument('--width',type=int,default=1280)
    p.add_argument('--height',type=int,default=720)
    p.add_argument('--ffmpeg',default='ffmpeg',help='ffmpeg executable with libx264, used only by --video')
    args = p.parse_args()
    if not np.isfinite(args.speed) or args.speed<=0:p.error('speed must be positive')
    if args.fps<1 or args.width<64 or args.height<64:p.error('Invalid video settings')
    row,trace = load_trace(args.trace.resolve(strict=True))
    verify_model(json.loads((args.trace.resolve().parent/'results.json').read_text()))
    model = mujoco.MjModel.from_xml_path(str(ROOT/'g1_description/scene.xml'))
    data = mujoco.MjData(model)
    def pose(index):
        data.qpos[:],data.qvel[:],data.time = trace['qpos'][index],trace['qvel'][index],trace['time_s'][index]
        mujoco.mj_forward(model,data)
    def camera(cam):
        cam.lookat[:] = [0.,0.,.5]
        cam.distance,cam.azimuth,cam.elevation = 3.4,135.,-18.
    if args.video:
        if args.video.exists():p.error('Video output already exists')
        args.video.parent.mkdir(parents=True,exist_ok=True)
        model.vis.global_.offwidth,model.vis.global_.offheight = args.width,args.height
        cam = mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam);camera(cam)
        renderer = mujoco.Renderer(model,height=args.height,width=args.width)
        command = [args.ffmpeg,'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pixel_format','rgb24',
            '-video_size',f'{args.width}x{args.height}','-framerate',str(args.fps),'-i','-','-an','-c:v','libx264',
            '-crf','20','-pix_fmt','yuv420p',str(args.video)]
        process = subprocess.Popen(command,stdin=subprocess.PIPE)
        try:
            for stamp in np.arange(trace['time_s'][0],trace['time_s'][-1]+1e-6,args.speed/args.fps):
                index = min(int(np.searchsorted(trace['time_s'],stamp)),len(trace['time_s'])-1)
                pose(index);renderer.update_scene(data,camera=cam)
                markers(renderer.scene,model,data,trace,index,row)
                process.stdin.write(renderer.render().tobytes())
        finally:
            process.stdin.close();renderer.close()
            code=process.wait()
        if code!=0:raise RuntimeError('ffmpeg rendering failed')
        print(str(args.video.resolve()))
        return
    from mujoco import viewer as mjviewer
    control = {'pause':False,'restart':False}
    def key(value):
        if value==32:control['pause']=not control['pause']
        elif value in (ord('R'),ord('r')):control['restart']=True
    with mjviewer.launch_passive(model,data,key_callback=key) as viewer:
        camera(viewer.cam)
        index=0
        while viewer.is_running():
            started=time.monotonic()
            if control['restart']:index=0;control['restart']=False
            pose(index)
            with viewer.lock():
                viewer.user_scn.ngeom=0
                markers(viewer.user_scn,model,data,trace,index,row)
            viewer.sync()
            if not control['pause']:
                if index<len(trace['time_s'])-1:index+=1
                elif args.loop:index=0
                else:control['pause']=True
            time.sleep(max(0.,.02/args.speed-(time.monotonic()-started)))


if __name__=='__main__':main()
