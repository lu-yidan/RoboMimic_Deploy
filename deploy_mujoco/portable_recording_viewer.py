"""Standalone recorded-state viewer: NumPy + MuJoCo, no project imports."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import mujoco
import numpy as np


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load(package):
    package=package.resolve();manifest=json.loads((package/'package.json').read_text(encoding='utf8'))
    if mujoco.__version__!=manifest['mujoco_version']:
        raise ValueError('Use the pinned MuJoCo version in requirements.txt for this compiled model')
    for name,expected in manifest['files'].items():
        file=(package/name).resolve()
        if not file.is_relative_to(package) or digest(file)!=expected:
            raise ValueError('Package file missing or changed: '+name)
    audit=json.loads((package/'evidence/v2_multicycle_audit.json').read_text())
    if not audit['passed'] or not audit['physical_passed'] or not audit['policy_passed']:
        raise ValueError('Passed physical and policy audits required')
    for key,name in [('trajectory_sha256','trajectory.npz'),('manifest_sha256','evidence/manifest.json'),
                     ('summary_sha256','evidence/summary.json'),('independent_audit_sha256','evidence/independent_audit.json'),
                     ('policy_audit_sha256','evidence/policy_transfer_audit.json')]:
        if audit[key]!=digest(package/name):raise ValueError('Evidence chain mismatch: '+name)
    model=mujoco.MjModel.from_binary_path(str(package/'model.mjb'));data=mujoco.MjData(model)
    with np.load(package/'trajectory.npz',allow_pickle=False) as trace:
        records={k:trace[k].copy() for k in ('qpos','qvel','time_s')}
    with np.load(package/'reference.npz',allow_pickle=False) as ref:ends=ref['cycle_ends_s'].copy()
    n=len(records['time_s'])
    if records['qpos'].shape!=(n,model.nq) or records['qvel'].shape!=(n,model.nv):
        raise ValueError('Model/recording dimensions differ')
    if not all(np.isfinite(v).all() for v in records.values()):raise ValueError('Nonfinite recording')
    if not np.allclose(np.diff(records['time_s']),.02,rtol=0,atol=1e-8):raise ValueError('Incomplete recording cadence')
    if abs(records['time_s'][-1]-ends[-1])>1e-6:raise ValueError('Incomplete recorded horizon')
    return model,data,records,ends,manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--verify-only',action='store_true');p.add_argument('--speed',type=float,default=8.)
    p.add_argument('--loop',action='store_true');a=p.parse_args()
    if not np.isfinite(a.speed) or not .1<=a.speed<=20:raise ValueError('Speed must be within 0.1..20')
    model,data,r,ends,manifest=load(a.package)
    report=dict(verified=True,case=manifest['case'],cycles=len(ends),horizon_s=float(r['time_s'][-1]),
        root_rise_m=float(r['qpos'][-1,2]-r['qpos'][0,2]),mode='Recorded kinematic replay only; no live physics, policy inference, or hardware connection.')
    print(json.dumps(report),flush=True)
    if a.verify_only:return
    import mujoco.viewer
    pelvis=model.body('pelvis').id;state=dict(paused=False,restart=False)
    def key_callback(key):
        if key==32:state['paused']=not state['paused']
        elif key in (ord('R'),ord('r')):state['restart']=True
    for geom in range(model.ngeom):
        if (model.geom(geom).name or '').startswith('rung_'):model.geom_rgba[geom]=[.55,.63,.68,1.]
    with mujoco.viewer.launch_passive(model,data,key_callback=key_callback) as viewer:
        viewer.opt.geomgroup[3],viewer.opt.geomgroup[4]=0,1
        viewer.cam.azimuth,viewer.cam.elevation,viewer.cam.distance=65.,-8.,2.5
        cursor=0.;previous=time.perf_counter()
        while viewer.is_running():
            now=time.perf_counter()
            if state['restart']:cursor=0.;state.update(paused=False,restart=False)
            elif not state['paused']:cursor+=(now-previous)*a.speed
            previous=now
            if cursor>=r['time_s'][-1]:
                if a.loop:cursor%=float(r['time_s'][-1])
                else:cursor=float(r['time_s'][-1]);state['paused']=True
            i=min(int(np.searchsorted(r['time_s'],cursor)),len(r['time_s'])-1)
            with viewer.lock():
                data.qpos[:]=r['qpos'][i];data.qvel[:]=r['qvel'][i];data.time=float(r['time_s'][i])
                mujoco.mj_kinematics(model,data);viewer.cam.lookat[:]=data.xpos[pelvis]+np.array([0,0,.05])
            cycle=min(int(np.searchsorted(ends,cursor,side='right'))+1,len(ends))
            viewer.set_texts((None,None,'RECORDED REPLAY - not live simulation\nTime / cycle / root rise\nControls',
                f'{a.speed:g}x | {"PAUSED" if state["paused"] else "PLAYING"}\n{cursor:.2f} s / {cycle} of {len(ends)} / {r["qpos"][i,2]-r["qpos"][0,2]:.3f} m\nSpace: pause | R: restart | Mouse: camera'))
            viewer.sync();time.sleep(max(0.,1./60.-(time.perf_counter()-now)))


if __name__=='__main__':main()
