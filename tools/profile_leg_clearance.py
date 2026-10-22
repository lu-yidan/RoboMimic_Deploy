"""Measure signed leg-to-rung distances on reference or recorded states."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.climb_robustness import apply_rung_offsets


def main():
    p=argparse.ArgumentParser();source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--run',type=Path);source.add_argument('--plan',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--dt',type=float,default=.02)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset()
    if a.run:
        z=np.load(a.run/'trajectory.npz');ref=np.load(a.run/'reference.npz')
        times=z['time_s'];states=z['qpos'];path=a.run/'trajectory.npz'
        variant=json.loads((a.run/'scene_variant.json').read_text()) if (a.run/'scene_variant.json').exists() else {'offsets':{}}
        apply_rung_offsets(e,variant['offsets'])
    else:
        ref=np.load(a.plan);times=np.arange(0,float(ref['time'][-1])+1e-8,a.dt)
        states=np.stack([np.interp(times,ref['time'],ref['qpos'][:,i]) for i in range(e.model.nq)],axis=1)
        path=a.plan
    groups={}
    for side in ('left','right'):
        for part,words in [('leg',('hip','knee')),('foot',('ankle',))]:
            groups[side+'_'+part]=[i for i in range(e.model.ngeom) if e.model.geom_contype[i] and
                e.model.body(e.model.geom_bodyid[i]).name.startswith(side+'_') and
                any(w in e.model.body(e.model.geom_bodyid[i]).name for w in words)]
    bars=[e.model.geom(f'rung_{i:02d}').id for i in range(5)]
    records=[];last=-np.inf
    for t,q in zip(times,states):
        if t-last<a.dt-1e-7:continue
        last=t;e.data.qpos[:]=q;mujoco.mj_kinematics(e.model,e.data)
        j=min(np.searchsorted(ref['time'],t,side='right'),len(ref['time'])-1);phase=str(ref['phase'][j])
        for group,geoms in groups.items():
            if group.endswith('foot') and phase!=('lf_swing' if group.startswith('left') else 'rf_swing'):continue
            best=(.05,None,None)
            for g in geoms:
                for b in bars:
                    distance=mujoco.mj_geomDistance(e.model,e.data,g,b,.05,None)
                    if distance<best[0]:best=(float(distance),g,b)
            distance,g,b=best
            records.append(dict(time_s=float(t),phase=phase,group=group,distance_m=distance,
                                body=None if g is None else e.model.body(e.model.geom_bodyid[g]).name,
                                robot_geom_id=g,rung=None if b is None else e.model.geom(b).name))
    minima=[]
    for phase,group in dict.fromkeys((r['phase'],r['group']) for r in records):
        rows=[r for r in records if r['phase']==phase and r['group']==group]
        minima.append(min(rows,key=lambda r:r['distance_m']))
    result=dict(source=str(path.resolve()),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                sample_dt_s=a.dt,scope='Offline geometry, clipped at 50 mm. Feet checked only in swing, all hip/knee geoms checked throughout.',
                minima=minima)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2));(a.out/'samples.json').write_text(json.dumps(records))
    print(json.dumps(sorted(minima,key=lambda r:r['distance_m'])[:12],indent=2));e.close()


if __name__=='__main__':main()
