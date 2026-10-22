"""Report native joint margins and phase-wise recorded torque utilization."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from training.wall_support_env import WallSupportEnv


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.run/'manifest.json').read_text())
    e=WallSupportEnv(manifest['bundle']);z=np.load(a.run/'trajectory.npz')
    rows=list(csv.DictReader((a.run/'control_log.csv').open(encoding='utf-8')))
    names=[e.model.joint(j).name for j in e.joints]
    limits=e.model.jnt_range[e.joints];q=z['qpos'][:,e.qadr]
    margin=np.minimum(q-limits[:,0],limits[:,1]-q)
    i,j=np.unravel_index(np.argmin(margin),margin.shape)
    scale=np.maximum(abs(e.limits[:,0]),abs(e.limits[:,1]));utilization=abs(z['torques'])/scale
    phases=[]
    for phase in dict.fromkeys(r['phase'] for r in rows):
        selected=np.array([r['phase']==phase for r in rows]);u=utilization[selected]
        phases.append(dict(phase=phase,samples=int(selected.sum()),
            torque_peak_ratio={n:float(v) for n,v in zip(names,u.max(axis=0))},
            torque_near_limit_fraction={n:float(v) for n,v in zip(names,(u>.99).mean(axis=0))},
            mean_contact_vertical_n={g:float(np.mean([float(r[g+'_vertical_n']) for r in rows if r['phase']==phase])) for g in ['left_foot','right_foot','left_hook','right_hook']}))
    result=dict(scope='50 Hz recorded samples, not a continuous-time limit certificate',
        source_sha256={name:hashlib.sha256((a.run/name).read_bytes()).hexdigest() for name in ['trajectory.npz','control_log.csv']},
        minimum_native_joint_margin_rad=float(margin[i,j]),minimum_margin_joint=names[j],minimum_margin_time_s=float(z['time_s'][i]),
        phases=phases)
    with (a.run/'load_profile.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ['minimum_native_joint_margin_rad','minimum_margin_joint','minimum_margin_time_s']}));e.close()


if __name__=='__main__':main()
