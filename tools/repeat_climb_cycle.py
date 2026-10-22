"""Build and audit a translated multi-cycle reference; never reset live physics."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from training.wall_support_env import WallSupportEnv
from tools.plan_climb_cycle import audit_plan

ROOT=Path(__file__).resolve().parents[1]


def repeat_reference(source,count):
    if count<2 or count!=int(count):raise ValueError('At least two whole cycles required')
    q=source['qpos'];t=source['time'];shift=q[-1,:3]-q[0,:3]
    if not np.allclose(shift,[.25*np.tan(np.deg2rad(5)),0,.25],atol=1e-7):
        raise ValueError('Reference must ascend one nominal 25 cm rung')
    if not np.allclose(q[-1,3:],q[0,3:],atol=1e-4):
        raise ValueError('Reference joint posture must close before repetition')
    if t[0]!=0 or np.any(np.diff(t)<=0):raise ValueError('Reference time must be strictly increasing from zero')
    result={k:[] for k in ['qpos','time','phase','active','errors']}
    for k in range(count):
        start=0 if k==0 else 1
        pose=q[start:].copy();pose[:,:3]+=k*shift
        result['qpos'].append(pose);result['time'].append(t[start:]+k*t[-1])
        for name in ['phase','active','errors']:result[name].append(source[name][start:])
    result={k:np.concatenate(v) for k,v in result.items()}
    result.update(initial=q[0].copy(),cycle_ends_s=np.arange(1,count+1)*t[-1],cycle_rise_m=np.array(.25))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--count',type=int,default=2);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    parent=json.loads((a.source.parent/'manifest.json').read_text());digest=hashlib.sha256(a.source.read_bytes()).hexdigest()
    if parent['plan_sha256']!=digest or not json.loads((a.source.parent/'audit.json').read_text())['passed']:
        raise ValueError('Parent must match a passed reference audit')
    result=repeat_reference(np.load(a.source),a.count)
    e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset()
    audit=audit_plan(e,result);a.out.mkdir(parents=True)
    np.savez_compressed(a.out/'plan.npz',**result)
    (a.out/'audit.json').write_text(json.dumps(audit,indent=2))
    sources=['tools/repeat_climb_cycle.py','tools/plan_climb_cycle.py']
    for src in sources:
        dest=a.out/'source_snapshot'/src;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/src,dest)
    (a.out/'manifest.json').write_text(json.dumps(dict(parent_plan=str(a.source),parent_sha256=digest,
        plan_sha256=hashlib.sha256((a.out/'plan.npz').read_bytes()).hexdigest(),
        bundle_manifest_sha256=hashlib.sha256((e.bundle/'manifest.json').read_bytes()).hexdigest(),
        sources={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
        cycles=a.count,scope='Translated reference with dense scene collision audit; requires uninterrupted dynamics validation'),indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k!='failures'},indent=2));print('Failures',len(audit['failures']));e.close()


if __name__=='__main__':main()
