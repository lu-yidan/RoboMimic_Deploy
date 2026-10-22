"""Identify which native joints cause endpoint closure rejection."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from training.wall_support_env import WallSupportEnv


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.run/'manifest.json').read_text());e=WallSupportEnv(manifest['bundle'])
    z=dict(np.load(a.run/'trajectory.npz'));ref=dict(np.load(a.run/'reference.npz'))
    ids=np.flatnonzero(z['time_s']>ref['time'][-1]-2.)
    if len(ids)<99:raise ValueError('Full endpoint window required')
    errors=z['qpos'][ids][:,e.qadr]-ref['qpos'][0,e.qadr]
    rows=[]
    for j,joint in enumerate(e.joints):
        i=int(np.argmax(abs(errors[:,j])))
        rows.append(dict(joint=e.model.joint(joint).name,peak_error_rad=float(errors[i,j]),peak_time_s=float(z['time_s'][ids[i]]),final_error_rad=float(errors[-1,j]),native_range_rad=e.model.jnt_range[joint].tolist()))
    rows.sort(key=lambda r:abs(r['peak_error_rad']),reverse=True)
    result=dict(joints=rows,body_position_delta_m=(z['qpos'][-1,:3]-ref['qpos'][-1,:3]).tolist(),threshold_rad=.04,scope='Endpoint tracking diagnosis only; no tolerance change, geometric feasibility or corrective control proof.')
    (a.run/'endpoint_joint_diagnostic.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(result,joints=rows[:8]),indent=2));e.close()


if __name__=='__main__':main()
