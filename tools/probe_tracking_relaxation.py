"""Evaluate discrete tracking demands at an unchanged recorded failure state."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.climb_robustness import apply_rung_offsets
from training.climb_cycle import CycleTorqueController
from training.tracking_relaxation import scaled_reference


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset()
    apply_rung_offsets(e,json.loads((a.run/'scene_variant.json').read_text())['offsets'])
    f=np.load(a.run/'failure_state.npz');results=[]
    for scale in [1.,.75,.5,.25,0.]:
        mujoco.mj_resetData(e.model,e.data);e.data.qpos[:]=f['qpos'];e.data.qvel[:]=f['qvel'];mujoco.mj_forward(e.model,e.data)
        q,v=scaled_reference(e,f['qref'],f['vref'],scale)
        try:torque,info=CycleTorqueController(e,torque_weight=10000.).command(q,v,f['active'])
        except RuntimeError as exc:results.append(dict(scale=scale,feasible=False,reason=str(exc)))
        else:results.append(dict(scale=scale,feasible=True,torque_ratio=info['torque_ratio']))
    with a.out.open('x') as out:json.dump(dict(scope='Instantaneous allocation only, no dynamics or recovery claim',results=results),out,indent=2)
    print(json.dumps(results,indent=2));e.close()


if __name__=='__main__':main()
