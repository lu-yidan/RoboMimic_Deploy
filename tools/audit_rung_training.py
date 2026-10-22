"""Offline, sampled geometry and motor checks on residual-policy evaluations."""
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
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    summary=json.loads((a.run/'summary.json').read_text());manifest=json.loads((a.run/'manifest.json').read_text())
    source_ok=all(hashlib.sha256((a.run/'source_snapshot'/s).read_bytes()).hexdigest()==h for s,h in manifest['sources'].items())
    output=dict(source_hashes_match=source_ok,evaluations=[],note='50 Hz sampled geometry check, not a 2000 Hz contact-force replay.')
    for stage in ('zero','before','after'):
        for k,result in enumerate(summary[stage]):
            e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');e.reset()
            apply_rung_offsets(e,{'rung_05':result['offset_m']});z=np.load(a.run/f'{stage}_{k}.npz')
            unexpected=0.;flight=0.;first=None
            for t,q in zip(z['time_s'],z['qpos']):
                e.data.qpos[:]=q;mujoco.mj_kinematics(e.model,e.data);mujoco.mj_collision(e.model,e.data)
                for c in e.data.contact:
                    ba,bb=int(e.model.geom_bodyid[c.geom1]),int(e.model.geom_bodyid[c.geom2])
                    ia,ib=ba in e.robot_bodies,bb in e.robot_bodies
                    if not (ia or ib):continue
                    expected=ia!=ib and (ba if ia else bb) in e.contact_groups
                    if not expected:unexpected=max(unexpected,float(-c.dist))
                    if 5.005<=t<=12.-1e-7 and ia!=ib and (ba if ia else bb)==e.model.body('right_hook').id:
                        flight=max(flight,float(-c.dist))
                        if c.dist < -1e-5 and first is None:first=float(t)
            checks=dict(runtime_success=result['success'],duration=abs(z['time_s'][-1]-16.)<1e-7,
                        finite=bool(np.isfinite(z['qpos']).all() and np.isfinite(z['qvel']).all()),
                        torque_bounds=bool(np.all(z['torques']>=e.limits[:,0]-1e-5) and np.all(z['torques']<=e.limits[:,1]+1e-5)),
                        residual_action_bounds=bool(np.max(abs(z['actions']))<=1.+1e-7),
                        sampled_unexpected_penetration=unexpected<1e-5,sampled_clear_swing=flight<1e-5)
            checks={key:bool(value) for key,value in checks.items()}
            output['evaluations'].append(dict(stage=stage,case=k,passed=bool(all(checks.values()) and source_ok),checks=checks,
                        unexpected_penetration_m=unexpected,flight_penetration_m=flight,first_flight_penetration_s=first))
            e.close()
    (a.run/'independent_audit.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
