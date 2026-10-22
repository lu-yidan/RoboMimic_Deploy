"""Kinematic endpoint candidates after a measured left-foot search depth.

Not a dynamic result; preserves both hand targets and shifts both feet by the
measured depth. Scans body-height compensation before choosing a dynamic test.
"""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.climb_cycle import LimbIK


def main():
    e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');ik=LimbIK(e)
    plan=np.load(ROOT/'outputs/clearance_back20_leg14_plan_v1/plan.npz');q=plan['qpos'][-1].copy()
    ik.d.qpos[:]=q;mujoco.mj_kinematics(e.model,ik.d)
    positions={k:ik.d.xpos[b].copy() for k,b in ik.bodies.items()}
    orientations={k:ik.d.xquat[b].copy() for k,b in ik.bodies.items()}
    results=[]
    for depth in (.0041,.0086):
        targets={k:p-np.array([0,0,depth if k in ('lf','rf') else 0]) for k,p in positions.items()}
        for ratio in (0.,.5,.75,1.,1.25):
            seed=q.copy();seed[2]-=depth*ratio;seed[0]-=np.tan(np.deg2rad(5.))*depth*ratio
            fitted,errors=ik.pose(seed,targets,orientations=orientations)
            delta=fitted[e.qadr]-plan['qpos'][0,e.qadr]
            results.append(dict(depth_m=depth,body_ratio=ratio,max_joint_difference_rad=float(max(abs(delta))),
                leg_difference_rad=float(max(abs(delta[:12]))),arm_difference_rad=float(max(abs(delta[15:]))),
                max_ik_error=float(max(errors.values())),base_rise_m=float(fitted[2]-plan['qpos'][0,2])))
    out=ROOT/'outputs/landing_endpoint_candidates_v1.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))


if __name__=='__main__':main()
