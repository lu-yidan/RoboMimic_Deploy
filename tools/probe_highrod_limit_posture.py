"""Scratch-only native-limit headroom IK at failed saved poses.

Bounds are deviations from the actual pose, NOT policy residual authority.
No state is injected into a live simulation and no dynamic success is implied.
"""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--lock-waist',action='store_true');args=parser.parse_args();args.output.mkdir(exist_ok=False)
    protocol=dict(cases=['true_height_12.5mm','true_height_15mm'],actual_joint_deviation_bound_rad=.015,base_translation_norm_bound_m=.005,
        right_ankle_lower_margin_rad=.005,position_tolerance_m=1e-5,rotation_tolerance_rad=1e-4,waist_locked=args.lock_waist,
        scope='Local IK preserves all four endpoint positions/orientations at the failed terminal pose; fixed base orientation. No dynamic motion, force balance, residual-command feasibility or global optimality claim.')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2));results=[]
    for label in protocol['cases']:
        p=Path('outputs/highrod_height_mismatch_v1')/label;m=json.loads((p/'manifest.json').read_text());a=json.loads((p/'independent_audit.json').read_text())
        assert a['integrity_passed'] and a['trajectory_sha256']==sha(p/'trajectory.npz')
        env=FullCycleResidualEnv(plan=m['reference'],cases={m['case']:m['offsets']});env.reset(options={'case':m['case']});e=env.env;model=e.model;d=mujoco.MjData(model)
        with np.load(p/'trajectory.npz') as z:q0=z['qpos'][-1].copy();time=float(z['time_s'][-1])
        d.qpos[:]=q0;mujoco.mj_kinematics(model,d)
        indices=env.indices if args.lock_waist else np.arange(len(e.joints))
        joints=e.joints[indices];qadrs=e.qadr[indices]
        bodies=[model.body(n).id for n in ('hook','right_hook','left_ankle_roll_link','right_ankle_roll_link')]
        target_pos=d.xpos[bodies].copy();target_quat=d.xquat[bodies].copy()
        low=np.r_[np.full(3,-.005/np.sqrt(3)),np.maximum(-.015,model.jnt_range[joints,0]+1e-7-q0[qadrs])]
        high=np.r_[np.full(3,.005/np.sqrt(3)),np.minimum(.015,model.jnt_range[joints,1]-1e-7-q0[qadrs])]
        ankle=int(np.flatnonzero(joints==model.joint('right_ankle_pitch_joint').id)[0]);low[3+ankle]=max(low[3+ankle],model.jnt_range[joints[ankle],0]+.005-q0[qadrs[ankle]])
        def errors(x):
            d.qpos[:]=q0;d.qpos[:3]+=x[:3];d.qpos[qadrs]+=x[3:];mujoco.mj_kinematics(model,d)
            rotation=[]
            for body,quat in zip(bodies,target_quat):
                delta=np.zeros(4);mujoco.mju_mulQuat(delta,d.xquat[body],quat*np.array([1,-1,-1,-1]));v=np.zeros(3);mujoco.mju_quat2Vel(v,delta,1.);rotation.append(v)
            return d.xpos[bodies]-target_pos,np.asarray(rotation)
        def objective(x):
            pos,rot=errors(x);return np.r_[pos.ravel(),.2*rot.ravel(),1e-6*x]
        candidates=[]
        for seed in (np.zeros(3+len(joints)),np.r_[[0.,0.,.002],np.zeros(len(joints))]):
            fit=least_squares(objective,np.clip(seed,low+1e-10,high-1e-10),bounds=(low,high),max_nfev=500,ftol=1e-12,xtol=1e-12,gtol=1e-12)
            pos,rot=errors(fit.x);mujoco.mj_collision(model,d);penetration=0.
            for c in d.contact:
                aa,bb=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2]);ia,ib=aa in e.robot_bodies,bb in e.robot_bodies
                if not (ia or ib):continue
                expected=ia!=ib and (aa if ia else bb) in e.contact_groups
                if not expected:penetration=max(penetration,float(-c.dist))
            pos_error=float(np.linalg.norm(pos,axis=1).max());rot_error=float(np.linalg.norm(rot,axis=1).max());margin=float(d.qpos[qadrs[ankle]]-model.jnt_range[joints[ankle],0])
            all_margin=float(np.minimum(d.qpos[e.qadr]-model.jnt_range[e.joints,0],model.jnt_range[e.joints,1]-d.qpos[e.qadr]).min())
            row=dict(optimizer_success=bool(fit.success),nfev=fit.nfev,position_error_m=pos_error,rotation_error_rad=rot_error,
                ankle_margin_rad=margin,maximum_actual_joint_deviation_rad=float(abs(fit.x[3:]).max()),base_translation_m=fit.x[:3].tolist(),
                unexpected_penetration_m=penetration,all_native_margin_rad=all_margin,kinematic_candidate=bool(pos_error<1e-5 and rot_error<1e-4 and margin>=.005-1e-9 and all_margin>=-1e-7 and penetration<1e-5))
            np.savez_compressed(args.output/f'{label}_seed{len(candidates)}.npz',original_qpos=q0,candidate_qpos=d.qpos.copy(),offset=fit.x,target_positions=target_pos,target_quaternions=target_quat)
            candidates.append(row)
        results.append(dict(case=m['case'],terminal_time_s=time,source_trajectory_sha256=sha(p/'trajectory.npz'),candidates=candidates));env.close()
    result=dict(results=results,source_sha256=sha(__file__),protocol_sha256=sha(args.output/'protocol.json'),scope=protocol['scope'])
    (args.output/'summary.json').write_text(json.dumps(result,indent=2));(args.output/'probe_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(results,indent=2))


if __name__=='__main__':main()
