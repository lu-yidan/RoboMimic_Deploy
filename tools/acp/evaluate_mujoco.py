#!/usr/bin/env python3
"""Headless Stage 3 reach/turn/stop evaluation in the PHP deployment G1 model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from common.ctrlcomp import StateAndCmd,PolicyOutput
from common.deployment_config import load_acp_config
from policy.acp.ACP import ACP
from policy.acp.observations import yaw
from policy.acp.terrain import MujocoTerrain
from policy.fixedpose.FixedPose import FixedPose


# Flat Stage 3 targets are surface_z + BASE_STAND_HEIGHT in InstinctMJ
# goal_terrains.py/reset_goal_envs. This scene's plane surface is z=0.
BASE_STAND_HEIGHT = .78
CONTROL_DT = .02


def final_hold_result(trajectory, target, heading, target_height=BASE_STAND_HEIGHT):
    """Require every gate for the final 50 completed 50 Hz control periods.

    Rows: time, pelvis xyz, world linear velocity xyz, yaw, upright R_zz.
    """
    tail = np.asarray(trajectory, dtype=np.float64)[-round(1./CONTROL_DT):]
    result = {'stopped_last_second':False, 'held_all_criteria_last_second':False}
    if tail.shape != (round(1./CONTROL_DT), 9) or not np.isfinite(tail).all():
        return result
    distance = np.linalg.norm(tail[:,1:3]-np.asarray(target), axis=1)
    heading_error = np.abs(np.arctan2(np.sin(heading-tail[:,7]),np.cos(heading-tail[:,7])))
    speed = np.linalg.norm(tail[:,4:6], axis=1)
    height_error = np.abs(tail[:,3]-target_height)
    result['stopped_last_second'] = bool(np.all(speed < .15))
    result['held_all_criteria_last_second'] = bool(np.all(
        (distance < .2)&(heading_error < .25)&(speed < .15)&(height_error < .1)&(tail[:,8] > np.cos(.7))))
    return result


def final_pose_metrics(qpos, target, heading):
    """Keep a failed nonfinite simulation serializable as strict JSON."""
    position = [float(value) if np.isfinite(value) else None for value in qpos[:3]]
    distance = float(np.linalg.norm(qpos[:2]-target)) if np.isfinite(qpos[:2]).all() else None
    try:
        psi = yaw(qpos[3:7])
        heading_error = float(abs(np.arctan2(np.sin(heading-psi),np.cos(heading-psi))))
    except ValueError:
        heading_error = None
    return {'final_position':position,'distance':distance,'heading_error':heading_error}


def evaluate(config,target,heading,seconds,support_during_fixedpose=False):
    if not np.isfinite(seconds) or seconds < 1.:
        raise ValueError('Evaluation duration must be finite and at least one second')
    model = mujoco.MjModel.from_xml_path(str(ROOT/'g1_description/scene.xml'))
    model.opt.timestep = .005
    data = mujoco.MjData(model)
    names = [model.joint(int(model.actuator_trnid[i,0])).name for i in range(model.nu)]
    state,output = StateAndCmd(29),PolicyOutput(29)
    config = dict(config,entry_mode='fixed_pose',target_position=target,target_heading=heading)
    policy = ACP(state,output,config,names)
    fixed = FixedPose(state,output)
    fixed.set_runtime_target(policy.default,policy.kps,policy.kds)
    data.qpos[:3], data.qpos[7:36] = [0,0,BASE_STAND_HEIGHT],policy.default
    data.qpos[3:7] = [1,0,0,0]
    initial_root_pose = data.qpos[:7].copy()
    mujoco.mj_forward(model,data)
    state.q[:] = data.qpos[7:36]
    fixed.enter()
    terrain = MujocoTerrain(model)
    trajectory = []
    error,entered = None,False
    for tick in range(int((seconds+2)/.005)):
        if tick % 4 == 0:
            state.q[:] = data.qpos[7:36]
            state.dq[:] = data.qvel[6:35]
            terrain.capture(data,state)
            if tick < 400:
                fixed.run()
            else:
                if not entered:
                    error = policy.entry_error()
                    if error: break
                    policy.enter()
                    entered = True
                policy.run()
                if policy.failed:
                    error = policy.last_error
                    break
        if not all(np.isfinite(value).all() for value in (output.actions,output.kps,output.kds)):
            error = 'Nonfinite PD output'
            break
        torque = output.kps*(output.actions-data.qpos[7:36])-output.kds*data.qvel[6:35]
        if not np.isfinite(torque).all():
            error = 'Nonfinite PD torque'
            break
        data.ctrl[:] = np.clip(torque,-policy.effort_limits,policy.effort_limits)
        mujoco.mj_step(model,data)
        if support_during_fixedpose and not entered:
            # Explicit initialization fixture, equivalent to holding the base
            # while joint PD settles. Release completely before ACP inference;
            # never reset or constrain the free root during the policy test.
            data.qpos[:7] = initial_root_pose
            data.qvel[:6] = 0.
            data.qacc_warmstart[:6] = 0.
            mujoco.mj_forward(model,data)
        if not all(np.isfinite(value).all() for value in (data.qpos,data.qvel)) or data.qpos[2] < .3:
            error = 'Robot fell or state became nonfinite'
            break
        if entered and (tick+1) % 4 == 0:
            q = data.qpos[3:7]
            upright = 1-2*(q[1]**2+q[2]**2)
            trajectory.append(np.r_[data.time,data.qpos[:3],data.qvel[:3],yaw(q),upright])
    metrics = final_pose_metrics(data.qpos,target,heading)
    if metrics['distance'] is None or metrics['heading_error'] is None:
        error = error or 'Invalid final pelvis pose'
    hold = final_hold_result(trajectory,target,heading)
    return {'target':target,'heading':heading,'sim_seconds':float(data.time),'error':error,
        'policy_entered':entered,'failure_phase':None if error is None else ('acp' if entered else 'fixedpose'),
        'support_during_fixedpose':bool(support_during_fixedpose),
        'unassisted_acp_seconds':len(trajectory)*CONTROL_DT,
        **metrics,
        'target_height':BASE_STAND_HEIGHT, **hold,
        'passed':error is None and hold['held_all_criteria_last_second']},np.asarray(trajectory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=15.)
    parser.add_argument('--bundle',type=Path,help='Evaluate this explicit bundle in simulation')
    parser.add_argument('--support-during-fixedpose',action='store_true',
                        help='Hold the pelvis only during the two-second initialization; release before the complete unassisted ACP test')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/acp/mujoco_eval.json')
    args = parser.parse_args()
    config = load_acp_config()
    if args.bundle:
        config = dict(config,enabled=True,model_dir=str(args.bundle.expanduser().resolve(strict=True)))
    if not config['enabled']: raise SystemExit('Set LYD_DEPLOY_CONFIG=config/deploy_acp.yaml')
    results,paths = [],{}
    for index,(target,heading) in enumerate((([1.,0.],0.),([0.,1.],np.pi/2),([0.,0.],0.))):
        result,path = evaluate(config,target,heading,args.seconds,args.support_during_fixedpose)
        results.append(result)
        paths[f'scenario_{index}'] = path
        print(json.dumps(result),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    passed = all(result['passed'] for result in results)
    report = {'passed':passed,'scenarios':results,'bundle':config['model_dir'],
        'policy_sha256':hashlib.sha256((Path(config['model_dir'])/'policy.onnx').read_bytes()).hexdigest(),
        'criteria':{'distance':.2,'heading_error':.25,'speed_xy':.15,'height_error':.1,
                    'target_height':BASE_STAND_HEIGHT,'max_tilt_rad':.7,'hold_seconds':1.,
                    'test_seconds':float(args.seconds),
                    'support_during_fixedpose':bool(args.support_during_fixedpose),
                    'support_during_acp':False,
                    'success_semantics':'all criteria hold throughout the final second of the complete test'}}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(args.output.with_suffix('.npz'),**paths)
    sys.exit(0 if passed else 1)
