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


def evaluate(config,target,heading,seconds):
    model = mujoco.MjModel.from_xml_path(str(ROOT/'g1_description/scene.xml'))
    model.opt.timestep = .005
    data = mujoco.MjData(model)
    names = [model.joint(int(model.actuator_trnid[i,0])).name for i in range(model.nu)]
    state,output = StateAndCmd(29),PolicyOutput(29)
    config = dict(config,target_position=target,target_heading=heading)
    policy = ACP(state,output,config,names)
    fixed = FixedPose(state,output)
    fixed.set_runtime_target(policy.default,policy.kps,policy.kds)
    data.qpos[:3], data.qpos[7:36] = [0,0,.78],policy.default
    data.qpos[3:7] = [1,0,0,0]
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
                trajectory.append(np.r_[data.time,data.qpos[:3],data.qvel[:3],yaw(data.qpos[3:7])])
        torque = output.kps*(output.actions-data.qpos[7:36])-output.kds*data.qvel[6:35]
        data.ctrl[:] = np.clip(torque,-policy.effort_limits,policy.effort_limits)
        mujoco.mj_step(model,data)
        if not np.isfinite(data.qpos).all() or data.qpos[2] < .3:
            error = 'Robot fell or state became nonfinite'
            break
    distance = float(np.linalg.norm(data.qpos[:2]-target))
    heading_error = float(abs(np.arctan2(np.sin(heading-yaw(data.qpos[3:7])),np.cos(heading-yaw(data.qpos[3:7])))))
    tail = np.asarray(trajectory[-50:])
    stopped = bool(len(tail)==50 and np.max(np.linalg.norm(tail[:,4:6],axis=1)) < .15)
    return {'target':target,'heading':heading,'sim_seconds':float(data.time),'error':error,
        'final_position':data.qpos[:3].tolist(),'distance':distance,'heading_error':heading_error,
        'stopped_last_second':stopped,'passed':error is None and distance < .2 and heading_error < .25 and stopped},np.asarray(trajectory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=15.)
    parser.add_argument('--bundle',type=Path,help='Evaluate this explicit bundle in simulation')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/acp/mujoco_eval.json')
    args = parser.parse_args()
    config = load_acp_config()
    if args.bundle:
        config = dict(config,enabled=True,model_dir=str(args.bundle.expanduser().resolve(strict=True)))
    if not config['enabled']: raise SystemExit('Set LYD_DEPLOY_CONFIG=config/deploy_acp.yaml')
    results,paths = [],{}
    for index,(target,heading) in enumerate((([1.,0.],0.),([0.,1.],np.pi/2),([0.,0.],0.))):
        result,path = evaluate(config,target,heading,args.seconds)
        results.append(result)
        paths[f'scenario_{index}'] = path
        print(json.dumps(result),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    passed = all(result['passed'] for result in results)
    report = {'passed':passed,'scenarios':results,'bundle':config['model_dir'],
        'policy_sha256':hashlib.sha256((Path(config['model_dir'])/'policy.onnx').read_bytes()).hexdigest(),
        'criteria':{'distance':.2,'heading_error':.25,'speed_xy':.15,'hold_seconds':1.}}
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.output.with_suffix('.npz'),**paths)
    sys.exit(0 if passed else 1)
