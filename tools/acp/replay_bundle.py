#!/usr/bin/env python3
"""Rebuild all 939 deployment inputs from a native export rollout, then compare."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from policy.acp.observations import ObservationHistory, goal_command, grid_xy, rotation


def replay(directory, atol=1e-5, rtol=1e-4):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    for name, digest in (('policy.onnx',manifest['policy_sha256']),
                         ('parity.npz',manifest['parity']['sha256'])):
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Bundle hash mismatch: '+name)
    session = ort.InferenceSession(str(directory/'policy.onnx'),providers=['CPUExecutionProvider'])
    history = ObservationHistory()
    errors = {'observation':0.,'actions':0.,'contacts':0.,'occupancy':0.}
    with np.load(directory/'parity.npz',allow_pickle=False) as records:
        for tick, source in enumerate(records['planner_obs']):
            if tick == 0 or bool(records['done'][tick-1].any()):
                history.reset()
            quat = records['pelvis_quat_wxyz'][tick,0]
            angular = rotation(quat).T @ records['pelvis_ang_vel_w'][tick,0]
            proprio = history.append(angular,quat,
                records['joint_pos'][tick,0]-manifest['default_joint_pos'],
                records['joint_vel'][tick,0],records['last_action'][tick,0])
            hits = records['terrain_hit_pos_w'][tick,0]
            height = records['terrain_frame_pos_w'][tick,0,0,2]-hits[:,2]-.5
            height = height.reshape(11,17)
            miss = records['terrain_distances'][tick,0].reshape(11,17) < 0
            # Match the training edge-miss repair from independent sensor hits.
            repaired = height.copy()
            for y,x in zip(*np.where(miss)):
                ys,xs = slice(max(0,y-1),min(11,y+2)),slice(max(0,x-1),min(17,x+2))
                neighbours, valid = height[ys,xs], ~miss[ys,xs]
                threshold = manifest['scan_miss_fill_valid_fraction']*(valid.size-1)
                repaired[y,x] = np.min(neighbours[valid]) if valid.sum() >= threshold else manifest['scan_miss_height']
            points = np.column_stack((grid_xy(),repaired.ravel()))
            goal = goal_command(records['pelvis_pos_w'][tick,0],quat,
                records['target_pos_w'][tick,0,:2],float(records['target_yaw_w'][tick,0]),manifest['target_radius'])
            reconstructed = np.concatenate((proprio,points.ravel(),goal)).astype(np.float32)[None]
            np.testing.assert_allclose(reconstructed,source,atol=atol,rtol=rtol,err_msg=f'939-input reconstruction frame {tick}')
            errors['observation'] = max(errors['observation'],float(np.max(np.abs(reconstructed-source))))
            outputs = session.run(['actions','contacts','occupancy'],{'planner_obs':reconstructed})
            for name, actual in zip(('actions','contacts','occupancy'),outputs):
                native = records['native_'+name][tick]
                np.testing.assert_allclose(actual,native,atol=atol,rtol=rtol,err_msg=f'{name} frame {tick}')
                errors[name] = max(errors[name],float(np.max(np.abs(actual-native))))
    return {'frames':tick+1,'passed':True,'max_abs_error':errors}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path)
    parser.add_argument('--output',type=Path,help='Write the validation receipt as JSON')
    args = parser.parse_args()
    result = replay(args.bundle)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
