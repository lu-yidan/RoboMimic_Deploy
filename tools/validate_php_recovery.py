"""Reproduce Passive -> Y/SMP on the external PHP course, using the real FSM.

No viewer, gamepad, DDS or robot connection. Stops on warnings, reset or floor
penetration, rather than treating MuJoCo's automatic reset as successful recovery.
"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
import onnxruntime as ort
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.utils import FSMCommand, get_gravity_orientation
from deploy_mujoco.scene_builder import load_model, apply_initial_state
from FSM.FSM import FSM


def run_case(trigger, duration=30.):
    model = load_model(ROOT/'g1_description/scene.xml', {'name':'php_terrain'})
    model.opt.timestep = .002
    data = mujoco.MjData(model)
    apply_initial_state(model, data, {})
    state, output = StateAndCmd(29), PolicyOutput(29)
    options_type = ort.SessionOptions
    def options():
        options = options_type()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        return options
    with patch.object(ort, 'SessionOptions', options):
        fsm = FSM(state, output, enable_freekick=False)
    target, kp, kd = np.zeros(29), np.zeros(29), np.zeros(29)
    limits = np.array([88,139,88,139,50,50]*2 + [88,50,50] + [25,25,25,25,25,5,5]*2)
    failure, peak_speed, min_height = None, 0., float(data.qpos[2])
    watch = (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQVEL,
             mujoco.mjtWarning.mjWARN_BADQPOS)
    for step in range(round(duration/.002)):
        if step == round(trigger/.002):
            state.skill_cmd = FSMCommand.CMD_SMP_RECOVERY
        data.ctrl[:] = np.clip(kp*(target-data.qpos[7:36])-kd*data.qvel[6:35], -limits, limits)
        previous_time = data.time
        mujoco.mj_step(model,data)
        fsm.sim_counter += 1
        peak_speed = max(peak_speed, float(np.max(np.abs(data.qvel))))
        min_height = min(min_height, float(data.qpos[2]))
        if any(data.warning[w].number for w in watch) or data.time <= previous_time:
            failure = 'MuJoCo state warning/automatic reset'
            break
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or data.qpos[2] < -.1:
            failure = 'nonfinite state or penetration through source ground'
            break
        if fsm.sim_counter % 10 == 0:
            state.q = data.qpos[7:36].copy()
            state.dq = data.qvel[6:35].copy()
            state.ang_vel = data.qvel[3:6].copy()
            state.root_ang_vel_b = state.ang_vel.copy()
            state.gravity_ori = get_gravity_orientation(data.qpos[3:7])
            state.torso_quat_w = data.body('torso_link').xquat.copy()
            fsm.run()
            target, kp, kd = output.actions.copy(), output.kps.copy(), output.kds.copy()
    return dict(trigger_s=trigger, elapsed_s=(step+1)*.002, final_height=float(data.qpos[2]),
                min_height=min_height, peak_speed=peak_speed, failure=failure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    results = [run_case(t) for t in (.02,1.,3.,5.,10.,11.)]
    report = json.dumps(results, indent=2)
    print(report)
    if args.out:
        args.out.write_text(report+'\n')
    sys.exit(1 if any(r['failure'] for r in results) else 0)
