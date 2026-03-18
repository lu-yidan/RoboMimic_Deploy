import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.path_config import PROJECT_ROOT

import copy
import time
import mujoco.viewer
import mujoco
import numpy as np
import yaml
import os
from common.ctrlcomp import *
from FSM.FSM import *
from common.utils import get_gravity_orientation
from common.joystick import JoyStick, JoystickButton
from omegaconf import DictConfig
import hydra    



def quat_to_matrix(q):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),  2*(x*z + w*y)],
        [2*(x*y + w*z),  1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),  1 - 2*(x*x + y*y)],
    ])


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

@hydra.main(config_path="config", config_name="mujoco")
def main(cfg: DictConfig):
    xml_path = os.path.join(PROJECT_ROOT, cfg.xml_path)
    simulation_dt = cfg.simulation_dt
    control_decimation = cfg.control_decimation
    tau_limit = np.array(cfg.tau_limit)
        
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    torso_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    ball_body_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ball")  # -1 if no ball in scene

    # ---- Ghost model for reference motion visualization ----
    # Ghost color from config (RGBA, 0-1). Adjust ghost_rgba in mujoco.yaml.
    ghost_rgba = list(cfg.get("ghost_rgba", [0.2, 0.9, 0.4, 0.2]))
    ghost_m = copy.deepcopy(m)
    ghost_m.geom_rgba[:, :3] = ghost_rgba[:3]
    ghost_m.geom_rgba[:, 3]  = ghost_rgba[3]
    ghost_d = mujoco.MjData(ghost_m)
    # Use a list so the key_callback closure can mutate it.
    ghost_flags = [bool(cfg.get("ghost_flags[0]", True))]


    mj_per_step_duration = simulation_dt * control_decimation
    num_joints = m.nu
    print(f"num_joints: {num_joints}")
    policy_output_action = np.zeros(num_joints, dtype=np.float32)
    kps = np.zeros(num_joints, dtype=np.float32)
    kds = np.zeros(num_joints, dtype=np.float32)
    sim_counter = 0
    # Ball sensor runs at 10 Hz; control loop runs at 50 Hz → update every 5 control steps.
    control_hz = 1.0 / (simulation_dt * control_decimation)
    ball_sensor_decimation = max(1, round(control_hz / cfg.get("ball_sensor_hz", 10)))
    ball_sensor_counter = 0

    state_cmd = StateAndCmd(num_joints)
    policy_output = PolicyOutput(num_joints)
    FSM_controller = FSM(state_cmd, policy_output)

    joystick = JoyStick()
    prev_hat = (0, 0)
    Running = True
    with mujoco.viewer.launch_passive(m, d) as viewer:
        sim_start_time = time.time()
        while viewer.is_running() and Running:
            try:
                if(joystick.is_button_pressed(JoystickButton.SELECT)):
                    Running = False

                joystick.update()
                hat = joystick.get_hat_direction()
                hat_just_pressed = lambda hx, hy: (hat == (hx, hy) and prev_hat != (hx, hy))
                if joystick.is_button_released(JoystickButton.L3):                                                    # Ghost toggle, L3
                    ghost_flags[0] = not ghost_flags[0]
                    print(f"[Ghost] {'ON' if ghost_flags[0] else 'OFF'}")

                # PASSIVE: safety command — always overrides any pending command
                if joystick.is_button_released(JoystickButton.L1) and joystick.is_button_pressed(JoystickButton.R1):  # 阻尼保护, L1 release + R1
                    state_cmd.skill_cmd = FSMCommand.PASSIVE

                # All other skill commands: latched — only accepted when FSM has cleared the previous one.
                # This prevents a fast START → L1+Up sequence from overwriting POS_RESET before the FSM
                # processes it, which would leave the current controller without a clean exit().
                # elif state_cmd.skill_cmd == FSMCommand.INVALID:
                if joystick.is_button_released(JoystickButton.START):                                              # 回 FixedPose, START
                    state_cmd.skill_cmd = FSMCommand.POS_RESET
                elif joystick.is_button_released(JoystickButton.X) and joystick.is_button_pressed(JoystickButton.L1):   # 摔倒爬起, L1+X
                    state_cmd.skill_cmd = FSMCommand.STAND_UP
                elif joystick.is_button_released(JoystickButton.A) and joystick.is_button_pressed(JoystickButton.R1):   # Loco, R1+A
                    state_cmd.skill_cmd = FSMCommand.LOCO
                elif joystick.is_button_released(JoystickButton.X) and joystick.is_button_pressed(JoystickButton.R1):   # Dance, R1+X
                    state_cmd.skill_cmd = FSMCommand.SKILL_1
                elif joystick.is_button_released(JoystickButton.Y) and joystick.is_button_pressed(JoystickButton.R1):   # KungFu, R1+Y
                    state_cmd.skill_cmd = FSMCommand.SKILL_2
                elif joystick.is_button_released(JoystickButton.B) and joystick.is_button_pressed(JoystickButton.R1):   # Kick, R1+B
                    state_cmd.skill_cmd = FSMCommand.SKILL_3
                elif joystick.is_button_released(JoystickButton.Y) and joystick.is_button_pressed(JoystickButton.L1):   # KungFu2, L1+Y
                    state_cmd.skill_cmd = FSMCommand.SKILL_4
                elif joystick.is_button_released(JoystickButton.A) and joystick.is_button_pressed(JoystickButton.L1):   # ASAP, L1+A
                    state_cmd.skill_cmd = FSMCommand.SKILL_5
                elif joystick.is_button_released(JoystickButton.B) and joystick.is_button_pressed(JoystickButton.L1):   # BeyondMimic, L1+B
                    state_cmd.skill_cmd = FSMCommand.SKILL_6
                elif hat_just_pressed(1, 0) and joystick.is_button_pressed(JoystickButton.R1):                          # Score, R1+D-pad RIGHT
                    state_cmd.skill_cmd = FSMCommand.CMD_SCORE
                elif hat_just_pressed(0, -1) and joystick.is_button_pressed(JoystickButton.L1):                         # FallGetUpMJ, L1+D-pad DOWN
                    state_cmd.skill_cmd = FSMCommand.CMD_BEYONDMIMIC_MJ
                elif hat_just_pressed(0, 1) and joystick.is_button_pressed(JoystickButton.L1):                          # StandUpMJ, L1+D-pad UP
                    state_cmd.skill_cmd = FSMCommand.CMD_STANDUP_MJ

                prev_hat = hat
                state_cmd.vel_cmd[0] = -joystick.get_axis_value(1)
                state_cmd.vel_cmd[1] = -joystick.get_axis_value(0)
                state_cmd.vel_cmd[2] = -joystick.get_axis_value(3)
                
                step_start = time.time()
                
                tau = pd_control(policy_output_action, d.qpos[7:7+num_joints], kps, np.zeros_like(kps), d.qvel[6:6+num_joints], kds)
                tau = np.clip(tau, -tau_limit, tau_limit)
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)
                FSM_controller.sim_counter += 1
                if FSM_controller.sim_counter % control_decimation == 0:
                    
                    qj = d.qpos[7:7+num_joints]
                    dqj = d.qvel[6:6+num_joints]
                    quat = d.qpos[3:7]
                    
                    omega = d.qvel[3:6] 
                    gravity_orientation = get_gravity_orientation(quat)
                    
                    state_cmd.q = qj.copy()
                    state_cmd.dq = dqj.copy()
                    state_cmd.gravity_ori = gravity_orientation.copy()
                    state_cmd.ang_vel = omega.copy()

                    # Extra state for BeyondMimic policy
                    R_root = quat_to_matrix(quat)
                    state_cmd.root_lin_vel_b = (R_root.T @ d.qvel[0:3]).astype(np.float32)
                    state_cmd.root_ang_vel_b = d.qvel[3:6].astype(np.float32)  # body frame in MuJoCo
                    state_cmd.torso_pos_w  = d.xpos[torso_body_id].astype(np.float32)
                    state_cmd.torso_quat_w = d.xquat[torso_body_id].astype(np.float32)  # [w,x,y,z]
                    state_cmd.pelvis_pos_w  = d.qpos[0:3].astype(np.float32)
                    state_cmd.pelvis_quat_w = d.qpos[3:7].astype(np.float32)  # [w,x,y,z]

                    # Ball state (only valid when scene_with_ball.xml is loaded).
                    # Throttled to ball_sensor_hz to simulate real-sensor update rate.
                    if ball_body_id >= 0 and ball_sensor_counter % ball_sensor_decimation == 0:
                        state_cmd.ball_pos_w = d.xpos[ball_body_id].astype(np.float32)
                        ball_jnt_adr = m.body_jntadr[ball_body_id]
                        ball_qvel_adr = m.jnt_dofadr[ball_jnt_adr]
                        state_cmd.ball_vel_w = d.qvel[ball_qvel_adr:ball_qvel_adr+3].astype(np.float32)
                    ball_sensor_counter += 1

                    FSM_controller.run()
                    policy_output_action = policy_output.actions.copy()
                    kps = policy_output.kps.copy()
                    kds = policy_output.kds.copy()

                # ---- Ghost visualization ----
                if ghost_flags[0] and policy_output.ghost_qpos is not None:
                    ghost_d.qpos[:7 + num_joints] = policy_output.ghost_qpos
                    mujoco.mj_forward(ghost_m, ghost_d)
                    with viewer.lock():
                        viewer.user_scn.ngeom = 0
                        mujoco.mjv_addGeoms(
                            ghost_m, ghost_d,
                            mujoco.MjvOption(), mujoco.MjvPerturb(),
                            mujoco.mjtCatBit.mjCAT_DYNAMIC.value,
                            viewer.user_scn,
                        )
                else:
                    with viewer.lock():
                        viewer.user_scn.ngeom = 0

                viewer.sync()
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
            except ValueError as e:
                print(str(e))

if __name__ == "__main__":
    main()
            

        