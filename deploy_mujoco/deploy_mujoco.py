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
from common.utils import get_gravity_orientation, FSMStateName
from common.logger import Logger
from common.joystick import JoyStick, JoystickButton
from omegaconf import DictConfig
import hydra    


def _draw_viz_spheres(scn, viz_spheres):
    """Draw debug spheres/lines into a MuJoCo mjvScene.

    Each entry in viz_spheres is a dict with either:
      - sphere: {"pos": (3,), "radius": float, "rgba": (4,)}
      - line:   {"from": (3,), "to": (3,), "radius": float, "rgba": (4,)}
    """
    if not viz_spheres:
        return
    for item in viz_spheres:
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        rgba = np.asarray(item.get("rgba", [1.0, 1.0, 1.0, 1.0]), dtype=np.float32)
        if "from" in item:
            r = float(item.get("radius", 0.005))
            mujoco.mjv_makeConnector(
                g, mujoco.mjtGeom.mjGEOM_CAPSULE, r,
                *item["from"], *item["to"],
            )
            g.rgba[:] = rgba
        else:
            pos = item.get("pos", None)
            if pos is None:
                continue
            if "size" in item:
                size = np.asarray(item["size"], dtype=np.float64)
                if size.size == 1:
                    size = np.array([size.item(), size.item(), size.item()], dtype=np.float64)
                elif size.size != 3:
                    continue
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_BOX,
                    size,
                    np.asarray(pos, dtype=np.float64),
                    np.eye(3, dtype=np.float64).flatten(),
                    rgba,
                )
            else:
                r = float(item.get("radius", 0.03))
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([r, r, r], dtype=np.float64),
                    np.asarray(pos, dtype=np.float64),
                    np.eye(3, dtype=np.float64).flatten(),
                    rgba,
                )
        scn.ngeom += 1


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


def _reset_ball_state(m, d, ball_body_id, pos_w, vel_w, quat_w=None, ang_vel_w=None):
    """Reset the free-joint ball pose/velocity in-place."""
    if ball_body_id < 0 or m.body_jntnum[ball_body_id] <= 0:
        return False

    ball_jnt_id = m.body_jntadr[ball_body_id]
    qpos_adr = m.jnt_qposadr[ball_jnt_id]
    qvel_adr = m.jnt_dofadr[ball_jnt_id]

    quat_w = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) if quat_w is None else np.asarray(quat_w, dtype=np.float64)
    ang_vel_w = np.zeros(3, dtype=np.float64) if ang_vel_w is None else np.asarray(ang_vel_w, dtype=np.float64)

    d.qpos[qpos_adr:qpos_adr+3] = np.asarray(pos_w, dtype=np.float64)
    d.qpos[qpos_adr+3:qpos_adr+7] = quat_w
    d.qvel[qvel_adr:qvel_adr+3] = np.asarray(vel_w, dtype=np.float64)
    d.qvel[qvel_adr+3:qvel_adr+6] = ang_vel_w
    mujoco.mj_forward(m, d)
    return True

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
    ball_reset_pos_w = np.array(cfg.get("ball_reset_pos_w", [1.0, 0.0, 0.115]), dtype=np.float64)
    ball_reset_vel_w = np.array(cfg.get("ball_reset_vel_w", [0.0, 0.0, 0.0]), dtype=np.float64)
    ball_reset_quat_w = np.array(cfg.get("ball_reset_quat_w", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
    ball_reset_ang_vel_w = np.array(cfg.get("ball_reset_ang_vel_w", [0.0, 0.0, 0.0]), dtype=np.float64)

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

    # PHP parkour: bind the MjModel so the policy can build its joint→actuator map.
    if hasattr(FSM_controller, "php_parkour_policy"):
        FSM_controller.php_parkour_policy.bind_model(m)

    # PHP parkour: offscreen depth renderer for the chin-mounted camera.
    # Created lazily on first request to avoid GL init when PHP is never used.
    # Uses the XML-defined <camera name="php_depth"> attached to torso_link,
    # so the view rolls with the body (which MjvCamera free mode couldn't do).
    php_depth_ctx = {"renderer": None, "fixed_cam_id": -1}
    def _php_init_depth():
        if php_depth_ctx["renderer"] is not None:
            return
        if not hasattr(FSM_controller, "php_parkour_policy"):
            return
        php = FSM_controller.php_parkour_policy
        dcfg = php.depth_cfg
        cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "php_depth")
        if cam_id < 0:
            print("[PHP] XML <camera name='php_depth'> not found; depth disabled")
            return
        try:
            renderer = mujoco.Renderer(m, int(dcfg["height"]), int(dcfg["width"]))
        except Exception as e:
            print(f"[PHP] could not create mujoco.Renderer ({e}); "
                  f"set MUJOCO_GL=egl for headless or glfw/osmesa; depth disabled")
            return
        php_depth_ctx["renderer"] = renderer
        php_depth_ctx["fixed_cam_id"] = int(cam_id)

    def _php_render_depth():
        if php_depth_ctx["renderer"] is None:
            _php_init_depth()
        if php_depth_ctx["renderer"] is None:
            state_cmd.depth_image = None
            return
        php = FSM_controller.php_parkour_policy
        dcfg = php.depth_cfg
        near = float(dcfg["near"])
        far = float(dcfg["far"])
        # Override the model's global near/zfar so the depth buffer has the
        # same range as the browser's Three.js camera (near=0.3, far=3.0).
        # With terrain, the default extent-relative znear/zfar give a 3 km
        # zfar, which destroys depth precision in the 1-3 m range we care
        # about. Restored after the render so viewer/main cam aren't affected.
        extent = float(m.stat.extent) or 1.0
        saved_znear = float(m.vis.map.znear)
        saved_zfar  = float(m.vis.map.zfar)
        m.vis.map.znear = near / extent
        m.vis.map.zfar  = far  / extent
        try:
            php_depth_ctx["renderer"].update_scene(
                d, camera=php_depth_ctx["fixed_cam_id"])
            php_depth_ctx["renderer"].enable_depth_rendering()
            depth = php_depth_ctx["renderer"].render()
            php_depth_ctx["renderer"].disable_depth_rendering()
            state_cmd.depth_image = np.clip(depth, near, far).astype(np.float32)
        except Exception as e:
            print(f"[PHP] depth render failed ({e}); feeding None this tick")
            state_cmd.depth_image = None
        finally:
            m.vis.map.znear = saved_znear
            m.vis.map.zfar  = saved_zfar

        # Throttled depth stats so user can verify scene is visible.
        if state_cmd.depth_image is not None:
            php_depth_ctx.setdefault("dbg_counter", 0)
            php_depth_ctx["dbg_counter"] += 1
            if php_depth_ctx["dbg_counter"] % 50 == 0:  # every ~1 s at 50 Hz
                dimg = state_cmd.depth_image
                cy, cx = dimg.shape[0] // 2, dimg.shape[1] // 2
                print(f"[PHP depth] min={dimg.min():.2f} "
                      f"max={dimg.max():.2f} mean={dimg.mean():.2f} "
                      f"center={dimg[cy, cx]:.2f} "
                      f"(expect min≈0.3, max≈3.0 when obstacle ahead)")

        # Optional on-screen preview of the raw depth image so the user can
        # see what the policy sees. Matches the small inset in the PHP demo.
        if state_cmd.depth_image is not None:
            try:
                import cv2
                dimg = state_cmd.depth_image
                near = float(dcfg["near"])
                far = float(dcfg["far"])
                norm = np.clip((dimg - near) / max(far - near, 1e-6), 0.0, 1.0)
                vis = (norm * 255.0).astype(np.uint8)
                scale = int(os.environ.get("PHP_DEPTH_PREVIEW_SCALE", "4"))
                if scale != 1:
                    vis = cv2.resize(vis, (vis.shape[1]*scale, vis.shape[0]*scale),
                                     interpolation=cv2.INTER_NEAREST)
                cv2.imshow("PHP depth", vis)
                cv2.waitKey(1)
            except Exception:
                pass

    log_cfg = cfg.get("logging", {})
    logger = None
    log_states = set()
    if log_cfg.get("enabled", False):
        logger = Logger(
            log_cfg.get("log_dir", "logs"),
            log_cfg.get("tag", "score"),
            extra_meta={"robot_type": "mujoco", "xml_path": cfg.xml_path,
                        "control_dt": mj_per_step_duration},
        )
        log_states = {FSMStateName[s] for s in log_cfg.get("states", ["SKILL_SCORE"])}
    log_step = 0

    joystick = JoyStick()
    prev_hat = (0, 0)
    prev_r2_pressed = False
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
                r2_pressed = joystick.get_axis_value(5) > 0.5
                r2_just_pressed = r2_pressed and not prev_r2_pressed
                if joystick.is_button_released(JoystickButton.Y):                                                     # Ghost toggle, Y
                    ghost_flags[0] = not ghost_flags[0]
                    print(f"[Ghost] {'ON' if ghost_flags[0] else 'OFF'}")
                if joystick.is_button_released(JoystickButton.L3):                                                    # PHP high/low speed toggle, L3
                    state_cmd.php_high_speed = not state_cmd.php_high_speed
                    print(f"[PHP] speed={'HIGH' if state_cmd.php_high_speed else 'LOW'}")
                if joystick.is_button_released(JoystickButton.X) and joystick.is_button_pressed(JoystickButton.R1):   # Ball reset, R1+X
                    if _reset_ball_state(
                        m, d, ball_body_id,
                        ball_reset_pos_w, ball_reset_vel_w,
                        ball_reset_quat_w, ball_reset_ang_vel_w,
                    ):
                        print(
                            f"[Ball] Reset to pos={ball_reset_pos_w.tolist()} "
                            f"vel={ball_reset_vel_w.tolist()}"
                        )
                    else:
                        print("[Ball] Reset requested, but no ball body exists in this scene.")

                # PASSIVE: safety command — always overrides any pending command
                if joystick.is_button_released(JoystickButton.L1):                                                    # 阻尼保护, L1
                    state_cmd.skill_cmd = FSMCommand.PASSIVE

                # All other skill commands: latched — only accepted when FSM has cleared the previous one.
                # This prevents a fast START → policy switch sequence from overwriting POS_RESET before the FSM
                # processes it, which would leave the current controller without a clean exit().
                # elif state_cmd.skill_cmd == FSMCommand.INVALID:
                if joystick.is_button_released(JoystickButton.START):                                              # 回 FixedPose, START
                    state_cmd.skill_cmd = FSMCommand.POS_RESET
                elif joystick.is_button_released(JoystickButton.B):                                                # Loco, B
                    state_cmd.skill_cmd = FSMCommand.LOCO
                elif joystick.is_button_released(JoystickButton.A):                                                # AMP, A
                    state_cmd.skill_cmd = FSMCommand.CMD_AMP
                elif joystick.is_button_released(JoystickButton.R1):                                               # Score, R1
                    state_cmd.skill_cmd = FSMCommand.CMD_SCORE
                elif hat_just_pressed(0, -1):                                                                     # BeyondMimicMJ, D-pad DOWN
                    state_cmd.skill_cmd = FSMCommand.CMD_BEYONDMIMIC_MJ
                elif hat_just_pressed(0, 1):                                                                      # StandUpMJ, D-pad UP
                    state_cmd.skill_cmd = FSMCommand.CMD_STANDUP_MJ
                elif hat_just_pressed(-1, 0):                                                                     # PHP Parkour, D-pad LEFT
                    state_cmd.skill_cmd = FSMCommand.CMD_PHP_PARKOUR
                elif r2_just_pressed:                                                                             # Pinocchio1.6MJ, R2
                    state_cmd.skill_cmd = FSMCommand.CMD_PINOCCHIO_1_6_MJ

                prev_hat = hat
                prev_r2_pressed = r2_pressed
                state_cmd.vel_cmd[0] = -joystick.get_axis_value(1)
                state_cmd.vel_cmd[1] = -joystick.get_axis_value(0)
                state_cmd.vel_cmd[2] = -joystick.get_axis_value(3)
                
                step_start = time.time()

                if getattr(policy_output, "direct_torque", False):
                    # PHP parkour (and any other direct-torque policy) writes
                    # actuator-indexed torques straight into policy_output.actions.
                    d.ctrl[:] = np.clip(policy_output_action, -tau_limit, tau_limit)
                else:
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

                    # PHP parkour: render depth when the PHP policy is active.
                    if (hasattr(FSM_controller, "php_parkour_policy") and
                            FSM_controller.cur_policy is
                            FSM_controller.php_parkour_policy):
                        _php_render_depth()
                    else:
                        state_cmd.depth_image = None

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

                    if (logger is not None and
                            FSM_controller.cur_policy.name in log_states):
                        t = log_step * mj_per_step_duration
                        logger.log(log_step, t, state_cmd, policy_output)
                        log_step += 1

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
                        _draw_viz_spheres(viewer.user_scn, policy_output.viz_spheres)
                else:
                    with viewer.lock():
                        viewer.user_scn.ngeom = 0
                        _draw_viz_spheres(viewer.user_scn, policy_output.viz_spheres)

                viewer.sync()
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
            except ValueError as e:
                print(str(e))

    if logger is not None:
        logger.close()

if __name__ == "__main__":
    main()

            

        