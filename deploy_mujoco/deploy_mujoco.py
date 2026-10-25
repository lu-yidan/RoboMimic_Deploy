import sys
from pathlib import Path
# Put project root first so `import deploy_mujoco.*` resolves to the package
# directory, not this script (same basename) when launched as
# `python deploy_mujoco/deploy_mujoco.py`.
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from common.path_config import PROJECT_ROOT
from deploy_mujoco.reset_sim import reset_to_fixed_pose

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
from threading import Event
from deploy_mujoco.scene_builder import apply_initial_state, load_model


def _draw_viz_spheres(scn, viz_spheres):
    """Draw debug spheres/lines/arrows into a MuJoCo mjvScene.

    Each entry in viz_spheres is a dict with either:
      - sphere: {"pos": (3,), "radius": float, "rgba": (4,)}
      - line:   {"from": (3,), "to": (3,), "radius": float, "rgba": (4,)}
      - arrow:  {"from": (3,), "to": (3,), "radius": float, "rgba": (4,), "geom": "arrow"}
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
            geom_type = mujoco.mjtGeom.mjGEOM_CAPSULE
            if item.get("geom", "").lower() == "arrow":
                geom_type = getattr(
                    mujoco.mjtGeom, "mjGEOM_ARROW", mujoco.mjtGeom.mjGEOM_CAPSULE
                )
            mujoco.mjv_makeConnector(
                g, geom_type, r,
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


def _print_joint_pos(q: np.ndarray, fsm_state_name: str = ""):
    """Print current joint positions in MuJoCo DFS order, grouped by segment."""
    def fmt(vals):
        return ",  ".join(f"{v:7.4f}" for v in vals)
    tag = f"  [{fsm_state_name}]" if fsm_state_name else ""
    print(f"\n[Joint pos]{tag}")
    print(f"  L-leg  : [{fmt(q[0:6])}]")
    print(f"  R-leg  : [{fmt(q[6:12])}]")
    print(f"  waist  : [{fmt(q[12:15])}]")
    print(f"  L-arm  : [{fmt(q[15:22])}]")
    print(f"  R-arm  : [{fmt(q[22:29])}]")
    all_vals = ",  ".join(f"{v:7.4f}" for v in q)
    print(f"  all    : [{all_vals}]")


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

@hydra.main(version_base="1.1", config_path="config", config_name="mujoco")
def main(cfg: DictConfig):
    import torch
    torch.set_num_threads(int(cfg.get("torch_threads", 1)))
    render_hz = float(cfg.get("render_hz", 60))
    if render_hz <= 0:
        raise ValueError("render_hz must be positive")
    xml_path = os.path.join(PROJECT_ROOT, cfg.xml_path)
    simulation_dt = cfg.simulation_dt
    control_decimation = cfg.control_decimation
    tau_limit = np.array(cfg.tau_limit)
        
    m = load_model(xml_path, cfg.get("scene", {}))
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    apply_initial_state(m, d, cfg.get("initial_state", {}))
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
    ghost_flags = [bool(cfg.get("ghost_enabled", True))]


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
    from common.deployment_config import load_mujoco_scalebfm_config
    scalebfm_cfg = load_mujoco_scalebfm_config(cfg)
    if scalebfm_cfg and scalebfm_cfg.enabled and not np.isclose(mj_per_step_duration, .02):
        raise ValueError("ScaleBFM requires a 50 Hz control loop")
    joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[i, 0]))
                   for i in range(num_joints)]
    if scalebfm_cfg and scalebfm_cfg.enabled:
        actuator_joints = m.actuator_trnid[:, 0]
        if not (np.array_equal(m.jnt_qposadr[actuator_joints], np.arange(7, 7 + num_joints))
                and np.array_equal(m.jnt_dofadr[actuator_joints], np.arange(6, 6 + num_joints))):
            raise ValueError("Deployment loop requires actuator order to match qpos/dof order")
    FSM_controller = FSM(state_cmd, policy_output, scalebfm_cfg, joint_names, enable_freekick=False)

    catalog = None
    if FSM_controller.scalebfm_policy is not None:
        from policy.scalebfm.motion_catalog import MotionCatalog, play_selected
        catalog = MotionCatalog(FSM_controller.scalebfm_policy.motion_path,
                                scalebfm_cfg.get("motion_globs", []), PROJECT_ROOT)
        print(catalog.describe())
    startup_policy = str(cfg.get("startup_policy", "passive"))
    startup_commands = {"passive": FSMCommand.INVALID, "fixed_pose": FSMCommand.POS_RESET,
                        "smp_recovery": FSMCommand.CMD_SMP_RECOVERY}
    if startup_policy not in startup_commands:
        raise ValueError("startup_policy must be passive, fixed_pose, or smp_recovery")
    state_cmd.skill_cmd = startup_commands[startup_policy]
    print(f"[Startup] {startup_policy}; physics={simulation_dt}s; control={simulation_dt * control_decimation}s")
    print("[Controls] START: FixedPose; Y: SMP; keyboard R: reset simulation into passive mode")

    log_cfg = cfg.get("logging", {})
    logger = None
    log_states = set()
    if log_cfg.get("enabled", False):
        logger = Logger(
            log_cfg.get("log_dir", "logs"),
            log_cfg.get("tag", "freekick"),
            extra_meta={"robot_type": "mujoco", "xml_path": cfg.xml_path,
                        "control_dt": mj_per_step_duration},
        )
        log_states = {FSMStateName[s] for s in log_cfg.get("states", ["SKILL_FREEKICK"])}
    log_step = 0

    joystick = JoyStick()
    prev_hat = (0, 0)
    prev_r2_pressed = False
    prev_l2_pressed = False
    Running = True
    reset_requested = Event()
    standing_reset_requested = Event()
    def key_callback(key):
        if key in (ord('R'), ord('r')):
            reset_requested.set()
        elif key == 259:
            standing_reset_requested.set()
    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        camera = cfg.get("camera")
        if camera:
            viewer.cam.lookat[:] = camera.get("lookat", [0, 0, .4])
            viewer.cam.distance = float(camera.get("distance", 3.2))
            viewer.cam.azimuth = float(camera.get("azimuth", 45))
            viewer.cam.elevation = float(camera.get("elevation", -35))
        deadline = time.perf_counter()
        next_render = deadline
        while viewer.is_running() and Running:
            try:
                if reset_requested.is_set():
                    reset_requested.clear()
                    # Physics and FSM changes run on this thread, not the viewer callback.
                    FSM_controller.cur_policy.exit()
                    mujoco.mj_resetData(m, d)
                    apply_initial_state(m, d, cfg.get("initial_state", {}))
                    state_cmd.q = d.qpos[7:7 + num_joints].copy()
                    state_cmd.dq = d.qvel[6:6 + num_joints].copy()
                    state_cmd.skill_cmd = FSMCommand.INVALID
                    FSM_controller.get_next_policy(FSMStateName.PASSIVE)
                    FSM_controller.FSMmode = FSMMode.NORMAL
                    FSM_controller.sim_counter = 0
                    FSM_controller.cur_policy.enter()
                    FSM_controller.cur_policy.run()
                    policy_output_action[:] = policy_output.actions
                    kps[:] = policy_output.kps
                    kds[:] = policy_output.kds
                    print("[Reset] configured initial state; PASSIVE. Press Y to start SMP.")
                    viewer.sync()
                    deadline = time.perf_counter()
                    continue
                if(joystick.is_button_pressed(JoystickButton.SELECT)):
                    Running = False

                joystick.update()
                if joystick.is_button_pressed(JoystickButton.SELECT):
                    break
                if (standing_reset_requested.is_set() or
                        (joystick.is_button_pressed(JoystickButton.L1) and
                         joystick.is_button_released(JoystickButton.START))):
                    reset_to_fixed_pose(m, d, FSM_controller, state_cmd, policy_output)
                    policy_output_action = policy_output.actions.copy()
                    kps, kds = policy_output.kps.copy(), policy_output.kds.copy()
                    standing_reset_requested.clear()
                    deadline = time.perf_counter()
                    print("[Simulation] Reset to standing FixedPose; R1 starts ScaleBFM.")
                    continue
                hat = joystick.get_hat_direction()
                hat_just_pressed = lambda hx, hy: (hat == (hx, hy) and prev_hat != (hx, hy))
                r2_pressed = joystick.get_axis_value(5) > 0.5
                r2_just_pressed = r2_pressed and not prev_r2_pressed
                l2_pressed = joystick.get_axis_value(2) > 0.5
                l2_just_released = prev_l2_pressed and not l2_pressed
                if catalog is not None and not l2_just_released:
                    if not joystick.is_button_pressed(JoystickButton.L1):
                        delta = 1 if hat_just_pressed(1, 0) else -1 if hat_just_pressed(-1, 0) else 0
                        if delta:
                            catalog.move(delta)
                            FSM_controller.scalebfm_policy._close_progress()
                            print(catalog.describe())
                    if joystick.is_button_released(JoystickButton.R1):
                        # Consume edges even on a rejected file, so a held input cannot repeat.
                        prev_hat, prev_r2_pressed, prev_l2_pressed = hat, r2_pressed, l2_pressed
                        try:
                            play_selected(catalog, m, d, FSM_controller, state_cmd, policy_output)
                        except (OSError, ValueError, KeyError, AssertionError) as exc:
                            print(f"[Motion] Cannot load {catalog.selected.name}: {exc}")
                        policy_output_action = policy_output.actions.copy()
                        kps, kds = policy_output.kps.copy(), policy_output.kds.copy()
                        continue
                if joystick.is_button_released(JoystickButton.L3):                                                    # Ghost toggle, L3
                    ghost_flags[0] = not ghost_flags[0]
                    print(f"[Ghost] {'ON' if ghost_flags[0] else 'OFF'}")
                if joystick.is_button_released(JoystickButton.Y) and joystick.is_button_pressed(JoystickButton.L1):  # Print joint pos, L1+Y
                    _print_joint_pos(d.qpos[7:7+num_joints],
                                     FSM_controller.cur_policy.name_str)
                if joystick.is_button_released(JoystickButton.X):                                                        # Ball reset, X
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

                # Target Y-bias: L1 held + D-pad Right/Left (edge-triggered, ±5 cm)
                if joystick.is_button_pressed(JoystickButton.L1):
                    if hat_just_pressed(1, 0):    # D-pad Right → target moves right (pelvis -Y)
                        state_cmd.target_y_bias = float(np.clip(state_cmd.target_y_bias - 0.05, -1.50, 1.50))
                        print(f"\n[BIAS] target_y_bias = {state_cmd.target_y_bias:+.2f} m", flush=True)
                    elif hat_just_pressed(-1, 0):  # D-pad Left → target moves left (pelvis +Y)
                        state_cmd.target_y_bias = float(np.clip(state_cmd.target_y_bias + 0.05, -1.50, 1.50))
                        print(f"\n[BIAS] target_y_bias = {state_cmd.target_y_bias:+.2f} m", flush=True)

                # PASSIVE: safety command — L2 release
                if l2_just_released:                                                                                   # 阻尼保护, L2
                    state_cmd.skill_cmd = FSMCommand.PASSIVE

                # All other skill commands: latched — only accepted when FSM has cleared the previous one.
                # This prevents a fast START → policy switch sequence from overwriting POS_RESET before the FSM
                # processes it, which would leave the current controller without a clean exit().
                # elif state_cmd.skill_cmd == FSMCommand.INVALID:
                elif joystick.is_button_released(JoystickButton.START):                                              # 回 FixedPose, START
                    state_cmd.skill_cmd = FSMCommand.POS_RESET
                elif joystick.is_button_released(JoystickButton.B):                                                # Loco, B
                    state_cmd.skill_cmd = FSMCommand.LOCO
                elif joystick.is_button_released(JoystickButton.A):                                                # AMP, A
                    state_cmd.skill_cmd = FSMCommand.CMD_AMP
                elif joystick.is_button_just_pressed(JoystickButton.Y) and not joystick.is_button_pressed(JoystickButton.L1):  # SMP Recovery, Y
                    state_cmd.skill_cmd = FSMCommand.CMD_SMP_RECOVERY
                elif hat_just_pressed(0, -1):                                                                     # BeyondMimicMJ, D-pad DOWN
                    state_cmd.skill_cmd = FSMCommand.CMD_BEYONDMIMIC_MJ
                elif hat_just_pressed(0, 1):                                                                      # StandUpMJ, D-pad UP
                    state_cmd.skill_cmd = FSMCommand.CMD_STANDUP_MJ
                elif r2_just_pressed:                                                                             # Pinocchio1.6MJ, R2
                    state_cmd.skill_cmd = FSMCommand.CMD_PINOCCHIO_1_6_MJ

                prev_hat = hat
                prev_r2_pressed = r2_pressed
                prev_l2_pressed = l2_pressed
                state_cmd.vel_cmd[0] = -joystick.get_axis_value(1)
                state_cmd.vel_cmd[1] = -joystick.get_axis_value(0)
                state_cmd.vel_cmd[2] = -joystick.get_axis_value(3)
                
                # The deadline accumulates physics time, including inference/input cost.
                
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

                    if (logger is not None and
                            FSM_controller.cur_policy.name in log_states):
                        t = log_step * mj_per_step_duration
                        logger.log(log_step, t, state_cmd, policy_output)
                        log_step += 1

                # Rendering is independent of the 500 Hz physics / 50 Hz policy loop.
                now = time.perf_counter()
                if now >= next_render:
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
                    next_render = time.perf_counter() + 1.0 / render_hz

                deadline += m.opt.timestep
                now = time.perf_counter()
                remaining = deadline - now
                if remaining > 0:
                    time.sleep(remaining)
                elif remaining < -0.1:
                    # Do not burst through seconds of motion after a stalled window.
                    deadline = now
            except KeyboardInterrupt:
                Running = False
            except ValueError as e:
                print(str(e))

    if logger is not None:
        logger.close()

if __name__ == "__main__":
    main()

            

        