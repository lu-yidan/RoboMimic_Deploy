"""Exercise the actual deployment FSM and 50 Hz PD loop without a joystick."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import mujoco
import numpy as np
from omegaconf import OmegaConf
from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.utils import FSMCommand, FSMStateName, get_gravity_orientation
from FSM.FSM import FSM
from deploy_mujoco.deploy_mujoco import pd_control


def main():
    from common.deployment_config import load_scalebfm_config
    cfg = OmegaConf.load(ROOT / 'deploy_mujoco/config/mujoco.yaml')
    cfg.scalebfm = load_scalebfm_config(require_enabled=True)
    m = mujoco.MjModel.from_xml_path(str(ROOT / (sys.argv[1] if len(sys.argv) > 1 else 'g1_description/scene.xml')))
    m.opt.timestep = cfg.simulation_dt
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    state, output = StateAndCmd(29), PolicyOutput(29)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[i,0])) for i in range(m.nu)]
    fsm = FSM(state, output, cfg.scalebfm, names, enable_freekick=False)
    from deploy_mujoco.reset_sim import reset_to_fixed_pose
    limits = np.array(cfg.tau_limit)
    minimum_height, maximum_tilt = 10., 0.

    def tick(command=FSMCommand.INVALID):
        nonlocal minimum_height, maximum_tilt
        state.q, state.dq = d.qpos[7:36].copy(), d.qvel[6:35].copy()
        state.pelvis_pos_w = d.qpos[:3].astype(np.float32)
        state.pelvis_quat_w = d.qpos[3:7].astype(np.float32)
        state.root_ang_vel_b = d.qvel[3:6].astype(np.float32)
        state.ang_vel = state.root_ang_vel_b.copy()
        state.gravity_ori = get_gravity_orientation(d.qpos[3:7])
        state.skill_cmd = command
        fsm.run()
        assert np.isfinite(output.actions).all()
        for _ in range(cfg.control_decimation):
            assert np.all(d.xfrc_applied == 0)
            d.ctrl[:] = np.clip(pd_control(output.actions, d.qpos[7:36], output.kps,
                                           np.zeros(29), d.qvel[6:35], output.kds), -limits, limits)
            mujoco.mj_step(m,d)
        mujoco.mj_forward(m,d)
        tilt = float(np.degrees(np.arccos(np.clip(d.body('pelvis').xmat[8],-1,1))))
        minimum_height = min(minimum_height, float(d.qpos[2]))
        maximum_tilt = max(maximum_tilt, tilt)
        assert d.qpos[2] > .3 and tilt < 70, (d.qpos[2], tilt)

    tick(FSMCommand.CMD_SCALEBFM)
    assert fsm.cur_policy.name == FSMStateName.PASSIVE
    reset_to_fixed_pose(m, d, fsm, state, output)
    for _ in range(10): tick()
    assert fsm.cur_policy.name == FSMStateName.FIXEDPOSE
    tick(FSMCommand.CMD_SCALEBFM)
    for _ in range(500): tick()
    assert fsm.cur_policy.name == FSMStateName.SKILL_SCALEBFM
    assert not fsm.scalebfm_policy.failed
    tick(FSMCommand.POS_RESET)
    # A fallen, moving state must be reset, rather than interpolated back by PD.
    d.qpos[2] = .15
    d.qpos[3:7] = [np.cos(.7), np.sin(.7), 0, 0]
    d.qvel[:] = 3.
    mujoco.mj_forward(m, d)
    reset_to_fixed_pose(m, d, fsm, state, output)
    assert d.qpos[2] > .5 and np.all(d.qvel == 0)
    assert np.allclose(d.qpos[3:7], m.qpos0[3:7])
    for _ in range(10): tick()
    tick(FSMCommand.CMD_SCALEBFM)
    tick()
    assert int(fsm.scalebfm_policy.env.episode_length_buf.item()) == 0
    assert not fsm.scalebfm_policy.env.state_buffer["action_buffer"].any().item()
    for _ in range(100): tick()
    from policy.scalebfm.motion_catalog import MotionCatalog, play_selected
    catalog = MotionCatalog(fsm.scalebfm_policy.motion_path,
                            cfg.scalebfm.motion_globs, ROOT)
    fsm.scalebfm_policy.preload_motions(catalog.paths)
    assert fsm.scalebfm_policy.require_cached_motion
    policy_id = id(fsm.scalebfm_policy.agent.policy)
    for path in catalog.paths:
        name = path.name
        active = fsm.scalebfm_policy.motion_path
        target = next(i for i,p in enumerate(catalog.paths) if p.name == name)
        catalog.move(target - catalog.index)
        assert fsm.scalebfm_policy.motion_path == active  # selection is only pending
        before_q, before_v = d.qpos.copy(), d.qvel.copy()
        play_selected(catalog, m, d, fsm, state, output)
        assert np.array_equal(d.qpos, before_q) and np.array_equal(d.qvel, before_v)
        assert id(fsm.scalebfm_policy.agent.policy) == policy_id
        assert fsm.freekick_policy is None
        # tick() normally clears commands; explicitly submit the queued command here.
        tick(FSMCommand.CMD_SCALEBFM)
        tick()
        assert fsm.scalebfm_policy.env.episode_length_buf.item() == 0
        assert not fsm.scalebfm_policy.env.state_buffer['action_buffer'].any().item()
        for _ in range(100): tick()
    # Exercise real AMP/Loco actors and both directions without a physics reset.
    for command, policy in ((FSMCommand.CMD_AMP, FSMStateName.SKILL_AMP),
                            (FSMCommand.LOCO, FSMStateName.LOCOMODE)):
        tick(command)
        tick()
        assert fsm.cur_policy.name == policy
        for _ in range(25): tick()
        before_q, before_v = d.qpos.copy(), d.qvel.copy()
        play_selected(catalog, m, d, fsm, state, output)
        assert np.array_equal(d.qpos, before_q) and np.array_equal(d.qvel, before_v)
        tick(FSMCommand.CMD_SCALEBFM)
        tick()
        assert fsm.cur_policy.name == FSMStateName.SKILL_SCALEBFM
        assert fsm.scalebfm_policy.env.episode_length_buf.item() == 0
        for _ in range(50): tick()
    active = fsm.scalebfm_policy.env
    try:
        fsm.scalebfm_policy.load_motion(ROOT / 'missing-motion-for-test.npz')
    except (FileNotFoundError, ValueError):
        pass
    else:
        raise AssertionError('Missing motion must be rejected')
    assert fsm.scalebfm_policy.env is active
    tick(FSMCommand.PASSIVE)
    tick()
    assert fsm.cur_policy.name == FSMStateName.PASSIVE
    assert np.all(output.kps == 0)
    assert output.viz_spheres is None
    report = dict(passed=True, root_height_min_m=minimum_height,
                  max_pelvis_tilt_deg=maximum_tilt,
                  checks=['passive entry rejected', 'FixedPose to ScaleBFM', '10s physical rollout',
                          'fallen-state reset', 'reentry history reset', '2s second rollout', f'{len(catalog.paths)} motion selections with same checkpoint', 'AMP and Loco direct entry without reset', 'bad file preserves active motion', 'Passive exit'])
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
