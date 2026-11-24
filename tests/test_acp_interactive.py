"""Live keyboard goals and B-mode initialization, without display or transport."""
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.utils import FSMCommand, FSMStateName
from deploy_mujoco.acp_interactive import KeyboardControl, velocity_sticks, reset_to_loco
from policy.loco_mode.LocoMode import LocoMode
from policy.passive.PassiveMode import PassiveMode

ROOT = Path(__file__).resolve().parents[1]


def test_keyboard_goal_uses_current_pelvis_and_damping_wins():
    keyboard = KeyboardControl(True)
    state = StateAndCmd(29)
    state.pelvis_pos_w[:] = [2,3,.8]
    state.pelvis_quat_w[:] = [np.sqrt(.5),0,0,np.sqrt(.5)]
    loco = SimpleNamespace(range_velx=[-.4,.7],range_vely=[-.4,.4],range_velz=[-1.57,1.57])
    policy = SimpleNamespace(target=np.zeros(2),heading=0.)
    fsm = SimpleNamespace(loco_policy=loco,acp_policy=policy)
    for key in ('2','W','Y'): keyboard.push(ord(key))
    keyboard.poll(state,fsm)
    assert state.skill_cmd == FSMCommand.CMD_ACP
    np.testing.assert_allclose(policy.target,[1,3],atol=1e-6)
    np.testing.assert_allclose(policy.heading,np.pi,atol=1e-6)
    np.testing.assert_allclose(keyboard.velocity,[.1,0,0])
    normalized = velocity_sticks(loco,[0,0,0])
    assert normalized[0] != 0  # Neutral stick would command +0.15 m/s.
    state.skill_cmd = FSMCommand.INVALID
    keyboard.push(290);keyboard.push(ord('Y'))
    keyboard.poll(state,fsm)
    assert state.skill_cmd == FSMCommand.PASSIVE


def test_reset_initializes_actual_loco_and_does_not_apply_support():
    import torch
    torch.set_num_threads(1)
    state,output = StateAndCmd(29),PolicyOutput(29)
    loco = LocoMode(state,output)
    fsm = SimpleNamespace(loco_policy=loco,cur_policy=PassiveMode(state,output))
    model = mujoco.MjModel.from_xml_path(str(ROOT/'g1_description/scene.xml'))
    data = mujoco.MjData(model)
    data.xfrc_applied[:] = 12
    reset_to_loco(model,data,fsm,state,output)
    assert fsm.cur_policy.name == FSMStateName.LOCOMODE
    np.testing.assert_array_equal(data.xfrc_applied,0)
    np.testing.assert_array_equal(data.qvel,0)
    np.testing.assert_allclose(loco.cmd,0,atol=1e-7)
    assert np.isfinite(output.actions).all()
    before = data.qpos.copy()
    data.ctrl[:] = 0
    mujoco.mj_step(model,data)
    assert not np.array_equal(data.qpos[:3],before[:3])  # Root remains free.


def test_acp_gui_preset_starts_loco_and_supports_keyboard_without_gamepad():
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(ROOT/'deploy_mujoco/config'),version_base=None):
        cfg = compose(config_name='mujoco_acp')
    assert cfg.startup_policy == 'loco'
    assert cfg.keyboard_control and not cfg.joystick_required
