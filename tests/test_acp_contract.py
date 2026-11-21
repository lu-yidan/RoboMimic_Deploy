"""Contract tests use an explicit synthetic ONNX; no trained behavior is claimed."""
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest
import yaml

from common.ctrlcomp import StateAndCmd, PolicyOutput
from common.utils import FSMCommand
from policy.acp.ACP import ACP
from policy.acp.observations import ObservationHistory, goal_command, grid_xy
from policy.acp.terrain import MujocoTerrain, flat_grid


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bundle(tmp_path):
    names = yaml.safe_load((ROOT/'policy/php_parkour/config/php.yaml').read_text())['joint_names']
    controlled = [name for name in reversed(names) if '_wrist_' not in name]
    constants = {'actions': np.linspace(-.2,.2,23,dtype=np.float32)[None],
                 'contacts': np.zeros((1,9,3),np.float32), 'occupancy': np.full((1,9,1),.5,np.float32)}
    graph = helper.make_graph([helper.make_node('Constant', [], [name], value=numpy_helper.from_array(value))
        for name,value in constants.items()], 'synthetic_contract_only',
        [helper.make_tensor_value_info('planner_obs',TensorProto.FLOAT,[1,939])],
        [helper.make_tensor_value_info(name,TensorProto.FLOAT,list(value.shape)) for name,value in constants.items()])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('',18)], ir_version=9)
    onnx.save(model,tmp_path/'policy.onnx')
    manifest = {'schema_version':1,'stage':3,'planner_obs_dim':939,'history_length':5,
        'history_reset':'repeat','control_dt':.02,'joint_names':controlled,
        'default_joint_pos': np.linspace(0,.1,23).tolist(), 'kps':[40]*23,'kds':[1]*23,
        'effort_limits':[25]*23, 'fixed_joints':{name:{'position':.05,'kp':20,'kd':1,'effort_limit':5}
            for name in names if '_wrist_' in name}, 'action_scale':.25, 'action_clip':None,
        'target_radius':.5, 'policy_sha256':hashlib.sha256((tmp_path/'policy.onnx').read_bytes()).hexdigest()}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    return tmp_path,names,manifest,constants


def fresh(policy):
    state = policy.state_cmd
    state.q[:] = policy.default
    state.pelvis_pos_w[:] = [0,0,.78]
    state.pelvis_quat_w[:] = [1,0,0,0]
    state.acp_terrain = flat_grid(.78)
    state.acp_pose_time = state.acp_terrain_time = time.monotonic()
    state.acp_pose_valid = True
    return state


def make_policy(bundle):
    directory,names,_,_ = bundle
    return ACP(StateAndCmd(29),PolicyOutput(29),{'model_dir':str(directory)},names)


def test_raw_observation_history_and_named_action_mapping(bundle):
    policy = make_policy(bundle)
    state = fresh(policy)
    state.q[policy.to_model] += np.arange(23)*.001
    state.acp_ang_vel[:] = [1,2,3]
    policy.enter()
    policy.run()
    assert not policy.failed
    obs = policy.last_observation[0]
    assert obs.shape == (939,)
    np.testing.assert_allclose(obs[:15].reshape(5,3), np.tile([1,2,3],(5,1)))
    np.testing.assert_allclose(obs[15:30].reshape(5,3), np.tile([0,0,-1],(5,1)))
    np.testing.assert_allclose(obs[30:145].reshape(5,23), np.tile(np.arange(23)*.001,(5,1)),atol=1e-8)
    np.testing.assert_array_equal(obs[375:936].reshape(187,3),state.acp_terrain)
    np.testing.assert_array_equal(obs[-3:],[1,0,0])
    expected = policy.default.copy()
    expected[policy.to_model] += .25*bundle[3]['actions'][0]
    np.testing.assert_allclose(policy.policy_output.actions, expected)
    policy.run()
    history = policy.last_observation[0,260:375].reshape(5,23)
    np.testing.assert_array_equal(history[:-1], np.zeros((4,23)))
    np.testing.assert_array_equal(history[-1], bundle[3]['actions'][0])
    policy.exit()
    policy.enter()
    assert policy.history.terms is None
    np.testing.assert_array_equal(policy.last_action,0)


def test_goal_frame_and_near_goal_scaling():
    q = [np.sqrt(.5),0,0,np.sqrt(.5)]
    np.testing.assert_allclose(goal_command([2,3,.78],q,[2,4],np.pi/2,.5),[1,0,0],atol=1e-6)
    np.testing.assert_allclose(goal_command([0,0,.78],[1,0,0,0],[.1,0],2*np.pi-.1,.5),[.2,0,-.1],atol=1e-6)
    np.testing.assert_array_equal(goal_command([0,0,.78],[1,0,0,0],[0,0],0,.5),[0,0,0])


def test_terms_roll_oldest_to_newest():
    history = ObservationHistory()
    for tick in range(7):
        result = history.append([tick,0,0],[1,0,0,0],np.full(23,tick),np.zeros(23),np.zeros(23))
    np.testing.assert_array_equal(result[:15].reshape(5,3)[:,0], [2,3,4,5,6])
    np.testing.assert_array_equal(result[30:145].reshape(5,23)[:,0], [2,3,4,5,6])


@pytest.mark.parametrize('failure',['stale','nan','wrong_grid','low','tilted','missing_pose'])
def test_invalid_input_damps_immediately(bundle,failure):
    policy = make_policy(bundle)
    state = fresh(policy)
    if failure == 'stale': state.acp_terrain_time -= 1
    if failure == 'nan': state.acp_terrain[0,2] = np.nan
    if failure == 'wrong_grid': state.acp_terrain = state.acp_terrain[::-1]
    if failure == 'low': state.acp_terrain[:,2] = -.3
    if failure == 'tilted': state.pelvis_quat_w[:] = [np.cos(.5),np.sin(.5),0,0]
    if failure == 'missing_pose': state.acp_pose_valid = False
    policy.enter()
    policy.run()
    assert policy.failed and state.skill_cmd == FSMCommand.PASSIVE
    np.testing.assert_array_equal(policy.policy_output.kps,0)
    np.testing.assert_array_equal(policy.policy_output.kds,8)


def test_bundle_hash_and_incomplete_mapping_fail(bundle):
    directory,names,manifest,_ = bundle
    (directory/'policy.onnx').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='SHA256'):
        make_policy(bundle)
    del manifest['fixed_joints'][next(iter(manifest['fixed_joints']))]
    (directory/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='six explicitly fixed wrists'):
        make_policy(bundle)


def test_real_fsm_entry_ramp_reset_and_same_tick_damping(bundle):
    from FSM.FSM import FSM, FSMMode
    directory,names,_,_ = bundle
    state,output = StateAndCmd(29),PolicyOutput(29)
    fsm = FSM(state,output,joint_names=names,enable_freekick=False,
              acp_config={'enabled':True,'model_dir':str(directory)})
    policy = fsm.acp_policy
    fresh(policy)
    state.skill_cmd = FSMCommand.CMD_ACP
    fsm.run()
    assert fsm.cur_policy is fsm.passive_mode
    state.skill_cmd = FSMCommand.POS_RESET
    fsm.run()
    fsm.run()
    state.skill_cmd = FSMCommand.CMD_ACP
    fsm.run()
    assert fsm.cur_policy is fsm.fixed_pose_1
    for _ in range(100): fsm.run()
    np.testing.assert_array_equal(fsm.fixed_pose_1._default_out,policy.default)
    fresh(policy)
    state.skill_cmd = FSMCommand.CMD_ACP
    fsm.run()
    assert fsm.cur_policy is policy
    assert policy.last_observation is not None
    state.acp_pose_valid = False
    fsm.run()
    assert fsm.cur_policy is fsm.passive_mode
    np.testing.assert_array_equal(output.kps,0)
    assert policy.history.terms is None
    fsm.cur_policy = policy
    fsm.FSMmode = FSMMode.CHANGE
    state.skill_cmd = FSMCommand.PASSIVE
    fsm.run()
    assert fsm.cur_policy is fsm.passive_mode


def test_rays_skip_robot_and_rotate_grid_at_step():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="plane" size="10 10 .1"/>
      <geom type="box" pos=".5 0 .1" size=".25 2 .1"/>
      <body pos="0 0 .78"><freejoint/><geom type="sphere" size=".1"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    state = StateAndCmd(29)
    sampler = MujocoTerrain(model)
    mujoco.mj_forward(model,data)
    sampler.capture(data,state)
    assert state.acp_pose_valid
    expected = np.where((grid_xy()[:,0] >= .25)&(grid_xy()[:,0] <= .75), .08, .28)
    np.testing.assert_allclose(state.acp_terrain[:,2],expected,atol=1e-6)
    data.qpos[3:7] = [np.sqrt(.5),0,0,np.sqrt(.5)]
    mujoco.mj_forward(model,data)
    sampler.capture(data,state)
    expected = np.where((grid_xy()[:,1] <= -.25)&(grid_xy()[:,1] >= -.75), .08, .28)
    np.testing.assert_allclose(state.acp_terrain[:,2],expected,atol=1e-6)
