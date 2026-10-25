"""Export a training AMP actor to the existing Amp.py wire format.

Run in the AMP training environment (PYTHONPATH=runtime:.), with this deployment
checkout supplied via --deploy-root. The discriminators are not exported.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml


class LegacyAmpActor(torch.nn.Module):
    """Bidirectional joint-order/action-unit conversion around the trained actor."""
    def __init__(self, actor, normalizer, training_from_legacy, legacy_from_training,
                 training_scale, deployment_scale):
        super().__init__()
        self.actor = actor
        self.normalizer = normalizer
        self.register_buffer('training_from_legacy', torch.tensor(training_from_legacy))
        self.register_buffer('legacy_from_training', torch.tensor(legacy_from_training))
        self.register_buffer('training_scale', torch.tensor(training_scale))
        self.deployment_scale = deployment_scale

    def training_obs(self, obs):
        frames = obs.reshape(-1, 4, 96)
        q = frames[:, :, 9:38].index_select(-1, self.training_from_legacy)
        dq = frames[:, :, 38:67].index_select(-1, self.training_from_legacy)
        # Amp.py stores the last ONNX output, which has deployment action units.
        last = frames[:, :, 67:96].index_select(-1, self.training_from_legacy)
        last = last * self.deployment_scale / self.training_scale
        return torch.cat((frames[:, :, :9], q, dq, last), dim=-1).reshape(-1, 384)

    def forward(self, obs):
        raw = self.actor(self.normalizer(self.training_obs(obs)))
        return (raw * self.training_scale / self.deployment_scale).index_select(
            -1, self.legacy_from_training)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deploy-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', default='Unitree-G1-MultiAMP-Stratified-Guidance-Flat',
                        choices=['Unitree-G1-AMP-Stratified-Flat', 'Unitree-G1-MultiAMP-Stratified-Guidance-Flat'])
    args = parser.parse_args()
    torch.set_num_threads(1)
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from rsl_rl.runners.amp_on_policy_runner import _unpack_obs, _unpack_step

    task = args.task
    deploy_cfg = yaml.safe_load((args.deploy_root/'policy/amp/config/amp.yaml').read_text())
    xml = ET.parse(args.deploy_root/'g1_description/g1_29dof_rev_1_0.xml')
    motor_names = [a.attrib['joint'] for a in xml.getroot().find('actuator')]
    legacy_names = [motor_names[i] for i in deploy_cfg['joint2motor_idx']]
    bank = np.load(args.bank)
    selected = []
    for source in ['procedural', 'natural']:
        for label in ['supine', 'prone', 'left_side_down', 'right_side_down']:
            mask = (bank['sources']==source)&(bank['stages']=='low')&(bank['labels']==label)
            selected.extend(np.flatnonzero(mask)[:8].tolist())
    poses = bank['qpos'][selected]
    nr = len(poses); n = nr + 8

    def reset(env, env_ids):
        robot = env.scene['robot']
        ids = env_ids if env_ids is not None else torch.arange(n, device=env.device)
        root = robot.data.default_root_state[ids].clone()
        root[:, :3] += env.scene.env_origins[ids]
        joints = robot.data.default_joint_pos[ids].clone()
        low = ids < nr
        q = torch.tensor(poses, device=env.device)
        mapping = [bank['joint_names'].tolist().index(name) for name in robot.joint_names]
        root[low, :7] = q[ids[low], :7]
        root[low, :3] += env.scene.env_origins[ids[low]]
        joints[low] = q[ids[low], 7:][:, mapping]
        robot.write_root_link_pose_to_sim(root[:, :7], env_ids=ids)
        robot.write_root_link_velocity_to_sim(torch.zeros(len(ids), 6, device=env.device), env_ids=ids)
        robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=ids)

    cfg = load_env_cfg('Unitree-G1-AMP-Flat', play=True)
    cfg.seed = 12345; cfg.scene.num_envs = n
    cfg.events = {'reset': EventTermCfg(func=reset, mode='reset')}
    cfg.terminations = {}; cfg.curriculum = {}
    cmd = cfg.commands['twist']; cmd.heading_command = False; cmd.ranges.heading = None
    cmd.rel_standing_envs = 0.; cmd.rel_heading_envs = 0.
    cmd.ranges.lin_vel_x = (0., 0.); cmd.ranges.lin_vel_y = (0., 0.); cmd.ranges.ang_vel_z = (0., 0.)
    agent = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0')
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = load_runner_cls(task)(wrapped, asdict(agent), log_dir=None, device='cuda:0')
    runner.load(str(args.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device='cuda:0')
    obs, _ = _unpack_obs(wrapped.get_observations())
    robot = env.scene['robot']; action = env.action_manager.get_term('joint_pos')
    joint_names = list(robot.joint_names)
    assert list(action._target_names) == joint_names, 'Action ordering requires a separate map'
    t_from_l = [legacy_names.index(name) for name in joint_names]
    l_from_t = [joint_names.index(name) for name in legacy_names]
    scale = action.scale[0].cpu().numpy()
    defaults = robot.data.default_joint_pos[0].cpu().numpy()
    motor_from_t = [joint_names.index(name) for name in motor_names]
    np.testing.assert_allclose(defaults[motor_from_t], deploy_cfg['default_dof_pos'], atol=1e-7)
    assert deploy_cfg['obs_dim_per_frame']==96 and deploy_cfg['history_len']==4
    assert all(deploy_cfg[k]==1. for k in ['ang_vel_scale','dof_pos_scale','dof_vel_scale'])
    assert cfg.observations['actor'].history_ordering=='time' and env.step_dt==.02
    assert list(cfg.observations['actor'].terms)==['base_ang_vel','projected_gravity','command','joint_pos','joint_vel','actions']
    assert agent.clip_actions is None, 'Handle action clipping explicitly before exporting'
    kps=[]; kds=[]
    for name in motor_names:
        actuators=[a for a in cfg.scene.entities['robot'].articulation.actuators if any(re.fullmatch(e,name) for e in a.target_names_expr)]
        assert len(actuators)==1
        kps.append(actuators[0].stiffness);kds.append(actuators[0].damping)
    np.testing.assert_allclose(kps, deploy_cfg['kps'], rtol=.001)
    np.testing.assert_allclose(kds, deploy_cfg['kds'], rtol=.001)

    export = LegacyAmpActor(copy.deepcopy(runner.alg.policy.actor).cpu().eval(),
        copy.deepcopy(runner.obs_normalizer).cpu().eval(), t_from_l, l_from_t,
        scale, float(deploy_cfg['action_scale'])).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(export, torch.zeros(1,384), str(args.output), input_names=['obs'],
        output_names=['actions'], opset_version=17, dynamo=False)
    model = onnx.load(str(args.output));onnx.checker.check_model(model)
    onnx.helper.set_model_props(model, {'interface':'RoboMimic_Deploy Amp.py 96x4',
        'training_task':task,'checkpoint_sha256':sha(args.checkpoint),
        'adapter':'joint permutation + per-joint output scaling + inverse previous-action scaling',
        'observation_normalizer':'embedded'})
    onnx.save(model,str(args.output))
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(args.output),sess_options=options,providers=['CPUExecutionProvider'])

    # Reconstruct the unmodified deployment observation from simulator physical
    # state, independently of the native observation tensor. Compare all history
    # slots and action targets against the actual runner over a sequential rollout.
    history=None;last_deploy=np.zeros((n,29),np.float32)
    max_obs_error=0.;max_target_error=0.;max_onnx_error=0.;max_action=0.
    samples=[];expected=[];targets=[]
    with torch.inference_mode():
        for step in range(151):
            frame=np.concatenate((robot.data.root_link_ang_vel_b.cpu().numpy(),
                robot.data.projected_gravity_b.cpu().numpy(),
                env.command_manager.get_command('twist').cpu().numpy(),
                (robot.data.joint_pos.cpu().numpy()-defaults)[:,l_from_t],
                robot.data.joint_vel.cpu().numpy()[:,l_from_t],last_deploy),axis=-1).astype(np.float32)
            history=np.repeat(frame[:,None,:],4,axis=1) if history is None else np.concatenate((history[:,1:],frame[:,None,:]),axis=1)
            deploy_obs=history.reshape(n,384)
            native=export.training_obs(torch.from_numpy(deploy_obs)).numpy()
            error=float(np.abs(native-obs.cpu().numpy()).max());max_obs_error=max(max_obs_error,error)
            np.testing.assert_allclose(native,obs.cpu().numpy(),atol=5e-5,rtol=1e-5)
            raw=policy(obs);native_raw=raw.cpu().numpy()
            # Expected result is derived from native joint targets, not wrapper.forward.
            native_targets=native_raw*scale+defaults
            deploy_expected=((native_targets-defaults)/deploy_cfg['action_scale'])[:,l_from_t]
            if step%10==0:
                for idx in list(range(0,nr,8))+[nr]:
                    actual=session.run(['actions'],{'obs':deploy_obs[idx:idx+1]})[0][0]
                    max_onnx_error=max(max_onnx_error,float(np.abs(actual-deploy_expected[idx]).max()))
                    target=actual*deploy_cfg['action_scale']+defaults[l_from_t]
                    max_target_error=max(max_target_error,float(np.abs(target-native_targets[idx,l_from_t]).max()))
                    np.testing.assert_allclose(actual,deploy_expected[idx],atol=2e-4,rtol=2e-4)
                    samples.append(deploy_obs[idx]);expected.append(deploy_expected[idx]);targets.append(native_targets[idx,motor_from_t])
            last_deploy=deploy_expected
            max_action=max(max_action,float(np.abs(last_deploy).max()))
            command=env.command_manager.get_command('twist')
            command[nr:,0]=[.5,1.,2.][min(step//50,2)]
            command[nr:,1]=.15
            command[nr:,2]=.2
            obs,_,done,_=_unpack_step(*wrapped.step(raw))
            assert not done.any()
    assert max_action<100, 'Deployment output clipping would change evaluated actions'
    vectors=args.output.with_suffix('.vectors.npz')
    np.savez_compressed(vectors,obs=np.asarray(samples),actions=np.asarray(expected),motor_targets=np.asarray(targets))
    manifest={'task':task,'checkpoint':str(args.checkpoint),'checkpoint_sha256':sha(args.checkpoint),
        'onnx_sha256':sha(args.output),'validation_bank_sha256':sha(args.bank),'validation_bank_indices':selected,
        'input':{'name':'obs','shape':[1,384],'history':'4 frames oldest to newest, repeated initial frame','normalizer_embedded':True},
        'output':{'name':'actions','shape':[1,29],'units':'legacy AMP raw actions; multiply by existing 0.25'},
        'training_joint_names':joint_names,'deployment_motor_names':motor_names,'deployment_policy_joint_names':legacy_names,
        'training_from_legacy':t_from_l,'legacy_from_training':l_from_t,
        'training_action_scale':scale.tolist(),'training_default_positions':defaults.tolist(),
        'training_kps_motor_order':kps,'training_kds_motor_order':kds,
        'compatible_config':{k:deploy_cfg[k] for k in ['obs_dim_per_frame','history_len','ang_vel_scale','dof_pos_scale','dof_vel_scale','action_scale','joint2motor_idx','default_dof_pos','kps','kds']},
        'verification':{'rollout_envs':n,'rollout_steps':151,'samples':len(samples),'max_observation_error':max_obs_error,
            'max_onnx_action_error':max_onnx_error,'max_joint_target_error_rad':max_target_error,'max_deployment_action':max_action,
            'vectors_sha256':sha(vectors)},
        'limitations':['PD gains differ only by existing YAML rounding (<0.1%).',
            'Training torque-speed/friction model and deployment torque clipping are different; sim-to-sim testing is separate.',
            'No hardware evaluation.'],'source_sha256':{f:sha(f) for f in ['src/tasks/amp_loco/amp_env_cfg.py',
                'src/tasks/amp_loco/config/g1/env_cfgs.py','src/assets/robots/unitree_g1/g1_constants.py',
                'rsl_rl/runners/amp_on_policy_runner.py','runtime/mjlab/managers/observation_manager.py']}}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['verification'],indent=2));env.close()


if __name__=='__main__':main()
