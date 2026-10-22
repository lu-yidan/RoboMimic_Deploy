"""Run the bounded residual research policy inside the MuJoCo deployment entry."""
import hashlib
import json
from pathlib import Path
import time
import mujoco.viewer
import numpy as np
import torch
from stable_baselines3 import PPO
from training.rung_residual_env import RungResidualEnv, ROOT


def run_rung_residual(cfg):
    out=ROOT/str(cfg.output_dir)
    if out.exists():raise FileExistsError(out)
    checkpoint=ROOT/str(cfg.checkpoint)
    training=json.loads((checkpoint.parent/'summary.json').read_text())
    if not training['training_completed'] or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=training['checkpoint_sha256']:
        raise ValueError('Checkpoint integrity mismatch or incomplete training')
    manifest=json.loads((checkpoint.parent/'manifest.json').read_text())
    for rel,expected in manifest['sources'].items():
        if rel.startswith('training/') and hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=expected:
            raise ValueError('Research environment changed since training: '+rel)
    torch.set_num_threads(1);model=PPO.load(checkpoint,device='cpu')
    if model.observation_space.shape!=(77,) or model.action_space.shape!=(7,):
        raise ValueError('Expected rung residual 77D/7D policy, not the old deployment actor')
    e=RungResidualEnv();obs,_=e.reset(seed=20261010,options={'offset':list(cfg.offset)})
    out.mkdir(parents=True)
    states=[e.env.data.qpos.copy()];velocities=[e.env.data.qvel.copy()];times=[0.];actions=[];torques=[];rows=[]
    viewer=mujoco.viewer.launch_passive(e.env.model,e.env.data) if cfg.viewer else None
    try:
        for _ in range(e.max_steps):
            started=time.perf_counter();action,_=model.predict(obs,deterministic=True)
            obs,reward,terminated,truncated,info=e.step(action)
            states.append(e.env.data.qpos.copy());velocities.append(e.env.data.qvel.copy());times.append(float(e.env.data.time))
            actions.append(action);torques.append(e.last_torque.copy());rows.append(dict(reward=reward,**info))
            if viewer:
                if not viewer.is_running():info=dict(info,success=False,reason='viewer_closed');break
                viewer.sync();time.sleep(max(0.,.02-(time.perf_counter()-started)))
            if terminated or truncated:break
    finally:
        if viewer:viewer.close()
        e.close()
    np.savez_compressed(out/'trajectory.npz',qpos=states,qvel=velocities,time_s=times,actions=actions,torques=torques)
    (out/'control_log.json').write_text(json.dumps(rows,indent=2))
    (out/'scene_variant.json').write_text(json.dumps(e.variant,indent=2))
    (out/'summary.json').write_text(json.dumps(dict(final=info,checkpoint=str(checkpoint),checkpoint_sha256=training['checkpoint_sha256'],
                                                   full_cycle=False,hardware_connected=False),indent=2))
    print(json.dumps(info,indent=2))
