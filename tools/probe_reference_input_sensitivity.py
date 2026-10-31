"""Saved-state actor diagnostic for reference-dependent observation channels."""
import argparse
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.wall_support_env import WallSupportEnv
from training.closure_action_bias_policy_v2 import ClosureActionBiasV2Policy
from tools.validate_swing_reference import read, write, sha


def features(plan,times,qa,state):
    knots=plan['time'];q=plan['qpos']
    j=np.clip(np.searchsorted(knots,times,side='right')-1,0,len(knots)-2)
    dt=knots[j+1]-knots[j];u=np.clip((times-knots[j])/dt,0,1)
    reference=(1-u[:,None])*q[j]+u[:,None]*q[j+1]
    velocity=(q[j+1][:,qa]-q[j][:,qa])/dt[:,None]
    return (state[:,qa]-reference[:,qa]).astype(np.float32),(state[:,:3]-reference[:,:3]).astype(np.float32),velocity.astype(np.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    run=Path('outputs/v2_swing_two_telemetry_20261029_v1')
    m=read(run/'manifest.json');audit=read(run/'v2_multicycle_audit.json')
    assert audit['passed'] and audit['trajectory_sha256']==sha(run/'trajectory.npz')
    parent=Path(m['parent_experiment']);pm=read(parent/'manifest.json')
    new=Path('outputs/wholebody_hand_dynamic_protocol_20261029_v1/reference/plan.npz')
    assert sha(new)==read(new.parent/'manifest.json')['plan_sha256']
    with np.load(run/'trajectory.npz') as z:
        observations=z['observations'];actions=z['actions'];times=z['time_s'][:-1];state=z['qpos'][:-1]
    with np.load(run/'reference.npz') as z:original={k:z[k] for k in z.files}
    with np.load(new) as z:proposed={k:z[k] for k in z.files}
    selected=times>123.66+1e-7
    ids=np.flatnonzero(selected);obs=observations[ids];state=state[ids];times=times[ids]
    assert times.min()>original['cycle_ends_s'][0]+m['fade_seconds']+1e-7
    env=WallSupportEnv('outputs/wall_support_bundle_pd15_v2')
    try:qa=env.qadr.copy()
    finally:env.close()
    old_features=features(original,times,qa,state)
    spans=(slice(0,29),slice(61,64),slice(70,99))
    assert all(np.array_equal(value,obs[:,span]) for value,span in zip(old_features,spans))
    torch.set_num_threads(1)
    assert sha(parent/'policy.zip')==m['checkpoint_sha256']
    model=ClosureActionBiasV2Policy(PPO.load(parent/'policy.zip',device='cpu'),pm)
    expected=model.predict(obs,deterministic=True)[0]
    assert np.array_equal(expected,actions[ids])
    proposed_features=features(proposed,times,qa,state)
    protocol=dict(saved_run=str(run),trajectory_sha256=sha(run/'trajectory.npz'),
        original_reference_sha256=sha(run/'reference.npz'),new_reference_sha256=sha(new),
        checkpoint_sha256=m['checkpoint_sha256'],samples=len(ids),source_sha256=sha(__file__),
        original_features_exact=True,original_actions_exact=True,new_physics_steps=0,new_training_steps=0,
        scope='Same saved states and all other observations; replace reference-dependent tracking/base-error/velocity channels only. Deterministic actor output differences, not motor torque or a successful new trajectory. T12 states do not claim the target-shift pilot has completed.')
    write(a.output/'protocol.json',protocol)
    rows=[]
    for name,parts in [('position_only',(0,1)),('velocity_only',(2,)),('all_reference_channels',(0,1,2))]:
        changed=obs.copy()
        for part in parts:changed[:,spans[part]]=proposed_features[part]
        predicted=model.predict(changed,deterministic=True)[0]
        delta=abs(predicted-expected)
        rows.append(dict(intervention=name,samples=len(ids),maximum_absolute_normalized_action_difference=float(delta.max()),
            rms_normalized_action_difference=float(np.sqrt(np.mean(delta**2))),
            changed_action_frames=int(np.any(delta!=0,axis=1).sum())))
    write(a.output/'summary.json',dict(**protocol,rows=rows))
    (a.output/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    print(rows,flush=True)


if __name__=='__main__':main()
