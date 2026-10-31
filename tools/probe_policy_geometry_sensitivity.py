"""Frozen-state network sensitivity; never a rollout or estimated-input success."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.closure_action_bias_policy_v2 import ClosureActionBiasV2Policy
from tools.validate_swing_reference import read,write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--grid',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run=a.run.resolve();grid=read(a.grid);m=read(run/'manifest.json')
    audit=read(run/'v2_multicycle_audit.json');parent=Path(m['parent_experiment']);pm=read(parent/'manifest.json')
    assert audit['passed'] and audit['manifest_sha256']==sha(run/'manifest.json')
    assert audit['trajectory_sha256']==sha(run/'trajectory.npz')
    assert sha(parent/'policy.zip')==m['checkpoint_sha256']
    torch.set_num_threads(1)
    model=ClosureActionBiasV2Policy(PPO.load(parent/'policy.zip',device='cpu'),pm)
    with np.load(run/'trajectory.npz') as z:
        observations=z['observations'];actions=z['actions'];times=z['time_s'][:-1]
    with np.load(run/'reference.npz') as z:ends=z['cycle_ends_s']
    assert len(ends)==2 and len(grid['cases'])==16
    # The first six seconds after the seam blend a carried action. They are
    # excluded from this direct actor diagnostic, rather than misreconstructed.
    selected=(times<ends[0]-1e-7)|(times>=ends[0]+m['fade_seconds']+1e-7)
    ids=np.flatnonzero(selected);obs=observations[ids];t=times[ids]
    expected=model.predict(obs,deterministic=True)[0]
    assert np.array_equal(expected,actions[ids]),'Original actor must reproduce saved actions before counterfactual edits'
    cycle=np.where(t<ends[0],0,1);slots=np.where(cycle==0,3,2)
    assert m['geometry_observation_roles'][0][3]==m['geometry_observation_roles'][1][2]=='rung_06'
    original_offsets=np.asarray(m['offsets'].get('rung_06',[0.,0.,0.]))/.02
    assert np.allclose(np.stack([o[136+s*3:139+s*3] for o,s in zip(obs,slots)]),original_offsets,rtol=0,atol=1e-7)
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(run=str(run),trajectory_sha256=sha(run/'trajectory.npz'),manifest_sha256=sha(run/'manifest.json'),
        task_audit_sha256=sha(run/'v2_multicycle_audit.json'),checkpoint_sha256=m['checkpoint_sha256'],
        grid_protocol_sha256=sha(a.grid),source_sha256=sha(__file__),samples=len(ids),
        original_actions_exact=True,new_training_steps=0,new_physics_steps=0,
        scope='Frozen saved states and reference; only rung_06 geometry features are counterfactually replaced by the predeclared grid offsets. The six-second handoff blend is excluded. Action differences are normalized network-output differences, not motor torque, changed trajectories, success rates or perception validation. No parameters are selected from this diagnostic.',rows=[])
    write(a.output/'protocol.json',{k:v for k,v in report.items() if k!='rows'})
    for case,offsets in grid['cases'].items():
        modified=obs.copy();feature=np.asarray(offsets['rung_06'])/.02
        for slot in (2,3):modified[slots==slot,136+slot*3:139+slot*3]=feature
        proposed=model.predict(modified,deterministic=True)[0]
        delta=np.abs(proposed-expected);row=dict(case=case,estimated_rung06_offset_m=offsets['rung_06'],cycles={})
        for index in (0,1):
            values=delta[cycle==index]
            row['cycles'][str(index+1)]=dict(samples=len(values),
                maximum_absolute_normalized_action_difference=float(values.max()),
                rms_normalized_action_difference=float(np.sqrt(np.mean(values**2))),
                exact_action_fraction=float(np.mean(np.all(values==0,axis=1))))
        report['rows'].append(row);write(a.output/'partial.json',report)
        print(json.dumps(row),flush=True)
    write(a.output/'summary.json',report);(a.output/'analysis_source.py').write_bytes(Path(__file__).read_bytes())


if __name__=='__main__':main()
