"""Reconstruct first-cycle calibrated actions and the declared classical fade."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.closure_action_bias_policy_v2 import ClosureActionBiasV2Policy
from training.closure_handoff_reference_control import handoff_fade
from tools.compare_audited_closure_candidates import load_run,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    m=json.loads((a.run/'manifest.json').read_text());s=json.loads((a.run/'summary.json').read_text())
    physical=json.loads((a.run/'independent_audit.json').read_text());parent=Path(m['parent_experiment'])
    verified=load_run(parent);pm=json.loads((parent/'manifest.json').read_text())
    assert verified['audit_sha256']==m['parent_audit_sha256'] and verified['checkpoint_sha256']==m['checkpoint_sha256']
    assert sha(a.run/'trajectory.npz')==s['trajectory_sha256']==physical['trajectory_sha256']
    assert all(sha(a.run/'source_snapshot'/f)==h for f,h in m['sources'].items())
    model=ClosureActionBiasV2Policy(PPO.load(parent/'policy.zip',device='cpu'),pm)
    with np.load(a.run/'reference.npz') as r:ends=r['cycle_ends_s'].copy()
    assert len(ends)==2
    with np.load(a.run/'trajectory.npz') as z:data={k:z[k].copy() for k in z.files}
    boundary=round(float(ends[0])/.02);actions=[];clock_error=0.
    for i,obs in enumerate(data['observations']):
        t=float(data['time_s'][i]);index=min(int(np.searchsorted(ends,t,side='right')),1);start=0. if index==0 else float(ends[0])
        clock=np.float32(max(0.,1.-(t-start)/(float(ends[index])-start)))
        clock_error=max(clock_error,abs(float(obs[105])-float(clock)))
        action=model.predict(obs,deterministic=True)[0] if i<boundary else (data['actions'][boundary-1]*handoff_fade(t,float(ends[0]),m['fade_seconds'])[0]).astype(np.float32)
        actions.append(action)
    exact=bool(np.array_equal(actions,data['actions']))
    checks=dict(actions_exact=exact,clock_exact=clock_error==0.,bounded_actions=bool(abs(data['actions']).max()<=1.),first_cycle_exact=s['first_cycle_prefix_exact'])
    result=dict(passed=all(checks.values()),checks=checks,physical_task_passed=physical['passed'],trajectory_sha256=sha(a.run/'trajectory.npz'),source_sha256=sha(__file__),
        scope='First-cycle calibrated policy followed by a six-second fade to zero residual for the classical second cycle. This is not learned-policy control of the second cycle.')
    (a.run/'classical_action_audit.json').write_text(json.dumps(result,indent=2));(a.run/'classical_action_audit_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result),flush=True)
    assert result['passed']


if __name__=='__main__':main()
