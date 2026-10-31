"""Replay transferred policy actions and independently check cycle observation roles."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.closure_composite_policy import ClosureCompositePolicy
from training.closure_handoff_reference_control import handoff_fade
from tools.audit_multirung_cycle import sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    m = json.loads((a.run/'manifest.json').read_text())
    summary = json.loads((a.run/'summary.json').read_text())
    physical = json.loads((a.run/'independent_audit.json').read_text())
    assert sha(a.run/'trajectory.npz') == summary['trajectory_sha256'] == physical['trajectory_sha256']
    assert all(sha(a.run/'source_snapshot'/name) == digest for name, digest in m['sources'].items())
    parent = Path(m['parent_experiment'])
    pm = json.loads((parent/'manifest.json').read_text())
    assert sha(parent/'policy.zip') == m['checkpoint_sha256']
    model = ClosureCompositePolicy(PPO.load(parent/'policy.zip', device='cpu'), pm)
    with np.load(a.run/'trajectory.npz') as z:
        data = {key: z[key].copy() for key in z.files}
    with np.load(a.run/'reference.npz') as z:
        ends = z['cycle_ends_s'].copy()
    assert len(ends) >= 2
    expected_roles = [[f'rung_{row+cycle:02d}' for row in (1, 2, 5, 6)] for cycle in range(len(ends))]
    assert m['geometry_observation_roles'] == expected_roles
    boundary = round(float(ends[0])/.02)
    expected_actions = []
    role_error, clock_error = 0., 0.
    for i, obs in enumerate(data['observations']):
        t = float(data['time_s'][i])
        index = min(int(np.searchsorted(ends, t, side='right')), len(ends)-1)
        start = 0. if index == 0 else float(ends[index-1])
        clock = np.float32(max(0., 1.-(t-start)/(float(ends[index])-start)))
        clock_error = max(clock_error, abs(float(obs[105])-float(clock)))
        roles = m['geometry_observation_roles'][index]
        offsets = np.array([m['offsets'].get(row, [0., 0., 0.]) for row in roles]).reshape(-1)/.02
        if pm['geometry_mode'] == 'blind':
            offsets[:] = 0.
        role_error = max(role_error, float(abs(obs[136:]-offsets).max()))
        action = model.predict(obs, deterministic=True)[0]
        if i >= boundary:
            weight = handoff_fade(t, float(ends[0]), m['fade_seconds'])[0]
            action = (weight*data['actions'][boundary-1]+(1.-weight)*action).astype(np.float32)
        expected_actions.append(action)
    action_error = float(abs(np.asarray(expected_actions)-data['actions']).max())
    preserved = np.r_[np.arange(105), np.arange(106, 148)]
    checks = dict(actions_exact=action_error == 0., bounded_actions=bool(abs(data['actions']).max() <= 1.),
        cycle_local_clock_exact=clock_error == 0., cycle_relative_geometry=role_error < 1e-7,
        other_observations_unchanged=bool(np.array_equal(data['observations'][:, preserved], data['raw_observations'][:, preserved])),
        first_cycle_prefix_exact=summary['first_cycle_prefix_exact'])
    result = dict(passed=all(checks.values()), checks=checks, action_error=action_error,
        geometry_feature_error=role_error, remaining_time_error=clock_error,
        physical_task_passed=physical['passed'], trajectory_sha256=sha(a.run/'trajectory.npz'),
        manifest_sha256=sha(a.run/'manifest.json'), source_sha256=sha(__file__),
        scope='Saved-policy action and observation-interface audit. Physical acceptance is independently reported; '
              'the actor is frozen and this is transfer, not multicycle training or perception validation.')
    destination = a.run/'policy_transfer_audit.json'
    assert not destination.exists()
    destination.write_text(json.dumps(result, indent=2))
    (a.run/'policy_transfer_audit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result), flush=True)
    assert result['passed']


if __name__ == '__main__':
    main()
