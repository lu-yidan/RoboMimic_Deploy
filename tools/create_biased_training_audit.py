"""Derive a scenario-aware audit while preserving the contact baseline audit."""
from pathlib import Path
root = Path(__file__).resolve().parents[1]
s = (root / 'tools/audit_contact_load_transfer.py').read_text(encoding='utf-8')
s = s.replace("f'{label}_{name}.npz'", "f\"{label}_{row['scenario']}.npz\"")
s = s.replace("results.append(dict(label=label,case=name,", "results.append(dict(label=label,case=name,scenario=row['scenario'],evaluation_only_angle=row['evaluation_only_angle'],")
needle = "    result=dict(audit_source_sha256=sha(Path(__file__)),integrity_passed=all(r['integrity_passed'] for r in results),results=results,"
replacement = '''    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    action_replay={}
    for label,checkpoint in [('before','policy_before.zip'),('after','policy.zip')]:
        policy=PPO.load(a.run/checkpoint,device='cpu');difference=0.
        for row in summary[label]:
            z=dict(np.load(a.run/f"{label}_{row['scenario']}.npz"))
            predicted=policy.predict(z['observations'],deterministic=True)[0]
            difference=max(difference,float(np.max(abs(predicted-z['actions']))))
        action_replay[label]=dict(max_difference=difference,passed=difference<=2e-6,checkpoint_sha256=sha(a.run/checkpoint))
    checkpoint_ok=sha(a.run/'policy.zip')==summary['checkpoint_sha256']
    result=dict(audit_source_sha256=sha(Path(__file__)),integrity_passed=all(r['integrity_passed'] for r in results) and checkpoint_ok and all(r['passed'] for r in action_replay.values()),checkpoint_matches_summary=checkpoint_ok,action_replay=action_replay,results=results,'''
assert needle in s
s = s.replace(needle, replacement)
p = root / 'tools/audit_biased_load_transfer.py'
assert not p.exists()
p.write_text(s, encoding='utf-8')
