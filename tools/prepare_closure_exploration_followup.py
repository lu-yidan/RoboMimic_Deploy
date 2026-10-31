"""Conditionally freeze an exploration-only follow-up after reward evaluation."""
import json
from pathlib import Path
import numpy as np
from tools.continue_vector_full_cycle_training import sha


def main():
    control = Path('outputs/closure_quality98304_20261024_v1')
    comparison = Path('outputs/closure_quality_comparison_20261024_v1/comparison.json')
    c = json.loads(comparison.read_text())
    if c['eligible_for_grid']:
        raise ValueError('Reward candidate gained success: validate it before changing exploration')
    protocol = json.loads((control / 'matched_protocol.json').read_text())
    audit = json.loads((control / 'independent_training_audit.json').read_text())
    summary = json.loads((control / 'summary.json').read_text())
    manifest = json.loads((control / 'manifest.json').read_text())
    assert audit['passed'] and not summary['evaluation_pending']
    assert summary['additional_steps'] == protocol['steps'] == 98304
    assert manifest['seed'] == protocol['seed'] == 20261023
    for name, key in [('policy.zip', 'checkpoint_sha256'),
                      ('manifest.json', 'experiment_manifest_sha256'),
                      ('summary.json', 'experiment_summary_sha256')]:
        assert sha(control / name) == audit[key]
    with np.load(manifest['reference']) as plan:
        start = float(plan['time'][np.flatnonzero(plan['phase'] == 'posture_closure')[0] - 1])
        assert abs(start - 99.66) < 1e-6 and abs(float(plan['time'][-1]) - 113.66) < 1e-6
    gate = Path('outputs/closure_exploration_gate_20261024_v1')
    g = json.loads((gate / 'summary.json').read_text())
    assert g['passed'] and g['policy_source_sha256'] == sha('training/closure_exploration_policy.py')
    protocol.update(control_reuse=str(control.resolve()),
        control_reuse_hashes={name: sha(control / name) for name in (
            'manifest.json', 'summary.json', 'independent_training_audit.json', 'policy.zip', 'after.json')},
        exploration_gate=str(gate.resolve()), exploration_gate_sha256=sha(gate / 'summary.json'),
        trigger_comparison_sha256=sha(comparison),
        scope='Same-parent, same-seed, same-budget follow-up with identical positive quality reward. Only terminal leg exploration changes. Completed reward-only control is explicitly reused, its outcome was known. Deterministic initial actions and optimizer parameters match; stochastic distributions intentionally differ. No held-out or causal statistical claim.')
    protocol['sources'].update({name: sha(name) for name in (
        'training/closure_exploration_policy.py', 'tools/check_closure_exploration_policy.py',
        'tools/train_closure_exploration_followup.py', 'tools/prepare_closure_exploration_followup.py')})
    out = Path('outputs/closure_exploration_protocol_20261024_v1')
    out.mkdir(exist_ok=False)
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(out / 'protocol.json')


if __name__ == '__main__':
    main()
