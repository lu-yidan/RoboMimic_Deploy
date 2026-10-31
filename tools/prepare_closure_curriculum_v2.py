"""Freeze suffix training only after full-cycle exploration and cache gates."""
import json
from pathlib import Path
from tools.continue_vector_full_cycle_training import sha


def main():
    control = Path('outputs/closure_exploration98304_20261024_v1')
    comparison = Path('outputs/closure_exploration_comparison_20261024_v1/comparison.json')
    comparison_data = json.loads(comparison.read_text())
    if comparison_data['eligible_for_grid']:
        raise ValueError('Explore candidate gained success: finish its grid validation first')
    protocol = json.loads((control / 'matched_protocol.json').read_text())
    audit = json.loads((control / 'independent_training_audit.json').read_text())
    summary = json.loads((control / 'summary.json').read_text())
    assert audit['passed'] and not summary['evaluation_pending']
    assert summary['additional_steps'] == protocol['steps'] == 98304
    for name, key in [('policy.zip', 'checkpoint_sha256'),
                      ('manifest.json', 'experiment_manifest_sha256'),
                      ('summary.json', 'experiment_summary_sha256')]:
        assert sha(control / name) == audit[key]
    cache = Path('outputs/closure_prefix_cache_20261024_v4')
    gate = Path('outputs/closure_curriculum_env_gate_20261024_v2')
    cg = json.loads((cache / 'summary.json').read_text())
    eg = json.loads((gate / 'summary.json').read_text())
    cp = json.loads((cache / 'protocol.json').read_text())
    assert cg['passed'] and eg['passed']
    assert eg['prefix_cache_summary_sha256'] == sha(cache / 'summary.json')
    assert eg['environment_source_sha256'] == sha('training/closure_curriculum_env_v3.py')
    assert cp['parent_checkpoint_sha256'] == protocol['parent_checkpoint_sha256']
    assert cp['cases'] == protocol['training_cases']
    memory_gate = Path('outputs/closure_reset_memory_gate_20261024_v2/summary.json')
    mg = json.loads(memory_gate.read_text())
    assert mg['passed'] and mg['environment_source_sha256'] == sha('training/closure_curriculum_env_v3.py')
    protocol['reset_validation_gates'] = {str(memory_gate): sha(memory_gate), str(gate/'summary.json'): sha(gate/'summary.json')}
    protocol['restart_reason'] = 'Prior suffix run interrupted near 77824 steps by native allocation failure; fresh full budget from original selected parent, no partial checkpoint reuse.'
    protocol.update(control_reuse=str(control.resolve()),
        control_reuse_hashes={name: sha(control / name) for name in (
            'manifest.json', 'summary.json', 'independent_training_audit.json', 'policy.zip', 'after.json')},
        prefix_cache=str(cache.resolve()), prefix_cache_summary_sha256=sha(cache / 'summary.json'),
        curriculum_env_gate_sha256=sha(gate / 'summary.json'),
        trigger_comparison_sha256=sha(comparison),
        scope='Suffix-only training with a frozen parent prefix at deployment. Same parent, seed, PPO settings, quality reward and terminal leg exploration as the completed full-cycle control. Training samples now cover only 99.66..113.66 s. Resets restart OSQP caches with bounded measured differences; full-cycle evaluation never uses reset caches. Changes include reset distribution and frozen-prefix composition, so this is not a single-factor causal attribution. Five known cases, no held-out claim.')
    protocol['sources'].update({name: sha(name) for name in (
        'training/closure_curriculum_env.py', 'training/closure_composite_policy.py',
        'tools/check_closure_curriculum_env.py', 'tools/build_closure_prefix_cache_v4.py',
        'tools/train_closure_curriculum_v2.py', 'tools/prepare_closure_curriculum_v2.py',
        'training/closure_curriculum_env_v3.py', 'tools/check_closure_reset_memory_v2.py',
        'tools/check_closure_curriculum_env_v2.py',
        'tools/evaluate_closure_curriculum_policy.py', 'tools/audit_closure_curriculum_training.py',
        'tools/audit_closure_composite_prefix.py')})
    out = Path('outputs/closure_curriculum_protocol_20261024_v2')
    out.mkdir(exist_ok=False)
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(out / 'protocol.json')


if __name__ == '__main__':
    main()
