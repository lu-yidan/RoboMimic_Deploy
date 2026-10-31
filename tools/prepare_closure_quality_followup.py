"""Freeze one reward follow-up using the completed same-budget control."""
import json
from pathlib import Path
from tools.continue_vector_full_cycle_training import sha


def main():
    control = Path('outputs/selected_continuation98304_20261023_v1')
    gate = Path('outputs/closure_quality_gate_20261024_v1')
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
    checks = json.loads((gate / 'summary.json').read_text())
    assert checks['passed']
    protocol['gate'] = str(gate.resolve())
    protocol['gate_summary_sha256'] = sha(gate / 'summary.json')
    protocol['sources'].update({name: sha(name) for name in (
        'training/closure_quality_reward.py', 'tools/check_closure_quality_reward.py',
        'tools/train_closure_quality_followup.py', 'tools/prepare_closure_quality_followup.py')})
    protocol['control_reuse'] = str(control.resolve())
    protocol['control_reuse_hashes'] = {name: sha(control / name) for name in (
        'manifest.json', 'summary.json', 'independent_training_audit.json', 'policy.zip', 'after.json')}
    protocol.pop('closure_cost_parameters')
    protocol['quality_bonus_parameters'] = dict(joint_free_rad=.032, joint_scale_rad=.008,
        height_free_m=.006, height_scale_m=.004, ramp_seconds=3., weight=.1,
        formula='0.5*weight*ramp*(1/(1+joint_excess^2)+1/(1+height_excess^2))')
    protocol['weights'] = [.1]
    protocol['scope'] = ('Single reward follow-up from the same selected parent, seed, optimizer and budget '
        'as the completed continuation. Control outcome was known before this follow-up; not a prospective '
        'two-arm trial. Positive bonus changes objective and is not potential shaping. Oracle geometry, '
        'same five known cases; no held-out claim.')
    protocol['selection'] = ('Require complete training, five-case audit and no loss of parent successes. '
        'Run original twelve-case selection grid only if at least one new known-case success. '
        'No automatic promotion, no threshold changes.')
    out = Path('outputs/closure_quality_protocol_20261024_v1')
    out.mkdir(exist_ok=False)
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(out / 'protocol.json')


if __name__ == '__main__':
    main()
