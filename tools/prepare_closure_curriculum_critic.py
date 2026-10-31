"""Declare critic-only initialization comparison after completed suffix control."""
import json
from pathlib import Path
from tools.continue_vector_full_cycle_training import sha


def main():
    control = Path('outputs/closure_curriculum98304_20261024_v2')
    summary = json.loads((control/'summary.json').read_text())
    audit = json.loads((control/'independent_training_audit.json').read_text())
    prefix = json.loads((control/'composite_prefix_audit.json').read_text())
    assert audit['passed'] and prefix['passed'] and not summary['evaluation_pending']
    assert summary['additional_steps'] == 98304
    for name, key in [('policy.zip','checkpoint_sha256'),('manifest.json','experiment_manifest_sha256'),('summary.json','experiment_summary_sha256')]:
        assert sha(control/name) == audit[key]
    warm = Path('outputs/closure_critic_warmstart_20261024_v2')
    ws = json.loads((warm/'summary.json').read_text())
    assert ws['passed'] and ws['frozen_actor_and_features_exact']
    assert sha(warm/'policy_initial.zip') == ws['checkpoint_sha256']
    assert json.loads((warm/'adoption.json').read_text())['eligible_for_training_comparison']
    protocol = json.loads((control/'matched_protocol.json').read_text())
    assert ws['parent_checkpoint_sha256'] == protocol['parent_checkpoint_sha256']
    protocol.update(control_reuse=str(control.resolve()),
        control_reuse_hashes={name:sha(control/name) for name in ('manifest.json','summary.json','independent_training_audit.json','policy.zip','after.json','composite_prefix_audit.json')},
        critic_initialization=str(warm.resolve()), critic_initialization_sha256=sha(warm/'policy_initial.zip'),
        critic_calibration_summary_sha256=sha(warm/'summary.json'),
        scope='Same suffix curriculum, parent actor, seed, reward, exploration and PPO budget as completed control. Only critic heads and their optimizer moments differ at initialization. Extra calibration data: 21000 simulator steps, 6000 supervised updates; not equal total compute/sample budget. Validation episodes selected the calibration checkpoint; five known cases only.')
    protocol['sources'].update({name:sha(name) for name in ('tools/train_closure_curriculum_critic.py','tools/prepare_closure_curriculum_critic.py','tools/warmstart_closure_critic_v2.py')})
    out=Path('outputs/closure_curriculum_critic_protocol_20261024_v1')
    out.mkdir(exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(out/'protocol.json')


if __name__=='__main__':main()
