"""Freeze a bounded continuation without modifying previous experiments."""
import json
from pathlib import Path
from tools.continue_vector_full_cycle_training import sha


def main():
    parent = Path('outputs/closure_margin_control98304_v1')
    protocol = json.loads((parent / 'matched_protocol.json').read_text())
    manifest = json.loads((parent / 'manifest.json').read_text())
    audit = json.loads((parent / 'independent_training_audit.json').read_text())
    assert audit['passed']
    for name, key in [('policy.zip', 'checkpoint_sha256'),
                      ('manifest.json', 'experiment_manifest_sha256'),
                      ('summary.json', 'experiment_summary_sha256')]:
        assert sha(parent / name) == audit[key], name
    assert sha(manifest['reference']) == manifest['reference_sha256']
    for path, digest in manifest['sources'].items():
        if path.replace('\\', '/').startswith('training/'):
            assert sha(path) == digest, path
    protocol.pop('policy_lineage_audit_sha256')
    protocol.update(parent=str(parent.resolve()),
        parent_checkpoint_sha256=sha(parent / 'policy.zip'),
        parent_audit_sha256=sha(parent / 'independent_training_audit.json'),
        steps=98304, seed=20261023, weights=[0.0],
        training_cases=manifest['training_cases'],
        selection='Retain previous release. Assess all five known cases with unchanged gates; any regression rejects promotion. Grid validation is required before any future promotion.',
        scope='Single bounded continuation, not a matched reward comparison or held-out generalization experiment.')
    protocol['sources'].update({p: sha(p) for p in [
        'tools/train_selected_policy_continuation.py',
        'tools/prepare_selected_continuation.py']})
    out = Path('outputs/selected_continuation_protocol_20261023_v1')
    out.mkdir(exist_ok=False)
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(out / 'protocol.json')


if __name__ == '__main__':
    main()
