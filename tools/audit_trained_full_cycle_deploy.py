"""Verify the independent simulation entry point reproduces recorded inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('deployment', type=Path)
    args = parser.parse_args()
    provenance = json.loads((args.deployment/'provenance.json').read_text())
    summary = json.loads((args.deployment/'summary.json').read_text())
    experiment = Path(provenance['experiment'])
    original = experiment/f"after_{provenance['case']}.npz"
    replay = args.deployment/'trajectory.npz'
    audit = json.loads((experiment/'independent_training_audit.json').read_text())
    original_result = next(r for r in audit['trajectories'] if r['label'] == 'after' and r['case'] == provenance['case'])
    checks = dict(original_audit_unchanged=sha(experiment/'independent_training_audit.json') == provenance['integrity_audit_sha256'],
        experiment_manifest_unchanged=sha(experiment/'manifest.json') == provenance['experiment_manifest_sha256'],
        checkpoint_unchanged=sha(experiment/'policy.zip') == provenance['checkpoint_sha256'],
        original_trajectory_hash=sha(original) == original_result['trajectory_sha256'],
        deployment_trajectory_hash=sha(replay) == summary['trajectory_sha256'],
        independently_accepted=bool(summary['independently_accepted'] and original_result['independently_accepted']))
    differences = {}
    with np.load(original) as a, np.load(replay) as b:
        for key in ('qpos', 'qvel', 'time_s', 'actions', 'observations', 'torques', 'rewards'):
            same_shape = a[key].shape == b[key].shape
            checks[key+'_exact_match'] = bool(same_shape and np.array_equal(a[key], b[key]))
            differences[key] = float(abs(a[key]-b[key]).max()) if same_shape else None
    result = dict(passed=all(checks.values()), checks=checks, maximum_differences=differences,
        deployment_trajectory_sha256=sha(replay), original_trajectory_sha256=sha(original),
        audit_source_sha256=sha(__file__), scope='Separate simulation entry point; exact reproduction of the audited single-cycle policy evaluation, not hardware deployment.')
    (args.deployment/'deployment_reproduction_audit.json').write_text(json.dumps(result, indent=2))
    (args.deployment/'deployment_audit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result), flush=True)
    if not result['passed']:
        raise AssertionError('Deployment reproduction audit failed')
