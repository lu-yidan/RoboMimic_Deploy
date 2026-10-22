"""JSON records and verified reuse of unchanged zero-policy evaluations."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def numpy_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported JSON record value: {type(value).__name__}')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reuse_zero_evaluations(source, destination, manifest, root):
    """Reuse only an identical contract/controller/reference/case evaluation.

    Trainer logging may change; every frozen training module must still match.
    The original manifest and evidence hashes are recorded, never overwritten.
    """
    source, destination, root = Path(source), Path(destination), Path(root)
    old = json.loads((source/'manifest.json').read_text())
    for key in ('contract', 'reference_sha256', 'evaluation_cases', 'normal_bias_y_deg', 'force_history_memory'):
        if old[key] != manifest[key]:
            raise ValueError('Zero baseline reuse differs: '+key)
    legacy_bundle_evidence = None
    old_bundle_hash = old.get('bundle_manifest_sha256')
    if old_bundle_hash is None:
        gate = json.loads((root/old['gate']).read_text())
        baseline = root/gate['baseline']
        if digest(baseline/'manifest.json') != gate['baseline_manifest_sha256']:
            raise ValueError('Legacy baseline manifest changed')
        audit_path = baseline/'independent_audit.json'
        prior_audit = json.loads(audit_path.read_text())
        if not prior_audit['passed']:
            raise ValueError('Legacy independent baseline was not accepted')
        old_bundle_hash = prior_audit['bundle_manifest_sha256']
        legacy_bundle_evidence = dict(path=str(audit_path), sha256=digest(audit_path))
    if old_bundle_hash != manifest['bundle_manifest_sha256']:
        raise ValueError('Zero baseline scene bundle changed')
    checked_sources = {}
    for relative, value in old['sources'].items():
        normalized = relative.replace('\\', '/')
        if normalized.startswith('training/'):
            if digest(source/'source_snapshot'/relative) != value or digest(root/relative) != value:
                raise ValueError('Zero baseline controller source changed: '+relative)
            checked_sources[relative] = value
    if not any(key.replace('\\', '/').endswith('training/full_cycle_residual_env.py') for key in checked_sources):
        raise ValueError('Missing frozen full-cycle environment')
    records = json.loads((source/'zero.json').read_text())
    if len(records) != len(manifest['evaluation_cases']) or {r['case'] for r in records} != set(manifest['evaluation_cases']):
        raise ValueError('Incomplete zero baseline case set')
    files = {}
    for record in records:
        name = 'zero_'+record['case']+'.npz'
        path = source/name
        if digest(path) != record['trajectory_sha256']:
            raise ValueError('Zero baseline trajectory hash mismatch')
        with np.load(path) as z:
            if not np.all(z['actions'] == 0.):
                raise ValueError('Requested zero baseline contains nonzero actions')
        files[name] = record['trajectory_sha256']
    for name in files:
        shutil.copy2(source/name, destination/name)
    shutil.copy2(source/'zero.json', destination/'zero.json')
    provenance = dict(source=str(source.resolve()), source_manifest_sha256=digest(source/'manifest.json'),
                      bundle_manifest_sha256=old_bundle_hash, legacy_bundle_evidence=legacy_bundle_evidence,
                      checked_training_sources=checked_sources, files=files,
                      note='Reused unchanged zero-policy evaluations; no new baseline dynamics run.')
    (destination/'zero_reuse_provenance.json').write_text(json.dumps(provenance, indent=2))
    return records
