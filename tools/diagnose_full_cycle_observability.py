"""Measure how long geometry variants produce identical recorded actor inputs.

Identical input histories cannot identify which variant is present. They do
not rule out one conservative policy that succeeds on several variants.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=Path)
    args = parser.parse_args()
    audit = json.loads((args.experiment/'independent_training_audit.json').read_text())
    summary = json.loads((args.experiment/'summary.json').read_text())
    if not audit['passed'] or sha(args.experiment/'summary.json') != audit['experiment_summary_sha256']:
        raise ValueError('Completed audited experiment required')
    results = []
    hashes = {}
    for label in ('zero', 'after'):
        baseline_path = args.experiment/f'{label}_nominal.npz'
        baseline_record = next(row for row in summary[label] if row['case'] == 'nominal')
        if sha(baseline_path) != baseline_record['trajectory_sha256']:
            raise ValueError('Nominal trajectory changed')
        with np.load(baseline_path) as nominal:
            for row in summary[label]:
                if row['case'] == 'nominal':
                    continue
                path = args.experiment/f"{label}_{row['case']}.npz"
                if sha(path) != row['trajectory_sha256']:
                    raise ValueError('Variant trajectory changed')
                hashes[path.name] = sha(path)
                with np.load(path) as variant:
                    count = min(len(nominal['observations']), len(variant['observations']))
                    if not np.allclose(nominal['time_s'][:count], variant['time_s'][:count], rtol=0., atol=1e-8):
                        raise ValueError('Actor input timestamps are not aligned')
                    differences = abs(nominal['observations'][:count]-variant['observations'][:count]).max(axis=1)
                    exact_different = np.flatnonzero(differences != 0.)
                    meaningful_different = np.flatnonzero(differences > 1e-6)
                    first = int(exact_different[0]) if len(exact_different) else count
                    feature_differences = (np.flatnonzero(nominal['observations'][first] != variant['observations'][first]).tolist()
                                           if first < count else [])
                    results.append(dict(label=label, case=row['case'], runtime_success=row['success'], reason=row['reason'],
                        terminal_time_s=row['time_s'], compared_actor_inputs=count,
                        exact_common_prefix_steps=first,
                        last_identical_input_time_s=float(variant['time_s'][first-1]) if first else None,
                        first_different_input_time_s=float(variant['time_s'][first]) if first < count else None,
                        first_difference_above_1e_minus6_time_s=float(variant['time_s'][meaningful_different[0]]) if len(meaningful_different) else None,
                        first_different_features=feature_differences,
                        all_variant_inputs_identical_to_nominal=bool(first == count == len(variant['observations'])),
                        identical_prefix_actions=bool(np.array_equal(nominal['actions'][:first], variant['actions'][:first]))))
    result = dict(results=results, trajectory_hashes=hashes, audit_sha256=sha(args.experiment/'independent_training_audit.json'),
        source_sha256=sha(__file__), scope='Recorded actor-input histories only. Exact common prefixes demonstrate missing variant-specific information during that interval, not impossibility of a shared robust action or evidence about an untested visual sensor.')
    (args.experiment/'observability_diagnostic.json').write_text(json.dumps(result, indent=2))
    (args.experiment/'observability_diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
