"""Finalize saved telemetry after the legacy empty-array comparison error."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.compare_audited_closure_candidates import load_run, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verified = load_run(args.experiment)
    original = json.loads((args.experiment/'summary.json').read_text())
    assert not (args.output/'summary.json').exists()
    rows, comparisons = [], []
    for result_file in sorted(args.output.glob('*/result.json')):
        row = json.loads(result_file.read_text())
        case = row['case']
        assert not row['success'] and case in verified['rows']
        name = f'after_{case}.npz'
        previous = next(item for item in original['after'] if item['case'] == case)
        assert previous['reason'] == row['reason'] and previous['time_s'] == row['time_s']
        assert sha(args.experiment/name) == previous['trajectory_sha256']
        assert sha(args.output/name) == sha(result_file.parent/name) == row['trajectory_sha256']
        assert sha(result_file.parent/'physics_telemetry.npz') == row['physics_telemetry_sha256']
        with np.load(args.experiment/name) as old, np.load(args.output/name) as new:
            assert set(old.files) == set(new.files)
            checks = {key: bool(np.array_equal(old[key], new[key])) for key in old.files}
            differences = {key: float(np.max(np.abs(old[key]-new[key]), initial=0.)) for key in old.files}
        assert all(checks.values())
        rows.append(row)
        comparisons.append(dict(case=case, passed=True, checks=checks, maximum_differences=differences,
            original_sha256=previous['trajectory_sha256'], parallel_sha256=row['trajectory_sha256']))
    assert rows
    result = dict(parent_experiment=str(args.experiment.resolve()), checkpoint_sha256=verified['checkpoint_sha256'],
        parent_manifest_sha256=sha(args.experiment/'manifest.json'), trajectories=rows, comparisons=comparisons,
        cases=[row['case'] for row in rows], workers=1, exact_reproduction_passed=True,
        telemetry_source_sha256=sha('training/physics_telemetry.py'), source_sha256=sha(__file__),
        scope='Report recovered from already saved telemetry. Every original trajectory array is exact, including empty waist records. The failed task remains failed; no new simulation or changed thresholds.')
    (args.output/'summary.json').write_text(json.dumps(result, indent=2))
    (args.output/'report_repair_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(cases=result['cases'], exact_reproduction_passed=True)))


if __name__ == '__main__':
    main()
