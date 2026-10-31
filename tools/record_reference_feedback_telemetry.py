"""Instrument one full feedback rollout, requiring exact original-array replay.

The evaluator's environment factory is wrapped only inside this dedicated
process. Its original control loop is reused unchanged and restored on exit.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from training.physics_telemetry import install_physics_telemetry
from tools import evaluate_seeded_reference_feedback as runner
from tools.compare_audited_closure_candidates import load_run, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--case', default='foot_z_plus15')
    a = p.parse_args()
    verified = load_run(a.experiment)
    m = json.loads((a.experiment/'manifest.json').read_text())
    assert m['observation_contract'] == runner.FEEDBACK_CONTRACT
    assert verified['rows'][a.case]['complete']
    for name, digest in m['sources'].items():
        assert sha(a.experiment/'source_snapshot'/name) == digest
        if name.replace('\\', '/').startswith('training/'):
            assert sha(runner.ROOT/name) == digest
    # The candidate's control loop must also match the exact source being reused.
    evaluator_source = Path(runner.__file__).resolve().relative_to(runner.ROOT)
    assert sha(runner.__file__) == m['sources'][str(evaluator_source)]
    a.output.mkdir(exist_ok=False)
    original_factory = runner.FullCycleResidualEnv
    captured = []

    def instrumented_factory(*args, **kwargs):
        env = original_factory(*args, **kwargs)
        captured.append(install_physics_telemetry(env))
        return env

    runner.FullCycleResidualEnv = instrumented_factory
    try:
        row = runner.evaluate_case((m, str((a.experiment/'policy.zip').resolve()),
            verified['checkpoint_sha256'], a.case, str((a.output/a.case).resolve())))
    finally:
        runner.FullCycleResidualEnv = original_factory
    assert len(captured) == 1
    telemetry = captured[0]
    physics_path = a.output/a.case/'physics_telemetry.npz'
    np.savez_compressed(physics_path, **{k: v[:telemetry['count']] for k, v in telemetry['data'].items()})
    name = f'after_{a.case}.npz'
    shutil.copy2(a.output/a.case/name, a.output/name)
    checks = {}
    with np.load(a.experiment/name) as old, np.load(a.output/name) as new:
        assert set(old.files) == set(new.files)
        for key in old.files:
            checks[key] = bool(np.array_equal(old[key], new[key]))
    row.update(physics_telemetry_sha256=sha(physics_path), physics_samples=telemetry['count'])
    (a.output/a.case/'result.json').write_text(json.dumps(row, indent=2))
    result = dict(parent_experiment=str(a.experiment.resolve()),
        parent_manifest_sha256=sha(a.experiment/'manifest.json'), parent_audit_sha256=verified['audit_sha256'],
        checkpoint_sha256=verified['checkpoint_sha256'], trajectories=[row],
        exact_reproduction_passed=all(checks.values()), comparisons=[dict(case=a.case, checks=checks)],
        source_sha256=sha(__file__), evaluator_source_sha256=sha(runner.__file__),
        telemetry_source_sha256=sha('training/physics_telemetry.py'),
        scope='One from-zero rollout with the frozen 10 Hz reference actor. Read-only 2000 Hz telemetry; '
              'all original teacher, feedback, state and reference arrays must reproduce exactly. No hardware claim.')
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    (a.output/'evaluation_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(case=a.case, physics_samples=telemetry['count'], exact=checks)), flush=True)
    assert result['exact_reproduction_passed']


if __name__ == '__main__':
    main()
