"""Verify telemetry up to an actual failure; never interpret it as task success."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.compare_audited_closure_candidates import sha
from training.climb_robustness import GROUPS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    summary = json.loads((args.output/'summary.json').read_text())
    assert summary['exact_reproduction_passed']
    parent = Path(summary['parent_experiment'])
    manifest = json.loads((parent/'manifest.json').read_text())
    assert sha(parent/'manifest.json') == summary['parent_manifest_sha256']
    assert sha(manifest['reference']) == manifest['reference_sha256']
    rows = []
    for result in summary['trajectories']:
        assert not result['success'] and result['reason'] != 'success'
        case = result['case']
        trace = args.output/case/'physics_telemetry.npz'
        trajectory = args.output/f'after_{case}.npz'
        assert sha(trace) == result['physics_telemetry_sha256']
        assert sha(trajectory) == result['trajectory_sha256']
        with np.load(trace) as z, np.load(trajectory) as x, np.load(manifest['reference']) as plan:
            times = z['time_s']
            indices = np.rint(x['time_s'][1:]/.0005).astype(int)-1
            checks = dict(finite=all(np.isfinite(z[key]).all() for key in z.files),
                cadence=bool(np.all(abs(np.diff(times)-.0005)<1e-9)),
                start=bool(abs(times[0]-.0005)<1e-9),
                recorded_failure_end=bool(abs(times[-1]-result['time_s'])<1e-9),
                shorter_than_reference=bool(times[-1] < plan['time'][-1]),
                sample_count=len(times)==result['physics_samples'],
                time_alignment=bool(np.all(abs(times[indices]-x['time_s'][1:])<1e-9)),
                velocity_alignment=bool(np.array_equal(z['post_velocity'][indices],x['qvel'][1:,:6])),
                torque_alignment=bool(np.array_equal(z['motor_torque_nm'][indices],x['torques'])))
            mask = times > times[-1]-2.
            command_time = ((round(float(times[-1])/.0005)-1)//10)*.005
            phase_index = min(max(int(np.searchsorted(plan['time'], command_time, side='right')-1), 0), len(plan['time'])-2)+1
            terminal_phase = str(plan['phase'][phase_index])
            norms = np.linalg.norm(z['group_force_world_n'], axis=2)
            group_peaks = {}
            for i, group in enumerate(GROUPS):
                peak = np.flatnonzero(mask)[np.argmax(norms[mask, i])]
                group_peaks[group] = dict(norm_n=float(norms[peak, i]), time_s=float(times[peak]),
                                         last_force_world_n=z['group_force_world_n'][-1, i].tolist())
            rows.append(dict(case=case, passed=all(checks.values()), checks=checks,
                task_success=False, reason=result['reason'], failure_time_s=float(times[-1]),
                inferred_terminal_controller_phase=terminal_phase,
                physics_samples=len(times), last_two_recorded_seconds_group_force=group_peaks,
                trace_sha256=sha(trace)))
    report = dict(passed=all(row['passed'] for row in rows), rows=rows,
        summary_sha256=sha(args.output/'summary.json'), source_sha256=sha(__file__),
        scope='2000 Hz telemetry reproduces the declared failure. Last two recorded seconds are pre-failure diagnostics, not the task closure window. Group forces sum contacts and do not identify individual contacts.')
    destination = args.output/'failure_telemetry_audit.json'
    assert not destination.exists()
    destination.write_text(json.dumps(report, indent=2))
    (args.output/'failure_telemetry_audit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report))
    assert report['passed']


if __name__ == '__main__':
    main()
