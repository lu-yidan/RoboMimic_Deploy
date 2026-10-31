"""Describe actual feedback authority and verify inactive cases stay unchanged."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.compare_audited_closure_candidates import load_run, sha
from tools.evaluate_seeded_reference_feedback import FEEDBACK_CONTRACT


def main():
    p = argparse.ArgumentParser()
    for name in ('baseline', 'candidate', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    baseline, candidate = load_run(a.baseline), load_run(a.candidate)
    m = json.loads((a.candidate/'manifest.json').read_text())
    assert m['observation_contract'] == FEEDBACK_CONTRACT
    assert baseline['cases'] == candidate['cases']
    assert baseline['checkpoint_sha256'] == m['teacher_checkpoint_sha256']
    seed = np.asarray(m['adaptive_reference_command'], dtype=float)
    rows = []
    for case in candidate['cases']:
        with np.load(a.candidate/f'after_{case}.npz') as z:
            actions = z['feedback_actions']
            trace = z['adaptive_reference_trace']
            gain = float(z['controller_geometry_gain'])
            raw = seed+m['feedback_delta_scale']*actions.astype(float)
            mapped = np.clip(raw, -1., 1.)
            inactive = {}
            if gain == 0.:
                with np.load(a.baseline/f'after_{case}.npz') as old:
                    inactive = {key: bool(np.array_equal(z[key], old[key])) for key in
                        ('qpos', 'qvel', 'time_s', 'actions', 'observations', 'torques', 'rewards',
                         'waist_reference_trace', 'body_reference_trace')}
                inactive['zero_actual_reference'] = bool(np.array_equal(trace[:, 6:21], np.zeros_like(trace[:, 6:21])))
            rows.append(dict(case=case, geometry_gain=gain, decisions=len(actions),
                success=candidate['rows'][case]['success'],
                mean_delta=actions.mean(axis=0).tolist() if len(actions) else None,
                maximum_absolute_delta=abs(actions).max(axis=0).tolist() if len(actions) else None,
                command_clip_fraction=np.mean(abs(raw)>1., axis=0).tolist() if len(actions) else None,
                maximum_command_change=abs(mapped-seed).max(axis=0).tolist() if len(actions) else None,
                maximum_filtered_reference=abs(trace[:, 11:16]).max(axis=0).tolist(),
                inactive_checks=inactive, passed=all(inactive.values())))
    result = dict(passed=all(r['passed'] for r in rows), rows=rows,
        coordinate_order=['waist_yaw_rad', 'waist_roll_rad', 'waist_pitch_rad', 'body_height_m', 'body_foreaft_m'],
        candidate_audit_sha256=candidate['audit_sha256'], baseline_audit_sha256=baseline['audit_sha256'],
        source_sha256=sha(__file__),
        scope='Audit of bounded reference usage and unchanged zero-gain cases, not a task-success gate. '
              'Nonzero actor deltas can be clipped by seed saturation or multiplied by zero geometry gain; '
              'actor updates alone do not prove a physical improvement.')
    a.output.mkdir(exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    assert result['passed']


if __name__ == '__main__':
    main()
