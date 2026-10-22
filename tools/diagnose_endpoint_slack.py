"""Report signed endpoint slack without changing acceptance or claiming robustness."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.continue_vector_full_cycle_training import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=Path)
    args = parser.parse_args()
    p = args.experiment
    audit = json.loads((p/'independent_training_audit.json').read_text())
    manifest = json.loads((p/'manifest.json').read_text())
    assert audit['passed'] and sha(p/'manifest.json') == audit['experiment_manifest_sha256']
    assert sha(p/'summary.json') == audit['experiment_summary_sha256']
    with np.load(manifest['reference']) as plan:
        ends = np.asarray(plan.get('cycle_ends_s', [plan['time'][-1]])).copy()
        initial_z = float(plan['qpos'][0, 2])
    assert len(ends) == 1
    rows = []
    thresholds = dict(joint_closure_rad=.04, effector_error_m=.01, tilt_deg=3.,
                      linear_speed_m_s=.01, angular_speed_rad_s=.1, joint_speed_rad_s=.1)
    for record in audit['trajectories']:
        trajectory = p/f"{record['label']}_{record['case']}.npz"
        assert record['passed'] and sha(trajectory) == record['trajectory_sha256']
        endpoint = record['independent_endpoints'][0]
        if endpoint['sample_count'] < 99:
            rows.append(dict(case=record['case'], label=record['label'], accepted=False, endpoint_available=False))
            continue
        with np.load(trajectory) as z:
            times = z['time_s']
            mask = (times > ends[0]-2.) & (times <= ends[0]+1e-7)
            rise = z['qpos'][mask, 2]-initial_z
        assert len(rise) == endpoint['sample_count']
        slack = {key: limit-endpoint[key] for key, limit in thresholds.items()}
        slack['root_rise_m'] = .01-float(abs(rise-.25).max())
        normalized = {key: value/thresholds.get(key, .01) for key, value in slack.items()}
        rows.append(dict(case=record['case'], label=record['label'], accepted=record['independently_accepted'], endpoint_available=True,
                         signed_slack=slack, normalized_slack=normalized, root_rise_min_m=float(rise.min()), root_rise_max_m=float(rise.max()),
                         tightest_reported_constraint=min(normalized, key=normalized.get), trajectory_sha256=sha(trajectory)))
    result = dict(rows=rows, audit_sha256=sha(p/'independent_training_audit.json'), source_sha256=sha(__file__),
                  scope='Signed deterministic endpoint slack. These selected endpoint quantities omit loaded-contact margins and transient constraints; not a complete safety certificate, uncertainty bound, or noise tolerance. Acceptance remains the original independent audit.')
    target = p/'endpoint_slack_diagnostic.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2))
    (p/'endpoint_slack_diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps([dict(case=r['case'], joint_slack_mrad=1000*r['signed_slack']['joint_closure_rad'], height_slack_mm=1000*r['signed_slack']['root_rise_m']) for r in rows if r['label']=='after' and r.get('endpoint_available')], indent=2))


if __name__ == '__main__':
    main()
