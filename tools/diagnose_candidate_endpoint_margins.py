"""Locate limiting endpoint constraints in an audited single-cycle candidate."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.full_cycle_residual_env import FullCycleResidualEnv
from tools.compare_audited_closure_candidates import load_run, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verified = load_run(args.experiment)
    manifest = json.loads((args.experiment/'manifest.json').read_text())
    audit_paths = [args.experiment/name for name in ('independent_candidate_audit.json', 'independent_training_audit.json') if (args.experiment/name).exists()]
    assert len(audit_paths) == 1
    audit = json.loads(audit_paths[0].read_text())
    env = FullCycleResidualEnv(plan=manifest['reference'], cases=manifest['evaluation_cases'])
    assert len(env.ends) == 1
    initial = env.plan['qpos'][0]
    limits = dict(joint_closure_rad=.04, effector_error_m=.01, tilt_deg=3.,
                  linear_speed_m_s=.01, angular_speed_rad_s=.1, joint_speed_rad_s=.1)
    rows = []
    try:
        for case, accepted in verified['rows'].items():
            record = next(r for r in audit['trajectories'] if r['label']=='after' and r['case']==case)
            if not accepted['complete'] or not record['independent_endpoints']:
                rows.append(dict(case=case, success=accepted['success'], endpoint_available=False))
                continue
            endpoint = record['independent_endpoints'][0]
            path = args.experiment/f'after_{case}.npz'
            assert sha(path) == record['trajectory_sha256']
            with np.load(path) as z:
                mask = (z['time_s']>env.horizon-2.) & (z['time_s']<=env.horizon+1e-7)
                q = z['qpos'][mask]; times = z['time_s'][mask]
            assert len(q) == endpoint['sample_count'] and len(q)>=99
            errors = q[:,env.env.qadr]-initial[env.env.qadr]
            maximum = np.max(abs(errors),axis=0)
            joint_rows = []
            for j in np.argsort(maximum)[::-1][:5]:
                i = int(np.argmax(abs(errors[:,j])))
                joint_rows.append(dict(joint=env.env.model.joint(env.env.joints[j]).name,
                                       signed_peak_error_mrad=float(1000*errors[i,j]), time_s=float(times[i])))
            rise = q[:,2]-initial[2]
            slack = {key:limit-float(endpoint[key]) for key,limit in limits.items()}
            slack['root_rise_m'] = .01-float(np.max(abs(rise-.25)))
            normalized = {key:value/limits.get(key,.01) for key,value in slack.items()}
            rows.append(dict(case=case, success=accepted['success'], endpoint_available=True,
                             top_joint_errors=joint_rows, signed_slack=slack, normalized_slack=normalized,
                             limiting_reported_constraint=min(normalized,key=normalized.get),
                             root_rise_range_mm=[float(1000*rise.min()),float(1000*rise.max())],
                             trajectory_sha256=sha(path)))
        result = dict(rows=rows, experiment=str(args.experiment.resolve()), audit_sha256=verified['audit_sha256'],
                      source_sha256=sha(__file__),
                      scope='Deterministic recorded final-window margins and dominant joints. Original thresholds remain unchanged. This omits transient/contact margins and is not a robustness certificate or allowable sensor-noise estimate.')
        args.output.mkdir(exist_ok=False)
        (args.output/'summary.json').write_text(json.dumps(result,indent=2))
        (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
        print(json.dumps([dict(case=r['case'],constraint=r['limiting_reported_constraint'],joint=r['top_joint_errors'][0],height_margin_mm=1000*r['signed_slack']['root_rise_m']) for r in rows if r['endpoint_available']]))
    finally:
        env.close()


if __name__ == '__main__':
    main()
