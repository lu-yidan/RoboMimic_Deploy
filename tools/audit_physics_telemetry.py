"""Verify full-resolution telemetry, sampling alignment and deterministic replay."""
import argparse,json
from pathlib import Path
import numpy as np
from tools.continue_vector_full_cycle_training import sha
from training.climb_robustness import GROUPS


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    s=json.loads((a.output/'summary.json').read_text());rows=[]
    assert s['exact_reproduction_passed']
    for result in s['trajectories']:
        case=result['case'];trace=a.output/case/'physics_telemetry.npz';trajectory=a.output/f'after_{case}.npz'
        assert sha(trace)==result['physics_telemetry_sha256'] and sha(trajectory)==result['trajectory_sha256']
        with np.load(trace) as z,np.load(trajectory) as x:
            t=z['time_s'];idx=np.rint(x['time_s'][1:]/.0005).astype(int)-1
            checks=dict(finite=all(np.isfinite(z[k]).all() for k in z.files),
                cadence=bool(np.all(abs(np.diff(t)-.0005)<1e-9)),
                start=bool(abs(t[0]-.0005)<1e-9),complete=bool(abs(t[-1]-result['time_s'])<1e-9),
                sample_count=len(t)==result['physics_samples'],
                time_alignment=bool(np.all(abs(t[idx]-x['time_s'][1:])<1e-9)),
                velocity_alignment=bool(np.array_equal(z['post_velocity'][idx],x['qvel'][1:,:6])),
                torque_alignment=bool(np.array_equal(z['motor_torque_nm'][idx],x['torques'])))
            final=t>111.66;force=z['group_force_world_n'][final];acc=z['solver_qacc'][final]
            rows.append(dict(case=case,passed=all(checks.values()),checks=checks,physics_samples=len(t),
                final_support_vertical_mean_n={g:float(force[:,i,2].mean()) for i,g in enumerate(GROUPS)},
                final_support_vertical_minimum_n={g:float(force[:,i,2].min()) for i,g in enumerate(GROUPS)},
                final_root_acceleration_rms_m_s2=float(np.sqrt(np.mean(acc[:,:3]**2))),
                whole_cycle_root_acceleration_max_m_s2=float(np.linalg.norm(z['solver_qacc'][:,:3],axis=1).max()),
                trace_sha256=sha(trace)))
    report=dict(passed=all(r['passed'] for r in rows),rows=rows,summary_sha256=sha(a.output/'summary.json'),source_sha256=sha(__file__),
        scope='2000 Hz solver-stage acceleration/contact force, post-integration time and velocity. Matched 50 Hz records and exact original rollout. Support-force minima are diagnostics, not replacement success gates.')
    (a.output/'telemetry_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    assert report['passed']

if __name__=='__main__':main()
