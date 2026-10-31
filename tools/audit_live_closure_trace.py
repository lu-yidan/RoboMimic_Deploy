"""Audit read-only continuous telemetry and compare optional frozen rollout."""
import argparse,json
from pathlib import Path
import numpy as np
from tools.continue_vector_full_cycle_training import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--original',type=Path);a=p.parse_args()
    s=json.loads((a.run/'summary.json').read_text())
    with np.load(a.run/'live_trace.npz') as f:z={k:f[k].copy() for k in f.files}
    t=z['post_step_time'];steady=t>=102.66
    checks=dict(trace_hash=sha(a.run/'live_trace.npz')==s['live_trace_sha256'],
        finite=all(np.isfinite(v).all() for v in z.values()),
        uniform_2000hz=bool(np.allclose(np.diff(t),.0005,rtol=0,atol=1e-9)),
        final_time=bool(abs(t[-1]-s['runtime']['time_s'])<1e-9))
    exact={}
    if a.original:
        with np.load(a.original/'trajectory.npz') as x,np.load(a.run/'trajectory.npz') as y:
            exact={k:bool(np.array_equal(x[k],y[k])) for k in ('qpos','qvel','time_s','actions','observations','torques','rewards')}
        checks['logging_does_not_change_rollout']=all(exact.values())
    discrepancy=z['solver_qacc'][steady]-z['qp_prediction'][steady]
    result=dict(checks=checks,passed=all(checks.values()),sample_count=len(t),exact_arrays=exact,
        steady_root_acceleration_error_rms=float(np.sqrt(np.mean(discrepancy[:,2]**2))) if steady.any() else None,
        steady_predicted_root_acceleration_mean=float(z['qp_prediction'][steady,2].mean()) if steady.any() else None,
        steady_actual_root_acceleration_mean=float(z['solver_qacc'][steady,2].mean()) if steady.any() else None,
        steady_joint_friction_norm_mean=float(np.linalg.norm(z['joint_friction_force'][steady],axis=1).mean()) if steady.any() else None,
        final_contact_world_force_mean=z['contact_world_force_sum'][t>111.66].mean(axis=0).tolist() if (t>111.66).any() else None,
        trace_sha256=sha(a.run/'live_trace.npz'),trajectory_sha256=sha(a.run/'trajectory.npz'),source_sha256=sha(__file__),
        scope='qvel/time are post integration; solver_qacc/forces belong to the preceding solver stage (time-dt). Prediction held at200Hz; error statistics exclude blend ramp. Contact sum is world-frame force on geom2; scene world geoms0..91 precede robot geoms92+, and live controller rejects unintended contacts. Telemetry audit does not itself establish task success.')
    dest=a.run/'live_trace_audit.json'
    if dest.exists():raise FileExistsError(dest)
    dest.write_text(json.dumps(result,indent=2));(a.run/'live_trace_audit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
