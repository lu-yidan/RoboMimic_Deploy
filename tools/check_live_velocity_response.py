"""Distinguish solver-stage acceleration from integrated velocity change."""
import argparse,json
from pathlib import Path
import numpy as np
from tools.continue_vector_full_cycle_training import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    with np.load(args.run/'live_trace.npz') as z:
        t=z['post_step_time'];acc=np.diff(z['post_step_qvel'],axis=0)/np.diff(t)[:,None]
        mask=t[1:]>=102.66
        predicted=z['qp_prediction'][1:]
        solver=z['solver_qacc'][1:]
        result=dict(integrated_root_acceleration_mean=float(acc[mask,2].mean()),
            predicted_root_acceleration_mean=float(predicted[mask,2].mean()),
            root_response_error_rms=float(np.sqrt(np.mean((acc[mask,2]-predicted[mask,2])**2))),
            maximum_solver_integration_root_difference=float(abs(acc[:,2]-solver[:,2]).max()),
            maximum_solver_integration_joint_difference=float(abs(acc[:,6:]-solver[:,6:]).max()),
            trace_sha256=sha(args.run/'live_trace.npz'),source_sha256=sha(__file__),
            scope='Acceleration from consecutive post-step velocities is the actual interval response. Solver-stage acceleration can differ under implicitfast integration; both are retained. No claim of exact equality.')
    target=args.run/'velocity_response_audit.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2));(args.run/'velocity_response_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
