"""Dense scratch-geometry validation before any smoothed-reference execution."""
import json
from pathlib import Path
import numpy as np
from training.smooth_unload_reference_env import SmoothUnloadReferenceEnv
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.plan_clearance_cycle import audit_plan
from tools.continue_vector_full_cycle_training import sha


def main():
    out=Path('outputs/smooth_unload_reference_gate_v1');out.mkdir(exist_ok=False)
    plan='outputs/highrod_endpoint_adapt_plan_v1/plan.npz';env=SmoothUnloadReferenceEnv(plan=plan);base=FullCycleResidualEnv(plan=plan)
    try:
        times=np.arange(round(env.horizon/.005)+1)*.005
        references=[env.reference(float(t)) for t in times]
        q=np.array([r[0] for r in references]);v=np.array([r[1] for r in references])
        original=np.array([base.reference(float(t))[0] for t in times])
        dense=dict(time=times,qpos=q,active=np.array([r[2] for r in references]),phase=np.array([r[3] for r in references]),errors=env.plan['errors'])
        result=audit_plan(env.env,dense,dt=.005)
        nodes=np.flatnonzero(env.plan['phase']=='lf_unload');nodes=np.r_[nodes[0]-1,nodes]
        node_error=max(float(abs(env.reference(float(env.plan['time'][i]))[0]-env.plan['qpos'][i]).max()) for i in nodes)
        derivative_error=0.
        for t in np.linspace(env.smooth_window[0]+.01,env.smooth_window[1]-.01,101):
            finite=(env.reference(t+1e-6)[0]-env.reference(t-1e-6)[0])/2e-6
            velocity=env.reference(t)[1]
            derivative_error=max(derivative_error,float(abs(finite[env.env.qadr]-velocity[env.env.vadr]).max()))
        outside=(times<env.smooth_window[0])|(times>env.smooth_window[1])
        exact_outside=bool(np.array_equal(q[outside],original[outside]))
        result.update(reference_sha256=sha(plan),source_sha256=sha(__file__),environment_source_sha256=sha('training/smooth_unload_reference_env.py'),
            knot_max_error=node_error,velocity_finite_difference_max_error=derivative_error,outside_window_exact=exact_outside,
            maximum_position_change_m=float(abs(q[:,:3]-original[:,:3]).max()),maximum_joint_change_rad=float(abs(q[:,env.env.qadr]-original[:,env.env.qadr]).max()),
            analytic_peak_joint_speed_rad_s=float(abs(v[:,env.env.vadr]).max()),
            scope='Same reference knots, phase schedule, horizon and endpoint. Cubic interpolation changes inter-knot path only during lf_unload. Dense5ms nominal collision check; no perturbed contact/dynamic success claim.')
        result['passed']=bool(result['passed'] and node_error<1e-12 and derivative_error<1e-7 and exact_outside)
        (out/'audit.json').write_text(json.dumps(result,indent=2));(out/'checker_source.py').write_bytes(Path(__file__).read_bytes())
        print(json.dumps({k:v for k,v in result.items() if k!='failures'},indent=2))
        if not result['passed']:raise SystemExit(1)
    finally:env.close();base.close()


if __name__=='__main__':main()
