"""Local affine response fit; prediction is not a feasibility certificate."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from tools.continue_vector_full_cycle_training import sha


def main():
    source=Path('outputs/closure_action_authority_20261024_v1')
    summary=json.loads((source/'summary.json').read_text())
    protocol=json.loads((source/'protocol.json').read_text())
    assert summary['completed'] and summary['protocol_sha256']==sha(source/'protocol.json')
    rows=summary['rows'];assert len(rows)==13
    for r in rows:
        assert sha(source/f"trial_{r['trial']:02d}.npz")==r['trajectory_sha256']
        assert r['steps']==700
    def vector(r):return np.r_[r['mean_joint_errors'],r['mean_rise_m']-.25]
    y=vector(rows[0]);step=protocol['step']
    jac=np.column_stack([(vector(rows[2+2*j])-vector(rows[1+2*j]))/(2*step) for j in range(6)])
    scales=np.r_[np.full(29,.005),.002]
    limits=np.r_[np.full(29,.038),.008]
    def cost(p):
        excess=np.maximum(0.,np.abs(y+jac@p)-limits)/scales
        return float(excess@excess+1e-4*(p@p))
    fit=minimize(cost,np.zeros(6),method='SLSQP',bounds=[(-1.,1.)]*6,options={'ftol':1e-12,'maxiter':300})
    prediction=y+jac@fit.x
    result=dict(optimizer_success=bool(fit.success),message=fit.message,
        parameters=fit.x.tolist(),groups=protocol['groups'],basis=protocol['basis'],
        zero_cost=cost(np.zeros(6)),predicted_cost=cost(fit.x),
        predicted_mean_closure_rad=float(abs(prediction[:29]).max()),
        predicted_mean_rise_m=float(prediction[-1]+.25),response_jacobian=jac.tolist(),
        source_summary_sha256=sha(source/'summary.json'),source_sha256=sha(__file__),
        scope='Local central-difference model of mean last-2s response. Optimization on known failed case; predictions do not verify maximum-window gates or full-cycle success. Requires actual physics rollout with original limits.')
    out=Path('outputs/closure_action_response_fit_20261024_v1');out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    (out/'fit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k not in ('response_jacobian','basis')},indent=2))


if __name__=='__main__':main()
