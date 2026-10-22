"""Independently check a failed force QP using a linear feasibility solver."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import osqp
from scipy.optimize import linprog
from training.wall_support_env import WallSupportEnv
from training.climb_cycle import CycleTorqueController


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    state=np.load(a.run/'failure_state.npz');manifest=json.loads((a.run/'manifest.json').read_text());cfg=manifest['config']
    e=WallSupportEnv(manifest['bundle']);e.reset();e.data.qpos[:]=state['qpos'];e.data.qvel[:]=state['qvel']
    captured={};original=osqp.OSQP
    def factory():
        solver=original();setup=solver.setup
        def record(**kwargs):
            captured.update(kwargs);return setup(**kwargs)
        solver.setup=record;return solver
    with patch('training.climb_cycle.osqp.OSQP',side_effect=factory):
        c=CycleTorqueController(e,torque_weight=cfg['torque_weight'])
        try:c.command(state['qref'],state['vref'],state['active']);qp='solved'
        except RuntimeError as exc:qp=str(exc)
    A=captured['A'].toarray();lo=captured['l'];hi=captured['u'];n=A.shape[1]
    eq=np.isfinite(lo)&np.isfinite(hi)&(abs(hi-lo)<1e-10)
    up=np.flatnonzero(~eq&np.isfinite(hi));low=np.flatnonzero(~eq&np.isfinite(lo))
    U=np.vstack([A[up],-A[low]]);b=np.r_[hi[up],-lo[low]]
    lp=linprog(np.zeros(n),A_ub=U,b_ub=b,A_eq=A[eq],b_eq=hi[eq],bounds=[(None,None)]*n,method='highs')
    result=dict(time_s=float(state['time_s']),phase=str(state['phase']),controller_status=qp,
                solver_fallbacks=c.solver_fallbacks,linear_feasibility_status=int(lp.status),linear_feasibility_message=lp.message)
    if lp.success:
        result['maximum_constraint_violation']=float(max(0,np.max(lo-A@lp.x),np.max(A@lp.x-hi)))
    else:
        # Diagnostic only: quantify the smallest uniform motor-bound relaxation.
        motor=np.r_[(up>=6)&(up<35),(low>=6)&(low<35)].astype(float)
        relaxed=linprog(np.r_[np.zeros(n),1.],A_ub=np.c_[U,-motor],b_ub=b,
            A_eq=np.c_[A[eq],np.zeros(eq.sum())],b_eq=hi[eq],bounds=[(None,None)]*n+[(0,None)],method='highs')
        result['minimum_uniform_motor_relaxation_nm']=float(relaxed.x[-1]) if relaxed.success else None
        result['relaxation_scope']='Offline diagnosis only; never sent to simulation or used to pass acceptance'
    np.savez_compressed(a.run/'failed_force_qp.npz',A=A,lower=lo,upper=hi,P=captured['P'].toarray(),objective=captured['q'])
    (a.run/'allocation_diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));e.close()


if __name__=='__main__':main()
