"""Independently verify archived solver-status disputes and native torque bounds."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import linprog
from training.wall_support_env import WallSupportEnv


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);a=p.parse_args()
    e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');results=[]
    for trial in sorted(a.suite.iterdir()):
        events_path=trial/'solver_status_events.json'
        if not events_path.exists():continue
        manifest=json.loads((trial/'manifest.json').read_text())
        sources_match=all(hashlib.sha256((trial/'source_snapshot'/rel).read_bytes()).hexdigest()==sha for rel,sha in manifest['sources'].items())
        for event in json.loads(events_path.read_text()):
            path=trial/'solver_problems'/(str(event['problem'])+'.npz');z=dict(np.load(path))
            A=z['A'];lo=z['lower'];hi=z['upper'];n=A.shape[1]
            eq=np.isfinite(lo)&np.isfinite(hi)&(abs(hi-lo)<1e-10);low=np.isfinite(lo)&~eq;up=np.isfinite(hi)&~eq
            seed=linprog(np.zeros(n),A_ub=np.vstack([A[up],-A[low]]),b_ub=np.r_[hi[up],-lo[low]],
                A_eq=A[eq],b_eq=hi[eq],bounds=[(None,None)]*n,method='highs')
            original_bounds=np.allclose(hi[6:35]-lo[6:35],e.limits[:,1]-e.limits[:,0],rtol=0,atol=1e-12)
            classification=bool(seed.success)==bool(event['verified'])
            check=dict(trial=trial.name,event=event,independent_lp_feasible=bool(seed.success),
                native_bounds_match=bool(original_bounds),sources_match=sources_match,
                problem_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if event['verified']:
                force=z['solution'];violation=float(max(0.,np.max(lo-A@force),np.max(A@force-hi)))
                required=lo[6:35]+e.limits[:,1];torque=required-A[6:35]@force
                torque_violation=float(max(0.,np.max(e.limits[:,0]-torque),np.max(torque-e.limits[:,1])))
                check.update(maximum_constraint_violation=violation,maximum_native_torque_violation=torque_violation,
                    torque_peak_ratio=float(max(abs(torque)/e.limits[:,1])))
                classification &= np.isfinite(force).all() and violation<=1e-5 and torque_violation<=1e-5
            else:classification &= 'solution' not in z and seed.status==2
            check['passed']=bool(classification and original_bounds and sources_match);results.append(check)
    report=dict(passed=bool(results and all(r['passed'] for r in results)),events=len(results),
        verified_feasible=sum(r['independent_lp_feasible'] for r in results),results=results,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Archived optimization feasibility and native torque-bound audit; does not prove task success or contact persistence.')
    (a.suite/'solver_verification_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));e.close()
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
