"""Paired reference calibration with explicit endpoint changes and common metrics."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha


def common_geometry_error(directory,manifest):
    env=FullCycleResidualEnv(plan=manifest['reference']);e=env.env;d=mujoco.MjData(e.model)
    try:
        with np.load('outputs/early_swing14_lift26_plan_v1/plan.npz') as z:d.qpos[:]=z['qpos'][-1]
        mujoco.mj_kinematics(e.model,d);bodies=env.endpoint_bodies
        desired=d.xpos[bodies].copy();desired[2:,2]+=manifest['offsets']['rung_01'][2]
        with np.load(directory/'trajectory.npz') as z:
            ids=np.flatnonzero((z['time_s']>env.horizon-2)&(z['time_s']<=env.horizon+1e-7));q=z['qpos']
            if len(ids)<99:return None
            error=np.zeros(4)
            for i in ids:
                d.qpos[:]=q[i];mujoco.mj_kinematics(e.model,d);error=np.maximum(error,np.linalg.norm(d.xpos[bodies]-desired,axis=1))
        return error.tolist()
    finally:env.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--start',type=float,required=True);parser.add_argument('--changed-endpoint',action='store_true');args=parser.parse_args()
    p,base=args.candidate,args.baseline;protocol=json.loads((p/'protocol.json').read_text());audit=json.loads((p/'independent_comparison.json').read_text())
    assert len(audit['results'])==5 and {r['label'] for r in audit['results']}=={j[1] for j in protocol['jobs']}
    rows=[]
    for r in audit['results']:
        c=p/r['label'];b=base/r['label'];m=json.loads((c/'manifest.json').read_text());bm=json.loads((b/'manifest.json').read_text())
        ca=json.loads((c/'independent_audit.json').read_text());ba=json.loads((b/'independent_audit.json').read_text())
        assert r['audit_sha256']==sha(c/'independent_audit.json')
        for directory,a in ((c,ca),(b,ba)):
            assert a['integrity_passed'] and a['trajectory_sha256']==sha(directory/'trajectory.npz') and a['summary_sha256']==sha(directory/'summary.json')
        for key in ('bundle_manifest_sha256','offsets','normal_bias_y_deg','memory_enabled','contract','horizon_s'):
            assert m[key]==bm[key],key
        assert not any(k in m or k in bm for k in ('reference_modifier','action_rule'))
        assert not bm.get('controller_modifier') and m['controller_modifier']['type']=='oracle-acceleration-qp-window-v1'
        with np.load(m['reference']) as x,np.load(bm['reference']) as y:
            np.testing.assert_array_equal(x['qpos'][0],y['qpos'][0]);endpoint_delta=float(abs(x['qpos'][-1]-y['qpos'][-1]).max())
            if not args.changed_endpoint:assert endpoint_delta==0.
        with np.load(c/'trajectory.npz') as x,np.load(b/'trajectory.npz') as y:
            n=int(np.searchsorted(x['time_s'],args.start,side='left'));assert n==int(np.searchsorted(y['time_s'],args.start,side='left'))
            exact={k:bool(np.array_equal(x[k][:n],y[k][:n])) for k in ('qpos','qvel','time_s','actions','observations')}
            nr=int(np.searchsorted(x['time_s'][1:],args.start,side='left'))
            exact['torques_before_window']=bool(np.array_equal(x['torques'][:nr],y['torques'][:nr]))
            exact['rewards_before_window']=bool(np.array_equal(x['rewards'][:nr],y['rewards'][:nr]))
            assert all(exact.values()),exact
        before,after=ba['independently_accepted'],ca['independently_accepted']
        rows.append(dict(case=m['case'],baseline_passed=before,candidate_passed=after,fixed=not before and after,regressed=before and not after,
            time_s=r['time_s'],reason=r['reason'],prefix_exact=exact,endpoint_reference_max_delta=endpoint_delta,
            common_actual_geometry_error_m=dict(baseline=common_geometry_error(b,bm),candidate=common_geometry_error(c,m)),
            baseline_audit_sha256=sha(b/'independent_audit.json'),candidate_audit_sha256=sha(c/'independent_audit.json')))
    fixed=[r['case'] for r in rows if r['fixed']];regressed=[r['case'] for r in rows if r['regressed']]
    result=dict(rows=rows,baseline_passed=sum(r['baseline_passed'] for r in rows),candidate_passed=sum(r['candidate_passed'] for r in rows),
        fixed=fixed,regressed=regressed,selected=bool(fixed and not regressed),changed_endpoint=args.changed_endpoint,
        protocol_sha256=sha(p/'protocol.json'),source_sha256=sha(__file__),
        scope='Known-case local oracle controller calibration; actual contact geometry replaces the biased frame for the added QP. Full physical/closure thresholds unchanged; endpoint target changes explicitly labelled. Common actual-geometry errors are supplementary and do not relabel acceptance. No held-out or training claim.')
    (p/'selection.json').write_text(json.dumps(result,indent=2));(p/'selection_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:result[k] for k in ('baseline_passed','candidate_passed','fixed','regressed','selected','changed_endpoint')},indent=2))


if __name__=='__main__':main()
