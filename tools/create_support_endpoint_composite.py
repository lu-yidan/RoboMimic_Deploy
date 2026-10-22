"""Combine disjoint support and pre-existing endpoint reference corrections."""
import json
from pathlib import Path
import numpy as np
from training.wall_support_env import WallSupportEnv
from tools.plan_clearance_cycle import audit_plan
from tools.continue_vector_full_cycle_training import sha


def main():
    support=Path('outputs/support_retarget_plan_v1');endpoint=Path('outputs/highrod_gain_sensitivity_v1/plan_gain_0.75')
    parent=Path('outputs/highrod_endpoint_adapt_plan_v1/plan.npz');out=Path('outputs/support_endpoint075_plan_v1');out.mkdir(exist_ok=False)
    for folder in (support,endpoint):
        m=json.loads((folder/'manifest.json').read_text());assert m['plan_sha256']==sha(folder/'plan.npz') and json.loads((folder/'audit.json').read_text())['passed']
    with np.load(support/'plan.npz') as z:plan={k:z[k].copy() for k in z.files}
    with np.load(endpoint/'plan.npz') as x,np.load(parent) as y:
        np.testing.assert_array_equal(x['time'],y['time']);delta=x['qpos']-y['qpos']
        assert np.all(delta[x['time']<=99.66+1e-8]==0.)
        before=plan['qpos'].copy()
        plan['qpos']+=np.column_stack([np.interp(plan['time'],x['time'],delta[:,j]) for j in range(delta.shape[1])])
        np.testing.assert_array_equal(plan['qpos'][plan['time']<=99.66+1e-8],before[plan['time']<=99.66+1e-8])
        np.testing.assert_allclose(plan['qpos'][-1],x['qpos'][-1],rtol=0,atol=1e-15)
    np.savez_compressed(out/'plan.npz',**plan)
    e=WallSupportEnv('outputs/wall_support_bundle_pd15_v2')
    try:audit=audit_plan(e,plan,dt=.005)
    finally:e.close()
    audit['parent_ik_residual']=audit.pop('ik_residual');audit['scope']='Dense nominal collision/native check; inherited parent IK error. Composite endpoint differs from full gain1 target.'
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    manifest=dict(plan_sha256=sha(out/'plan.npz'),support_parent=str(support/'plan.npz'),support_parent_sha256=sha(support/'plan.npz'),
        endpoint_parent=str(endpoint/'plan.npz'),endpoint_parent_sha256=sha(endpoint/'plan.npz'),source_sha256=sha(__file__),
        scope='Support retarget57..65s combined with pre-existing endpoint gain0.75 after99.66s. Reference identical to support-only before99.66s. Final endpoint target changes, but all native/physical/closure thresholds remain unchanged.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'builder_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(passed=audit['passed'],failures=len(audit['failures']))))
    if not audit['passed']:raise SystemExit(1)


if __name__=='__main__':main()
