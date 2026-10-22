"""Preserve every reference pose; change only lf_unload knot timing."""
import json
from pathlib import Path
import numpy as np
from training.wall_support_env import WallSupportEnv
from tools.plan_clearance_cycle import audit_plan
from tools.continue_vector_full_cycle_training import sha


def main():
    parent=Path('outputs/highrod_endpoint_adapt_plan_v1/plan.npz')
    with np.load(parent) as z:original={k:z[k].copy() for k in z.files}
    indices=np.flatnonzero(original['phase']=='lf_unload')
    start=float(original['time'][indices[0]-1]);end=float(original['time'][indices[-1]])
    assert abs(start-52.2)<1e-8 and abs(end-70.2)<1e-8
    e=WallSupportEnv('outputs/wall_support_bundle_pd15_v2')
    try:
        for factor in (1.5,2.):
            out=Path(f'outputs/retimed_unload_factor{factor:g}_plan_v1');out.mkdir(exist_ok=False)
            plan={k:v.copy() for k,v in original.items()};t=plan['time']
            plan['time']=np.where(t<=start,t,np.where(t<=end,start+factor*(t-start),t+(factor-1)*(end-start)))
            if 'cycle_ends_s' in plan:plan['cycle_ends_s']=plan['cycle_ends_s']+(factor-1)*(end-start)
            for key in original:
                if key not in ('time','cycle_ends_s'):assert np.array_equal(plan[key],original[key])
            np.savez_compressed(out/'plan.npz',**plan)
            audit=audit_plan(e,plan)
            audit['parent_ik_residual']=audit.pop('ik_residual')
            audit['scope']='Dense nominal collision/native-position check after retiming. Same pose knots; parent IK residual inherited. No dynamic or perturbed-scene certificate.'
            (out/'audit.json').write_text(json.dumps(audit,indent=2))
            metadata=dict(parent=str(parent),parent_sha256=sha(parent),plan_sha256=sha(out/'plan.npz'),
                unload_time_factor=factor,original_window_s=[start,end],new_window_s=[start,start+factor*(end-start)],
                source_sha256=sha(__file__),scope='Same geometric path and endpoint; longer lf_unload only. Evaluation must map normal-bias window using original reference time.')
            (out/'manifest.json').write_text(json.dumps(metadata,indent=2));(out/'builder_source.py').write_bytes(Path(__file__).read_bytes())
            print(json.dumps(dict(factor=factor,passed=audit['passed'],horizon_s=float(plan['time'][-1]),failures=len(audit['failures']))))
            if not audit['passed']:raise RuntimeError('Retimed plan audit failed')
    finally:e.close()


if __name__=='__main__':main()
