"""Blend independently verified contact-location terminal poses into the support plan."""
import json
from pathlib import Path
import numpy as np
from training.wall_support_env import WallSupportEnv
from tools.plan_clearance_cycle import audit_plan
from tools.continue_vector_full_cycle_training import sha


def main():
    parent=Path('outputs/support_retarget_plan_v1/plan.npz');probe=Path('outputs/terminal_contact_minimax_v1')
    audit=json.loads((probe/'independent_audit.json').read_text());assert audit['passed'] and audit['summary_sha256']==sha(probe/'summary.json')
    with np.load(parent) as z:original={k:z[k].copy() for k in z.files}
    start=float(original['time'][np.flatnonzero(original['phase']=='posture_closure')[0]-1]);end=float(original['time'][np.flatnonzero(original['phase']=='closed_hold')[0]-1])
    e=WallSupportEnv('outputs/wall_support_bundle_pd15_v2')
    try:
        for row in audit['rows']:
            h=row['height_mm'];file=probe/f'height{h:g}.npz';assert row['candidate_sha256']==sha(file)
            with np.load(file) as z:target=z['qpos']
            out=Path(f'outputs/support_contact{h:g}_plan_v1');out.mkdir(exist_ok=False)
            plan={k:v.copy() for k,v in original.items()};u=np.clip((plan['time']-start)/(end-start),0,1);w=u*u*(3-2*u)
            plan['qpos']+=w[:,None]*(target-original['qpos'][-1]);plan['qpos'][plan['time']>=end]=target
            np.testing.assert_array_equal(plan['qpos'][plan['time']<=start],original['qpos'][plan['time']<=start])
            np.savez_compressed(out/'plan.npz',**plan);check=audit_plan(e,plan,dt=.005)
            check['parent_ik_residual']=check.pop('ik_residual');check['scope']='Nominal5ms collision/native audit; terminal tangency has separate independent pose validation.'
            (out/'audit.json').write_text(json.dumps(check,indent=2))
            manifest=dict(parent=str(parent),parent_sha256=sha(parent),candidate=str(file),candidate_sha256=sha(file),
                candidate_audit_sha256=sha(probe/'independent_audit.json'),estimated_height_mm=h,blend_window_s=[start,end],
                plan_sha256=sha(out/'plan.npz'),source_sha256=sha(__file__),scope='Same support retarget and duration. New terminal target permits fore-aft sole contact relocation within the existing foot, with bounded root height and joint closure. Dynamic acceptance unchanged.')
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'builder_source.py').write_bytes(Path(__file__).read_bytes())
            print(json.dumps(dict(height_mm=h,passed=check['passed'],failures=len(check['failures']))),flush=True)
            if not check['passed']:raise SystemExit(1)
    finally:e.close()


if __name__=='__main__':main()
