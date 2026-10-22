"""Compare interpolation candidates against the original audited linear reference."""
import json
from pathlib import Path
import numpy as np
from tools.continue_vector_full_cycle_training import sha


def main():
    p=Path('outputs/smooth_unload_calibration_v1');base=Path('outputs/highrod_height_mismatch_v1')
    protocol=json.loads((p/'protocol.json').read_text());audit=json.loads((p/'independent_comparison.json').read_text())
    assert len(audit['results'])==5 and {r['label'] for r in audit['results']}=={j[1] for j in protocol['jobs']}
    rows=[]
    for r in audit['results']:
        c=p/r['label'];b=base/r['label'];m=json.loads((c/'manifest.json').read_text());bm=json.loads((b/'manifest.json').read_text())
        ca=json.loads((c/'independent_audit.json').read_text());ba=json.loads((b/'independent_audit.json').read_text())
        assert r['audit_sha256']==sha(c/'independent_audit.json')
        for directory,a in ((c,ca),(b,ba)):
            assert a['integrity_passed'] and a['trajectory_sha256']==sha(directory/'trajectory.npz') and a['summary_sha256']==sha(directory/'summary.json')
        for key in ('reference_sha256','bundle_manifest_sha256','offsets','normal_bias_y_deg','memory_enabled','contract','horizon_s'):
            assert m[key]==bm[key],key
        assert m['reference_modifier']['type']=='smooth-unload-reference-v1' and 'reference_modifier' not in bm
        assert 'controller_modifier' not in m and 'action_rule' not in m
        with np.load(c/'trajectory.npz') as x,np.load(b/'trajectory.npz') as y:
            n=int(np.searchsorted(x['time_s'],52.2,side='left'));assert n==int(np.searchsorted(y['time_s'],52.2,side='left'))
            exact={k:bool(np.array_equal(x[k][:n],y[k][:n])) for k in ('qpos','qvel','time_s','actions','observations','torques')}
            # The boundary transition ends just after52.2s due to floating time.
            nr=int(np.searchsorted(x['time_s'][1:],52.2,side='left'))
            exact['rewards_before_window']=bool(np.array_equal(x['rewards'][:nr],y['rewards'][:nr]))
            boundary_reward_difference=float(x['rewards'][nr]-y['rewards'][nr])
            assert all(exact.values()),exact
        before,after=ba['independently_accepted'],ca['independently_accepted']
        rows.append(dict(case=m['case'],baseline_passed=before,candidate_passed=after,fixed=not before and after,regressed=before and not after,
            time_s=r['time_s'],reason=r['reason'],prefix_exact=exact,reward_prefix_samples=nr,boundary_reward_difference=boundary_reward_difference,audit_sha256=sha(c/'independent_audit.json')))
    fixed=[r['case'] for r in rows if r['fixed']];regressed=[r['case'] for r in rows if r['regressed']]
    result=dict(rows=rows,fixed=fixed,regressed=regressed,selected=bool(fixed and not regressed),
        protocol_sha256=sha(p/'protocol.json'),source_sha256=sha(__file__),scope='Known-case interpolation calibration. Same reference knots and duration, different inter-knot paths. No learned policy or held-out claim.')
    (p/'selection.json').write_text(json.dumps(result,indent=2));(p/'selection_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(passed=sum(r['candidate_passed'] for r in rows),fixed=fixed,regressed=regressed,selected=result['selected']),indent=2))


if __name__=='__main__':main()
