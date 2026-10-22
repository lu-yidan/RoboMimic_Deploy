"""Audit complete calibration coverage and preserve every rejected gain."""
import json
from pathlib import Path
import numpy as np
from tools.continue_vector_full_cycle_training import sha


def main():
    p=Path('outputs/temporal_leg_gain_calibration_v1');base=Path('outputs/highrod_height_mismatch_v1')
    protocol=json.loads((p/'protocol.json').read_text());audit=json.loads((p/'independent_comparison.json').read_text())
    assert {r['label'] for r in audit['results']}=={j[1] for j in protocol['jobs']}
    assert len(audit['results'])==len(protocol['jobs'])==10
    rows=[]
    for r in audit['results']:
        c=p/r['label'];m=json.loads((c/'manifest.json').read_text());height=m['offsets']['rung_01'][2]*1000
        b=base/f'true_height_{height:g}mm';bm=json.loads((b/'manifest.json').read_text())
        ca=json.loads((c/'independent_audit.json').read_text());ba=json.loads((b/'independent_audit.json').read_text())
        assert r['audit_sha256']==sha(c/'independent_audit.json')
        for directory,a in ((c,ca),(b,ba)):
            assert a['integrity_passed'] and a['trajectory_sha256']==sha(directory/'trajectory.npz') and a['summary_sha256']==sha(directory/'summary.json')
        for key in ('reference_sha256','bundle_manifest_sha256','offsets','normal_bias_y_deg','memory_enabled','contract','horizon_s'):
            assert m[key]==bm[key],key
        assert 'controller_modifier' not in bm and 'reference_modifier' not in m and 'action_rule' not in m
        with np.load(c/'trajectory.npz') as x,np.load(b/'trajectory.npz') as y:
            n=int(np.searchsorted(x['time_s'],57.,side='left'))
            assert n==int(np.searchsorted(y['time_s'],57.,side='left'))
            exact={k:bool(np.array_equal(x[k][:n],y[k][:n])) for k in ('qpos','qvel','time_s','actions','observations','torques','rewards')}
            assert all(exact.values())
        before,after=ba['independently_accepted'],ca['independently_accepted']
        rows.append(dict(label=r['label'],height_mm=height,peak_leg_kp=m['controller_modifier']['peak_leg_kp'],
            baseline_passed=before,candidate_passed=after,fixed=not before and after,regressed=before and not after,
            time_s=r['time_s'],reason=r['reason'],prefix_exact=exact,audit_sha256=sha(c/'independent_audit.json')))
    groups=[]
    for gain in (200.,300.):
        items=[r for r in rows if r['peak_leg_kp']==gain]
        assert {r['height_mm'] for r in items}=={5.,7.5,10.,12.5,15.}
        fixed=[r['height_mm'] for r in items if r['fixed']];regressed=[r['height_mm'] for r in items if r['regressed']]
        groups.append(dict(gain=gain,passed=sum(r['candidate_passed'] for r in items),fixed=fixed,regressed=regressed,eligible=bool(fixed and not regressed)))
    eligible=[g['gain'] for g in groups if g['eligible']]
    result=dict(rows=rows,groups=groups,selected_gain=min(eligible) if eligible else None,
        protocol_sha256=sha(p/'protocol.json'),audit_sha256=sha(p/'independent_comparison.json'),source_sha256=sha(__file__),
        scope='Known-case controller calibration with same reference and zero actions. No claim about learning or held-out robustness.')
    (p/'selection.json').write_text(json.dumps(result,indent=2));(p/'selection_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(groups=groups,selected_gain=result['selected_gain']),indent=2))


if __name__=='__main__':main()
