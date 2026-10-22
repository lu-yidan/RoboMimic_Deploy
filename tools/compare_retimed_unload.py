"""Compare timed paths without pretending their observation clocks are equal."""
import json
from pathlib import Path
import numpy as np
from tools.continue_vector_full_cycle_training import sha


def main():
    p=Path('outputs/retimed_unload_calibration_v1');base=Path('outputs/highrod_height_mismatch_v1')
    protocol=json.loads((p/'protocol.json').read_text());audit=json.loads((p/'independent_comparison.json').read_text())
    assert len(audit['results'])==len(protocol['jobs'])==10
    assert {r['label'] for r in audit['results']}=={j[1] for j in protocol['jobs']}
    rows=[]
    for r in audit['results']:
        c=p/r['label'];m=json.loads((c/'manifest.json').read_text());height=m['offsets']['rung_01'][2]*1000
        b=base/f'true_height_{height:g}mm';bm=json.loads((b/'manifest.json').read_text())
        ca=json.loads((c/'independent_audit.json').read_text());ba=json.loads((b/'independent_audit.json').read_text())
        assert r['audit_sha256']==sha(c/'independent_audit.json')
        for directory,a in ((c,ca),(b,ba)):
            assert a['integrity_passed'] and a['trajectory_sha256']==sha(directory/'trajectory.npz') and a['summary_sha256']==sha(directory/'summary.json')
        for key in ('bundle_manifest_sha256','offsets','normal_bias_y_deg','memory_enabled','contract'):
            assert m[key]==bm[key],key
        factor=m['controller_modifier']['factor']
        assert m['controller_modifier']['type']=='reference-clock-unload-v1'
        with np.load(m['reference']) as x,np.load(bm['reference']) as y:
            assert set(x.files)==set(y.files)
            for key in x.files:
                if key not in ('time','cycle_ends_s'):assert np.array_equal(x[key],y[key]),key
            old=y['time'];expected=np.where(old<=52.2,old,np.where(old<=70.2,52.2+factor*(old-52.2),old+18*(factor-1)))
            np.testing.assert_allclose(x['time'],expected,rtol=0,atol=1e-12)
        with np.load(c/'trajectory.npz') as x,np.load(b/'trajectory.npz') as y:
            n=int(np.searchsorted(x['time_s'],52.2,side='left'))
            assert n==int(np.searchsorted(y['time_s'],52.2,side='left'))
            exact={k:bool(np.array_equal(x[k][:n],y[k][:n])) for k in ('qpos','qvel','time_s','actions','torques')}
            # Rewards belong to transition END times, unlike action observations.
            nr=int(np.searchsorted(x['time_s'][1:],52.2,side='left'))
            exact['rewards_before_window']=bool(np.array_equal(x['rewards'][:nr],y['rewards'][:nr]))
            boundary_reward_difference=float(x['rewards'][nr]-y['rewards'][nr])
            exact['nonclock_observations']=bool(np.array_equal(x['observations'][:n,:103],y['observations'][:n,:103]) and np.array_equal(x['observations'][:n,106:],y['observations'][:n,106:]))
            assert all(exact.values()),exact
        before,after=ba['independently_accepted'],ca['independently_accepted']
        rows.append(dict(label=r['label'],height_mm=height,factor=factor,baseline_passed=before,candidate_passed=after,
            fixed=not before and after,regressed=before and not after,time_s=r['time_s'],reason=r['reason'],prefix_exact=exact,
            reward_prefix_samples=nr,boundary_reward_difference=boundary_reward_difference,audit_sha256=sha(c/'independent_audit.json')))
    groups=[]
    for factor in (1.5,2.):
        items=[r for r in rows if r['factor']==factor]
        assert {r['height_mm'] for r in items}=={5.,7.5,10.,12.5,15.}
        fixed=[r['height_mm'] for r in items if r['fixed']];regressed=[r['height_mm'] for r in items if r['regressed']]
        groups.append(dict(factor=factor,passed=sum(r['candidate_passed'] for r in items),fixed=fixed,regressed=regressed,eligible=bool(fixed and not regressed)))
    eligible=[g['factor'] for g in groups if g['eligible']]
    result=dict(rows=rows,groups=groups,selected_factor=min(eligible) if eligible else None,protocol_sha256=sha(p/'protocol.json'),
        source_sha256=sha(__file__),scope='Known-case slower-reference calibration; same geometric path/endpoint. Horizon, clock observations and bias exposure duration change. Memory timeout remains physical time. Not policy improvement at identical speed.')
    (p/'selection.json').write_text(json.dumps(result,indent=2));(p/'selection_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(groups=groups,selected_factor=result['selected_factor']),indent=2))


if __name__=='__main__':main()
