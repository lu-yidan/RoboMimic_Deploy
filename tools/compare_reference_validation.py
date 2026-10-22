"""Paired comparison on a fixed validation grid, preserving execution order caveat."""
import json
from pathlib import Path
import numpy as np
from tools.continue_vector_full_cycle_training import sha


def main():
    old=Path('outputs/prior_reference_validation_v1');new=Path('outputs/support_reference_validation_v1');output=Path('outputs/reference_validation_comparison_v1');output.mkdir(exist_ok=False)
    previous=json.loads((old/'validation_summary.json').read_text());current=json.loads((new/'validation_summary.json').read_text());protocol=json.loads((old/'protocol.json').read_text())
    assert protocol['paired_grid_sha256']==sha(new/'protocol.json')
    assert {r['case'] for r in previous['rows']}=={r['case'] for r in current['rows']} and len(previous['rows'])==len(current['rows'])==12
    rows=[]
    for r in current['rows']:
        before=next(v for v in previous['rows'] if v['case']==r['case']);assert before['offset_mm']==r['offset_mm']
        paths=[old/r['case'],new/r['case']];manifests=[json.loads((p/'manifest.json').read_text()) for p in paths]
        for p,row in zip(paths,[before,r]):
            assert sha(p/'independent_audit.json')==row['audit_sha256'];audit=json.loads((p/'independent_audit.json').read_text())
            assert audit['integrity_passed'] and audit['trajectory_sha256']==sha(p/'trajectory.npz')
        for key in ('contract','offsets','bundle_manifest_sha256','normal_bias_y_deg','memory_enabled','horizon_s'):assert manifests[0][key]==manifests[1][key]
        with np.load(paths[0]/'trajectory.npz') as a,np.load(paths[1]/'trajectory.npz') as b:
            n=int(np.searchsorted(a['time_s'],57.));nr=int(np.searchsorted(a['time_s'][1:],57.))
            prefix={key:bool(np.array_equal(a[key][:n],b[key][:n])) for key in ('qpos','qvel','time_s','actions','observations')}
            prefix.update({key:bool(np.array_equal(a[key][:nr],b[key][:nr])) for key in ('torques','rewards')});assert all(prefix.values())
        rows.append(dict(case=r['case'],offset_mm=r['offset_mm'],prior_passed=before['passed'],support_passed=r['passed'],fixed=not before['passed'] and r['passed'],regressed=before['passed'] and not r['passed'],prior_reason=before['reason'],support_reason=r['reason'],prefix_exact=prefix))
    result=dict(rows=rows,prior_passed=sum(r['prior_passed'] for r in rows),support_passed=sum(r['support_passed'] for r in rows),fixed=[r['case'] for r in rows if r['fixed']],regressed=[r['case'] for r in rows if r['regressed']],summary_hashes={str(p):sha(p/'validation_summary.json') for p in (old,new)},source_sha256=sha(__file__),scope='Paired deterministic interpolation/offset grid. Candidate was frozen before validation; old-reference comparator ran after candidate results. Not a prospectively registered two-arm test or physical generalization claim.')
    (output/'comparison.json').write_text(json.dumps(result,indent=2));(output/'comparison_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({k:result[k] for k in ('prior_passed','support_passed','fixed','regressed')},indent=2))


if __name__=='__main__':main()
