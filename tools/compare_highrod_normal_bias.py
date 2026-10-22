"""Keep reference/height fixed while contrasting declared normal biases."""
import json
from pathlib import Path
import numpy as np
from tools.continue_vector_full_cycle_training import sha


def main():
    p=Path('outputs/highrod_unbiased_height_v1');base=Path('outputs/highrod_height_mismatch_v1')
    protocol=json.loads((p/'protocol.json').read_text());audit=json.loads((p/'independent_comparison.json').read_text())
    assert len(audit['results'])==5 and {r['label'] for r in audit['results']}=={j[1] for j in protocol['jobs']}
    rows=[]
    for r in audit['results']:
        c=p/r['label'];b=base/r['label']
        m=json.loads((c/'manifest.json').read_text());bm=json.loads((b/'manifest.json').read_text())
        ca=json.loads((c/'independent_audit.json').read_text());ba=json.loads((b/'independent_audit.json').read_text())
        assert r['audit_sha256']==sha(c/'independent_audit.json')
        for directory,a in ((c,ca),(b,ba)):
            assert a['integrity_passed'] and a['trajectory_sha256']==sha(directory/'trajectory.npz') and a['summary_sha256']==sha(directory/'summary.json')
        for key in ('reference_sha256','bundle_manifest_sha256','offsets','memory_enabled','contract','horizon_s'):
            assert m[key]==bm[key],key
        assert m['normal_bias_y_deg']==0. and bm['normal_bias_y_deg']==-3.
        assert not any(k in m or k in bm for k in ('controller_modifier','reference_modifier','action_rule'))
        with np.load(c/'trajectory.npz') as x,np.load(b/'trajectory.npz') as y:
            n=int(np.searchsorted(x['time_s'],49.,side='left'))
            assert n==int(np.searchsorted(y['time_s'],49.,side='left'))
            exact={k:bool(np.array_equal(x[k][:n],y[k][:n])) for k in ('qpos','qvel','time_s','actions','observations','torques','rewards')}
            assert all(exact.values())
        rows.append(dict(case=m['case'],biased_passed=ba['independently_accepted'],unbiased_passed=ca['independently_accepted'],
            time_s=r['time_s'],reason=r['reason'],prefix_exact=exact,prefix_samples=n,
            audit_sha256=sha(c/'independent_audit.json')))
    result=dict(rows=rows,biased_passed=sum(r['biased_passed'] for r in rows),unbiased_passed=sum(r['unbiased_passed'] for r in rows),
        protocol_sha256=sha(p/'protocol.json'),source_sha256=sha(__file__),
        scope='Known-case paired comparison, no learned sensing. Exact contact geometry remains privileged even with zero declared normal bias.')
    (p/'comparison.json').write_text(json.dumps(result,indent=2));(p/'comparison_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
