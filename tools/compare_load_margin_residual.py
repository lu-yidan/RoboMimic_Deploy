"""Select the fixed candidate only if it repairs a failure without regressions."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.continue_vector_full_cycle_training import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('experiment',type=Path);args=parser.parse_args();p=args.experiment
    protocol=json.loads((p/'protocol.json').read_text());comparison=json.loads((p/'independent_comparison.json').read_text())
    expected=set(protocol['cases'])
    if {r['label'] for r in comparison['results']}!=expected or len(comparison['results'])!=len(expected):raise ValueError('Incomplete candidate audit')
    baseline=Path('outputs/highrod_height_mismatch_v1');rows=[]
    for r in comparison['results']:
        case=r['label'];candidate=p/case;m=json.loads((candidate/'manifest.json').read_text());s=json.loads((candidate/'summary.json').read_text());a=json.loads((candidate/'independent_audit.json').read_text())
        height=m['offsets']['rung_01'][2]*1000;old=baseline/f'true_height_{height:g}mm'
        bm=json.loads((old/'manifest.json').read_text());bs=json.loads((old/'summary.json').read_text());ba=json.loads((old/'independent_audit.json').read_text())
        if not a['integrity_passed'] or not ba['integrity_passed']:raise ValueError('Integrity failed')
        if sha(candidate/'independent_audit.json')!=r['audit_sha256']:raise ValueError('Candidate audit changed')
        for directory,summary,audit in [(candidate,s,a),(old,bs,ba)]:
            if sha(directory/'summary.json')!=audit['summary_sha256'] or sha(directory/'trajectory.npz')!=audit['trajectory_sha256']:raise ValueError('Audited evidence changed')
        for key in ('offsets','reference_sha256','bundle_manifest_sha256','normal_bias_y_deg','memory_enabled','contract','horizon_s'):
            if m[key]!=bm[key]:raise ValueError('Different task: '+key)
        rule=m['action_rule']
        if rule!=dict(type='fixed_vector_cubic_envelope',vector=protocol['normalized_action_vector'],start_s=57.,end_s=65.,ramp_s=2.):raise ValueError('Candidate changed')
        with np.load(old/'trajectory.npz') as x,np.load(candidate/'trajectory.npz') as y:
            tx=x['time_s'];ty=y['time_s'];nx=int(np.searchsorted(tx,57.,side='left'));ny=int(np.searchsorted(ty,57.,side='left'))
            if nx!=ny:raise ValueError('Prefix duration differs')
            exact={key:bool(np.array_equal(x[key][:nx],y[key][:ny])) for key in ('qpos','qvel','time_s','actions','observations','torques','rewards')}
        if not all(exact.values()):raise ValueError('Physical prefix differs before intervention')
        before,after=ba['independently_accepted'],a['independently_accepted']
        rows.append(dict(case=case,zero_passed=before,candidate_passed=after,fixed=not before and after,regressed=before and not after,
            time_s=s['time_s'],reason=s['reason'],endpoints=a['endpoints'],prefix_exact=exact,prefix_samples=nx,
            zero_audit_sha256=sha(old/'independent_audit.json'),candidate_audit_sha256=sha(candidate/'independent_audit.json')))
    fixed=[r['case'] for r in rows if r['fixed']];regressed=[r['case'] for r in rows if r['regressed']]
    result=dict(cases=rows,zero_passed=sum(r['zero_passed'] for r in rows),candidate_passed=sum(r['candidate_passed'] for r in rows),
        fixed_cases=fixed,regressed_cases=regressed,selected=bool(fixed and not regressed),protocol_sha256=sha(p/'protocol.json'),
        source_sha256=sha(__file__),scope='Known-case fixed candidate, unchanged reference/controller and motor-only intervention. Select only if at least one failure repaired and no baseline success lost. Not a learned policy or unseen generalization result.')
    (p/'selection.json').write_text(json.dumps(result,indent=2));(p/'selection_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result,indent=2))


if __name__=='__main__':main()
