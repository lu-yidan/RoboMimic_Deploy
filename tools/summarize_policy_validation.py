"""Verified paired zero/actor outcomes for an existing geometry-policy evaluation."""
import argparse
import json
from pathlib import Path
from tools.continue_vector_full_cycle_training import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('experiment',type=Path);args=parser.parse_args();p=args.experiment
    m=json.loads((p/'manifest.json').read_text());s=json.loads((p/'summary.json').read_text());a=json.loads((p/'independent_training_audit.json').read_text());assert a['passed']
    for file,key in [('policy.zip','checkpoint_sha256'),('manifest.json','experiment_manifest_sha256'),('summary.json','experiment_summary_sha256')]:assert sha(p/file)==a[key]
    rows=[]
    for case,offsets in m['evaluation_cases'].items():
        members=[]
        for label in ('zero','after'):
            found=[r for r in a['trajectories'] if r['case']==case and r['label']==label];assert len(found)==1;r=found[0]
            assert r['passed'] and sha(p/f'{label}_{case}.npz')==r['trajectory_sha256'];runtime=next(v for v in s[label] if v['case']==case)
            members.append(dict(accepted=r['independently_accepted'],reason=runtime['reason'],time_s=runtime['time_s'],endpoints=r['independent_endpoints'],sampled_native_margin_rad=r['sampled_minimum_margin_rad']))
        before,after=members;rows.append(dict(case=case,offset_mm=[v*1000 for v in offsets['rung_01']],zero=before,policy=after,fixed=not before['accepted'] and after['accepted'],regressed=before['accepted'] and not after['accepted']))
    assert len(a['trajectories'])==2*len(rows)
    result=dict(rows=rows,zero_passed=sum(r['zero']['accepted'] for r in rows),policy_passed=sum(r['policy']['accepted'] for r in rows),total=len(rows),fixed=[r['case'] for r in rows if r['fixed']],regressed=[r['case'] for r in rows if r['regressed']],audit_sha256=sha(p/'independent_training_audit.json'),checkpoint_sha256=sha(p/'policy.zip'),source_sha256=sha(__file__),scope='Deterministic fixed-grid paired validation. Original physical criteria, no statistical or hardware claim. Used for model selection only under its separately frozen rules.')
    (p/'policy_validation_summary.json').write_text(json.dumps(result,indent=2));(p/'policy_validation_summary_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({k:result[k] for k in ('zero_passed','policy_passed','total','fixed','regressed')},indent=2))


if __name__=='__main__':main()
