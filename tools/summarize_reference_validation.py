"""Report every frozen validation case and horizontal-offset subgroup."""
import argparse
import json
from pathlib import Path
from tools.continue_vector_full_cycle_training import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('experiment',type=Path);args=parser.parse_args();p=args.experiment
    protocol=json.loads((p/'protocol.json').read_text());audit=json.loads((p/'independent_comparison.json').read_text());results=json.loads((p/'results.json').read_text())
    labels={j[1] for j in protocol['jobs']};assert len(labels)==len(protocol['jobs'])==len(results)==len(audit['results']) and labels=={r['label'] for r in results}=={r['label'] for r in audit['results']}
    assert audit['protocol_sha256']==sha(p/'protocol.json');rows=[]
    for r in audit['results']:
        directory=p/r['label'];a=json.loads((directory/'independent_audit.json').read_text());m=json.loads((directory/'manifest.json').read_text())
        assert r['audit_sha256']==sha(directory/'independent_audit.json') and a['integrity_passed'] and a['trajectory_sha256']==sha(directory/'trajectory.npz')
        endpoint=a['endpoints'][0];rows.append(dict(case=m['case'],offset_mm=[v*1000 for v in m['offsets']['rung_01']],passed=a['independently_accepted'],reason=r['reason'],time_s=r['time_s'],sampled_native_margin_rad=a['sampled_native_margin_rad'],endpoint=endpoint,audit_sha256=sha(directory/'independent_audit.json')))
    groups=[]
    for x in sorted({r['offset_mm'][0] for r in rows}):
        subset=[r for r in rows if r['offset_mm'][0]==x];groups.append(dict(x_mm=x,passed=sum(r['passed'] for r in subset),total=len(subset)))
    result=dict(passed_cases=sum(r['passed'] for r in rows),total_cases=len(rows),groups=groups,rows=rows,protocol_sha256=sha(p/'protocol.json'),audit_sha256=sha(p/'independent_comparison.json'),source_sha256=sha(__file__),scope=protocol['scope'])
    (p/'validation_summary.json').write_text(json.dumps(result,indent=2));(p/'summary_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({k:result[k] for k in ('passed_cases','total_cases','groups')},indent=2))


if __name__=='__main__':main()
