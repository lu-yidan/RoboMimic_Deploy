"""Verify all two-cycle gates without waiting for the longer eight-cycle suite."""
import argparse
import json
from pathlib import Path
from tools.summarize_v2_multicycle import read,sha,verify_suite_member,summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    protocol=read(a.suite/'protocol.json');parent=Path(protocol['candidate']);pm=read(parent/'manifest.json')
    assert sha(parent/'manifest.json')==protocol['parent_manifest_sha256']
    assert sha(parent/'policy.zip')==protocol['checkpoint_sha256']
    assert sha(a.suite/'runner_source.py')==protocol['source_sha256']
    cases=protocol['cases'];assert len(cases)==len(set(cases))==24
    assert set(cases)==set(pm['evaluation_cases'])
    rows=[]
    for case in cases:
        reused=protocol['reuse'][case]['two']
        run=Path(reused) if reused else a.suite/'2_cycles'/case
        verify_suite_member(run,case,2,protocol,pm)
        row=summarize(run)
        assert row['case']==case and row['declared_cycles']==2
        rows.append(row)
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(passed=all(r['passed'] for r in rows),cases=24,passed_cases=sum(r['passed'] for r in rows),
        cycle_endpoints=sum(r['completed_cycles'] for r in rows),rows=rows,
        final_rise_range_m=[min(r['final_rise_m'] for r in rows),max(r['final_rise_m'] for r in rows)],
        minimum_endpoint_normal_n=min(c['min_normal_n'] for r in rows for c in r['cycle_metrics']),
        minimum_endpoint_vertical_n=min(c['min_vertical_n'] for r in rows for c in r['cycle_metrics']),
        protocol_sha256=sha(a.suite/'protocol.json'),source_sha256=sha(__file__),
        shared_auditor_sha256=sha(Path(__file__).with_name('summarize_v2_multicycle.py')),new_training_steps=0,
        scope='Complete 24-case two-cycle development gates only; does not declare completion of the still-running eight-cycle suite. Exact first-cycle replay, one physical reset, source-bound physical/policy audits. Local rung_01 geometry, not unseen or random-wall generalization.')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    (a.output/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    (a.output/'shared_analysis_source.py').write_bytes(Path(__file__).with_name('summarize_v2_multicycle.py').read_bytes())
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','scope')}),flush=True)


if __name__=='__main__':main()
