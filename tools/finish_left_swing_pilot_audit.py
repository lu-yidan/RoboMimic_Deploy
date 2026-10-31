"""Finish report-only checks after the pilot's summary-field lookup error."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.compare_audited_closure_candidates import load_run,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();out=a.directory
    m=json.loads((out/'manifest.json').read_text());s=json.loads((out/'summary.json').read_text());current=load_run(out)
    hashes={f:sha(out/f) for f in ('manifest.json','summary.json')};rows=[]
    for name in ('controller_modifier_audit_v2.json','adaptive_reference_audit.json'):
        audit=json.loads((out/name).read_text());assert audit['passed'] and audit['manifest_sha256']==hashes['manifest.json'] and audit['summary_sha256']==hashes['summary.json']
    for record in s['after']:
        case=record['case'];parent=Path(m['case_parents'][case]['path']);old=load_run(parent)
        assert old['audit_sha256']==m['case_parents'][case]['audit_sha256']
        before=parent/f'after_{case}.npz'
        old_summary=json.loads((parent/'summary.json').read_text())
        assert sha(before)==next(r for r in old_summary['after'] if r['case']==case)['trajectory_sha256']
        with np.load(before) as x,np.load(out/f'after_{case}.npz') as y:
            exact={}
            for key in ('qpos','qvel','time_s','actions','observations','torques','rewards'):
                n=3511 if key in ('qpos','qvel','time_s') else 3510
                exact[key]=bool(np.array_equal(x[key][:n],y[key][:n]))
        rows.append(dict(case=case,passed=all(exact.values()),exact_before_70_2_s=exact,success=record['success'],reason=record['reason'],
                         before=old['rows'][case],after=current['rows'][case]))
    result=dict(passed=all(r['passed'] for r in rows),successes=sum(r['success'] for r in rows),cases=len(rows),rows=rows,
                manifest_sha256=hashes['manifest.json'],summary_sha256=hashes['summary.json'],automatic_promotion=False,source_sha256=sha(__file__),
                report_repair='Original evaluator completed dynamics and three audits, then looked up trajectory_sha256 in compact load_run rows. This report uses the original audited summary hash instead. No rerun, source-snapshot edit, manifest edit or outcome change.')
    assert not (out/'pilot_audit.json').exists()
    (out/'pilot_audit.json').write_text(json.dumps(result,indent=2));(out/'pilot_audit_repair_source.py').write_bytes(Path(__file__).read_bytes())
    assert all(sha(out/f)==digest for f,digest in hashes.items());print(json.dumps(result));assert result['passed']


if __name__=='__main__':main()
