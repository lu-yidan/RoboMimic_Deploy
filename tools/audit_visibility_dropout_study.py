"""Check declared localization blackouts and persistence expiry without hiding failed tasks."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
def read(p):return json.loads(p.read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();protocol=read(a.run/'protocol.json');rows=read(a.run/'results.json')
    expected={(d,b) for d in protocol['durations_s'] for b in protocol['persistence']};assert len(rows)==8 and {(r['duration_s'],r['persistence']) for r in rows}==expected
    base=dict(np.load(a.run/'raw_0ms/nominal.npz'));m0=read(a.run/'raw_0ms/manifest.json');results=[]
    for row in rows:
        run=a.run/row['run'];m=read(run/'manifest.json');z=dict(np.load(run/'nominal.npz'));memory=read(run/'memory_events.json');masks=read(run/'mask_events.json')
        assert row['integrity_passed'] and read(run/'independent_audit.json')['integrity_passed']
        assert m['sources']==m0['sources'] and m['prefixes']==m0['prefixes']
        n=int(np.sum(base['time_s']<=53.65+1e-8));prefix=float(np.max(abs(z['qpos'][:n]-base['qpos'][:n])))
        age_ok=all(0<=e['age_s']<=.020000001 and e['gap_query_used'] is False and e['group']=='lh' for e in memory)
        masks_ok=all(e['group']=='lh' and 53.65<=e['time_s']<53.65+row['duration_s'] for e in masks)
        expected_success=row['duration_s']==0. or row['persistence'] and row['duration_s']<=.02
        expected_reason='Contact memory expired for lh' if row['persistence'] and row['duration_s']>.02 else 'No visible rung contact for scheduled support lh'
        behavior=row['success']==expected_success and (row['success'] or expected_reason in row['reason'])
        results.append(dict(**row,comparison_integrity=prefix<1e-10 and age_ok and masks_ok,prefix_qpos_difference=prefix,memory_uses=len(memory),max_memory_age_s=max((e['age_s'] for e in memory),default=0.),mask_samples=len(masks),expected_behavior=bool(behavior)))
    result=dict(integrity_passed=all(r['comparison_integrity'] for r in results),expected_bounded_behavior=all(r['expected_behavior'] for r in results),results=results,scope=protocol['scope'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.run/'study_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if not result['integrity_passed']:raise SystemExit(1)

if __name__=='__main__':main()
