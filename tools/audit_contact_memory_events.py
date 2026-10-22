"""Check logged memory age/gap/support bounds without claiming force-history replay."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.suite/'manifest.json').read_text())
    protocol=json.loads((Path(manifest['suite'])/'protocol.json').read_text())
    reference=dict(np.load(protocol['reference']))
    results=json.loads((a.suite/'results.json').read_text())
    checks=[]
    for result in results:
        path=a.suite/result['trial']/'memory_events.json'
        events=json.loads(path.read_text());violations=[]
        for i,event in enumerate(events):
            j=min(max(int(np.searchsorted(reference['time'],event['time_s'],side='right')),1),len(reference['time'])-1)
            group=['lf','rf','lh','rh'].index(event['group'])
            valid=0<=event['age_s']<=.020000001 and event['gap_m']<=.000100001 and bool(reference['active'][j,group])
            if not valid:violations.append(i)
        checks.append(dict(trial=result['trial'],passed=not violations and len(events)==result['memory_uses'],count=len(events),max_age_s=max((e['age_s'] for e in events),default=0.),max_gap_m=max((e['gap_m'] for e in events),default=0.),violations=violations,events_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    output=dict(passed=all(r['passed'] for r in checks),trials=checks,scope='Logged age/gap/scheduled-support bounds only. Previous >=10N force gate enforced in frozen controller; not independently reconstructed from high-rate force history. Cached point is not measured current contact.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.suite/'memory_bounds_audit.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))
    if not output['passed']:raise SystemExit(1)


if __name__=='__main__':main()
