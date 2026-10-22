"""Audit matched continuous experiments and the limited intervention scope."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);a=p.parse_args()
    protocol=read(a.suite/'protocol.json');rows=[]
    for case,_ in protocol['cases']:
        raw=a.suite/('raw__'+case);memory=a.suite/('memory__'+case)
        m0=read(raw/'manifest.json');m1=read(memory/'manifest.json')
        assert m0['plan_sha256']==m1['plan_sha256']
        assert m0['sources']==m1['sources']
        assert read(raw/'scene_variant.json')['geometry_sha256']==read(memory/'scene_variant.json')['geometry_sha256']
        configs=[dict(m['config']) for m in [m0,m1]]
        for cfg in configs:
            cfg.pop('output_dir');cfg.pop('window_memory_enabled')
        assert configs[0]==configs[1]
        z0=dict(np.load(raw/'trajectory.npz'));z1=dict(np.load(memory/'trajectory.npz'))
        n=int(np.sum(z0['time_s']<=49.+1e-8))
        prefix_diff=float(np.max(abs(z0['qpos'][:n]-z1['qpos'][:n])))
        ref=dict(np.load(memory/'reference.npz'));events=read(memory/'memory_events.json');violations=[]
        for i,event in enumerate(events):
            j=min(max(int(np.searchsorted(ref['time'],event['time_s'],side='right')),1),len(ref['time'])-1)
            active=ref['active'][j,['lf','rf','lh','rh'].index(event['group'])]
            if not (49.<=event['time_s']<61.56 and 0<=event['age_s']<=.020000001 and event['gap_m']<=.000100001 and active):violations.append(i)
        raw_pass=bool(read(raw/'summary.json')['passed'] and read(raw/'independent_audit.json')['passed'] and read(raw/'joint_limit_audit.json')['passed'])
        memory_pass=bool(read(memory/'summary.json')['passed'] and read(memory/'independent_audit.json')['passed'] and read(memory/'joint_limit_audit.json')['passed'])
        identity=all(read(run/'contact_identity_audit.json')['passed'] for run in [raw,memory])
        rows.append(dict(case=case,raw_passed=raw_pass,memory_passed=memory_pass,prefix_qpos_max_difference=prefix_diff,pair_audit_passed=prefix_diff<1e-10 and not violations and identity,memory_uses=len(events),max_memory_age_s=max((e['age_s'] for e in events),default=0.),max_memory_gap_m=max((e['gap_m'] for e in events),default=0.),memory_event_violations=violations,full_qpos_max_difference=float(np.max(abs(z0['qpos']-z1['qpos']))) if z0['qpos'].shape==z1['qpos'].shape else None))
    output=dict(passed=all(r['pair_audit_passed'] for r in rows),raw_successes=sum(r['raw_passed'] for r in rows),memory_successes=sum(r['memory_passed'] for r in rows),pairs=rows,scope='Same reference, geometry, source and configuration except memory flag. Physical run continuous from0. Bias only49..61.56s; not persistent full-cycle sensing error. Cached geometry gate remains privileged; historicalforce gate checked in frozen source, not independent high-rate force replay.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.suite/'paired_audit.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))
    if not output['passed']:raise SystemExit(1)


if __name__=='__main__':main()
