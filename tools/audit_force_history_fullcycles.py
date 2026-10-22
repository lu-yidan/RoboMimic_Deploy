"""Compare continuous no-gap persistence to frozen geometric-memory full cycles."""
import argparse
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);a=p.parse_args();baseline=ROOT/'outputs/window_memory_fullcycles_v1';results=[]
    for case,_ in read(a.suite/'protocol.json')['cases']:
        run=a.suite/('persistence__'+case);old=baseline/('memory__'+case);m=read(run/'manifest.json');m0=read(old/'manifest.json');configs=[dict(v['config']) for v in [m,m0]]
        for c in configs:c.pop('output_dir')
        common=set(m['sources'])&set(m0['sources']);ref=dict(np.load(run/'reference.npz'));z=dict(np.load(run/'trajectory.npz'));original=dict(np.load(old/'trajectory.npz'))
        events=read(run/'memory_events.json');prior=read(old/'memory_events.json');valid=True
        for event in events:
            j=int(np.clip(np.searchsorted(ref['time'],event['time_s'],side='right'),1,len(ref['time'])-1));group=['lf','rf','lh','rh'].index(event['group'])
            valid &= 49.<=event['time_s']<61.56 and 0<=event['age_s']<=.020000001 and bool(ref['active'][j,group]) and event['gap_query_used'] is False
        qdelta=float(np.max(abs(z['qpos']-original['qpos']))) if z['qpos'].shape==original['qpos'].shape else None
        vdelta=float(np.max(abs(z['qvel']-original['qvel']))) if z['qvel'].shape==original['qvel'].shape else None
        checks=dict(shared_sources=all(m['sources'][s]==m0['sources'][s] for s in common),same_config=configs[0]==configs[1],same_reference=m['plan_sha256']==m0['plan_sha256'],same_geometry=read(run/'scene_variant.json')['geometry_sha256']==read(old/'scene_variant.json')['geometry_sha256'],memory_bounds=bool(valid),same_event_schedule=[(e['time_s'],e['group'],e['age_s']) for e in events]==[(e['time_s'],e['group'],e['age_s']) for e in prior])
        accepted=all(read(run/name)['passed'] for name in ['summary.json','independent_audit.json','joint_limit_audit.json','contact_identity_audit.json'])
        results.append(dict(case=case,task_passed=accepted,comparison_integrity=all(checks.values()),checks=checks,qpos_difference=qdelta,qvel_difference=vdelta,memory_uses=len(events)))
    result=dict(integrity_passed=all(r['comparison_integrity'] for r in results),successes=sum(r['task_passed'] for r in results),results=results,scope='Only three continuous singlecycles with -3deg normalbias restricted49..61.56s. No-gap persistence removes current gap gate, not all privileged contact/force information. Same trajectories, if observed, do not prove arbitrary separation recovery.')
    (a.suite/'paired_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if not result['integrity_passed']:raise SystemExit(1)

if __name__=='__main__':main()
