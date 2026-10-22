"""Audit force-history persistence and compare to the geometric-memory method."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--compare-memory',type=Path);a=p.parse_args();protocol=read(a.run/'protocol.json')
    parent=read(ROOT/'outputs/biased_load_transfer_ppo_32768_v1/manifest.json');results=[];proof={}
    for angle in protocol['angles']:
        run=a.run/f'persistence__normal_{angle:+g}deg';m=read(run/'manifest.json');rows=read(run/'results.json');audit=read(run/'independent_audit.json')
        assert audit['integrity_passed'] and len(rows)==len(protocol['cases']) and {r['case'] for r in rows}==set(protocol['cases'])
        for source in ['training/contact_load_transfer_residual_env.py','training/contact_foot_residual_env.py','training/climb_cycle.py','training/climb_robustness.py','training/wall_support_env.py']:
            assert m['sources'][source]==parent['sources'][source]
        reference=dict(np.load(read(Path(m['suite'])/'protocol.json')['reference']))
        for row in rows:
            path=run/(row['case']+'_memory_events.json');events=read(path);assert len(events)==row['memory_uses']
            for event in events:
                j=int(np.clip(np.searchsorted(reference['time'],event['time_s'],side='right'),1,len(reference['time'])-1));group=['lf','rf','lh','rh'].index(event['group'])
                assert 0<=event['age_s']<=.020000001 and reference['active'][j,group] and event['gap_query_used'] is False
            item=dict(case=row['case'],angle=angle,success=row['success'],reason=row['reason'],time_s=row['time_s'],memory_uses=len(events),max_age_s=max((v['age_s'] for v in events),default=0.),events_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if a.compare_memory:
                baseline=a.compare_memory/f'memory__normal_{angle:+g}deg';other=read(baseline/'manifest.json');assert other['prefixes']==m['prefixes']
                assert other['sources']['training/contact_memory_controller.py']==m['sources']['training/contact_memory_controller.py']
                z=dict(np.load(run/(row['case']+'.npz')));old=dict(np.load(baseline/(row['case']+'.npz')))
                item['qpos_max_difference_vs_geometric_memory']=float(np.max(abs(z['qpos']-old['qpos']))) if z['qpos'].shape==old['qpos'].shape else None
                item['qvel_max_difference_vs_geometric_memory']=float(np.max(abs(z['qvel']-old['qvel']))) if z['qvel'].shape==old['qvel'].shape else None
                oldevents=read(baseline/(row['case']+'_memory_events.json'))
                item['same_memory_event_schedule']=[(v['time_s'],v['group'],v['age_s']) for v in events]==[(v['time_s'],v['group'],v['age_s']) for v in oldevents]
            results.append(item)
        proof[run.name]=hashlib.sha256((run/'results.json').read_bytes()).hexdigest()
    expected={(c,float(angle)) for c in protocol['cases'] for angle in protocol['angles']};assert {(r['case'],r['angle']) for r in results}==expected
    result=dict(integrity_passed=True,successes=sum(r['success'] for r in results),count=len(results),results=results,source_hashes=proof,scope='No current geometric-gap query in persistence decision; <=20ms and previous>=10N force confirmation retained. Current contact points/topology and forcefeedback still ideal, same baseline assumptions. Age/schedule audited; historicalforce gate from frozen inherited source. No real sensing or arbitrary contact-loss guarantee.')
    (a.run/'study_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
