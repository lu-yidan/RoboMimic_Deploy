"""Add pure contact-memory baseline with explicit extra information and bounds."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    protocol=read(a.run/'protocol.json');base=ROOT/'outputs/untrained_geometry_comparison_v1';prior=read(base/'paired_summary.json');assert prior['integrity_passed']
    rows=[];events_audit=[];proof={}
    for angle in protocol['angles']:
        run=a.run/f'memory__normal_{angle:+g}deg';m=read(run/'manifest.json');original=read(base/f'zero__normal_{angle:+g}deg'/'manifest.json');data=read(run/'results.json')
        assert read(run/'independent_audit.json')['integrity_passed']
        assert len(data)==5 and {r['case'] for r in data}==set(protocol['cases'])
        assert m['prefixes']==original['prefixes']
        for rel in ['training/contact_load_transfer_residual_env.py','training/contact_foot_residual_env.py','training/climb_cycle.py','training/climb_robustness.py','training/wall_support_env.py']:
            assert m['sources'][rel]==original['sources'][rel]
        ref=dict(np.load(read(Path(m['suite'])/'protocol.json')['reference']))
        for row in data:
            path=run/(row['case']+'_memory_events.json');events=read(path);bad=[]
            for i,event in enumerate(events):
                j=int(np.clip(np.searchsorted(ref['time'],event['time_s'],side='right'),1,len(ref['time'])-1));group=['lf','rf','lh','rh'].index(event['group'])
                if not(0<=event['age_s']<=.020000001 and event['gap_m']<=.000100001 and ref['active'][j,group]):bad.append(i)
            assert not bad and len(events)==row['memory_uses']
            events_audit.append(dict(case=row['case'],angle=angle,uses=len(events),max_age_s=max((x['age_s'] for x in events),default=0.),max_gap_m=max((x['gap_m'] for x in events),default=0.),events_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            rows.append(row)
        proof[run.name]=hashlib.sha256((run/'results.json').read_bytes()).hexdigest()
    mapped={(r['case'],r['normal_bias_y_deg']):r for r in rows};zero={(r['case'],r['angle']):r['methods']['zero'] for r in prior['scenarios']}
    assert set(mapped)==set(zero) and len(mapped)==15
    counts=dict(successes=sum(r['success'] for r in rows),count=15,added_vs_zero=[list(k) for k in sorted(mapped) if mapped[k]['success'] and not zero[k]['success']],lost_vs_zero=[list(k) for k in sorted(mapped) if not mapped[k]['success'] and zero[k]['success']])
    result=dict(integrity_passed=True,counts=dict(prior['counts'],memory=counts),memory_events=events_audit,memory_results=rows,source_hashes=proof,scope='Same prefixes/sharedcore and frozen zero actions. Memory uses additional exact geometric gap and <=20ms cache with historicalforce gate; not equalinformation to PPO. Age/gap/active audited, historicalforce gate from frozen source, not independent force replay. No statistical or blind-test claim.')
    (a.run/'paired_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['counts'],indent=2))

if __name__=='__main__':main()
