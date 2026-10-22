"""Paired scenario results with regressions and unequal information stated."""
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def key(row):
    return row['case'], float(row['normal_bias_y_deg'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--memory',type=Path);a=p.parse_args()
    manifest=read(a.run/'manifest.json');summary=read(a.run/'summary.json')
    audit=read(a.run/'independent_audit.json')
    mapped={label:{key(r):r for r in summary[label]} for label in ['zero','before','after']}
    expected={(case,float(angle)) for case in manifest['cases'] for angle in manifest['evaluation_angles_deg']}
    assert len(expected)==15 and all(set(rows)==expected for rows in mapped.values())
    result=dict(training_steps=summary['actual_steps'],integrity_passed=audit['integrity_passed'],scope='One seed and fifteen deterministic scenarios; no statistical success probability or full-cycle policy claim.',counts={},added_vs_zero=[],lost_vs_zero=[],scenarios=[])
    for label,rows in mapped.items():
        result['counts'][label]=dict(all=sum(r['success'] for r in rows.values()),training_angles=sum(r['success'] for r in rows.values() if not r['evaluation_only_angle']),evaluation_only_angles=sum(r['success'] for r in rows.values() if r['evaluation_only_angle']))
    for case,angle in sorted(expected):
        rows={label:mapped[label][case,angle] for label in mapped}
        result['scenarios'].append(dict(case=case,normal_bias_y_deg=angle,**{label:dict(success=r['success'],time_s=r['time_s'],reason=r['reason'],return_sum=r['return_sum']) for label,r in rows.items()}))
        if rows['after']['success'] and not rows['zero']['success']:result['added_vs_zero'].append(rows['after']['scenario'])
        if rows['zero']['success'] and not rows['after']['success']:result['lost_vs_zero'].append(rows['after']['scenario'])
    with (a.run/'episodes.monitor.csv').open() as stream:
        stream.readline();episodes=list(csv.DictReader(stream))
    result['completed_training_episode_counts']=dict(Counter(r['scenario'] for r in episodes))
    result['training_sampling_note']='50% probability per reset for known hard scenario; episode length differences mean not50% of transition samples.'
    if a.memory:
        memory_manifest=read(a.memory/'manifest.json');memory={key(r):r for r in read(a.memory/'results.json')}
        assert set(memory)==expected
        assert manifest['prefixes']==memory_manifest['prefixes']
        for path in ['training/contact_load_transfer_residual_env.py','training/contact_foot_residual_env.py','training/climb_cycle.py','training/wall_support_env.py','training/climb_robustness.py']:
            assert manifest['sources'][path]==memory_manifest['sources'][path],path
        result['memory_comparison']=dict(successes=sum(r['success'] for r in memory.values()),all_integrity_passed=all(r['integrity_passed'] for r in memory.values()),assumptions='Additional known-world-Y uncertainty interval and exact geometric-gap gated <=20ms memory. Not the same information budget as raw biased-normal/PPO control.',scenarios=[dict(case=case,normal_bias_y_deg=angle,success=row['success'],reason=row['reason'],time_s=row['time_s'],memory_uses=row['memory_uses']) for (case,angle),row in sorted(memory.items())])
    result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (a.run/'paired_comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['scenarios','memory_comparison']},indent=2))
    if not result['integrity_passed']:raise SystemExit(1)


if __name__=='__main__':main()
