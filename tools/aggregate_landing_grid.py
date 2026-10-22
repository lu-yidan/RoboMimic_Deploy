"""Compare completed landing grid with the prior exact-contact grid."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    previous={r['name']:r for r in json.loads((ROOT/'outputs/contact_frame_continued_grid_v1/results.json').read_text())}
    joined={};hashes={}
    for folder in ['contact_landing_fullcycle_v1','contact_landing_remaining_v1']:
        source=ROOT/'outputs'/folder
        hashes[folder]=hashlib.sha256((source/'results.json').read_bytes()).hexdigest()
        for row in json.loads((source/'results.json').read_text()):
            name=row['name'];assert name not in joined
            old=ROOT/previous[name]['source'];new=source/name
            before=json.loads((old/'manifest.json').read_text());after=json.loads((new/'manifest.json').read_text())
            assert before['plan_sha256']==after['plan_sha256']
            for key in ['joint_kp','torque_weight','leg_kp','leg_gain_mode','rung_offsets']:
                assert before['config'][key]==after['config'][key]
            assert json.loads((old/'scene_variant.json').read_text())['geometry_sha256']==json.loads((new/'scene_variant.json').read_text())['geometry_sha256']
            for rel in ['training/climb_cycle.py','training/wall_support_env.py','training/climb_env.py','training/climb_robustness.py']:
                assert before['sources'][rel]==after['sources'][rel]
            assert row['passed']==bool(json.loads((new/'summary.json').read_text())['passed'] and row['audit']['passed'] and row['landing_audit']['passed'])
            joined[name]=dict(row,source=str(new.relative_to(ROOT)))
    assert set(joined)==set(previous) and len(joined)==17
    completed=[n for n,r in joined.items() if r['reason']=='complete']
    summary=dict(cases=17,strict_passes=sum(r['passed'] for r in joined.values()),
        prior_strict_passes=sum(r['passed'] for r in previous.values()),completed=len(completed),
        prior_completed=sum(r['reason']=='complete' for r in previous.values()),
        added_strict_passes=[n for n,r in joined.items() if r['passed'] and not previous[n]['passed']],
        lost_strict_passes=[n for n,r in joined.items() if not r['passed'] and previous[n]['passed']],
        added_completed_cases=[n for n in completed if previous[n]['reason']!='complete'],
        complete_case_extra_audits_passed=sum(joined[n]['landing_audit']['passed'] for n in completed),
        source_results_sha256=hashes,
        matched='Nominal reference, initial scene/BVH, physical model/control core source hashes and gains; online landing correction is the experimental change.',
        scope='17 deterministic cases; exact contact topology and ideal force feedback. Completion does not imply unchanged periodic acceptance.')
    out=ROOT/'outputs/contact_landing_grid_v1';out.mkdir(exist_ok=False)
    (out/'results.json').write_text(json.dumps(list(joined.values()),indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
