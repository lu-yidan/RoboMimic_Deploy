"""Join disjoint completed grids, verifying common reference and scene hashes."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    base=ROOT/'outputs/clearance_back20_leg14_grid_v1'
    sources=[ROOT/'outputs/contact_frame_continued_cycle_v1',ROOT/'outputs/contact_frame_continued_remaining_v1']
    original={r['name']:r for r in json.loads((base/'results.json').read_text())}
    joined={};hashes={}
    for source in sources:
        hashes[str(source.relative_to(ROOT))]=hashlib.sha256((source/'results.json').read_bytes()).hexdigest()
        for row in json.loads((source/'results.json').read_text()):
            name=row['name'];assert name not in joined
            before=json.loads((base/name/'manifest.json').read_text())
            after=json.loads((source/name/'manifest.json').read_text())
            assert before['plan_sha256']==after['plan_sha256']
            for key in ('joint_kp','torque_weight','leg_kp','leg_gain_mode','rung_offsets'):
                assert before['config'][key]==after['config'][key],(name,key)
            old_scene=json.loads((base/name/'scene_variant.json').read_text())
            new_scene=json.loads((source/name/'scene_variant.json').read_text())
            assert old_scene['geometry_sha256']==new_scene['geometry_sha256']
            joined[name]=dict(row,source=str((source/name).relative_to(ROOT)))
    assert set(joined)==set(original) and len(joined)==17
    out=ROOT/'outputs/contact_frame_continued_grid_v1';out.mkdir(exist_ok=False)
    summary=dict(cases=17,baseline_passed=sum(r['passed'] for r in original.values()),
        contact_frame_passed=sum(r['passed'] for r in joined.values()),
        completed=sum(r['reason']=='complete' for r in joined.values()),
        added_passes=[n for n,r in joined.items() if r['passed'] and not original[n]['passed']],
        lost_passes=[n for n,r in joined.items() if not r['passed'] and original[n]['passed']],
        source_hashes=hashes,matched='Reference SHA256, scene geometry/BVH SHA256, gains, torque weight and offsets',
        scope='17 deterministic cases, not a statistical probability; privileged exact contact positions, normals and identity from weight transfer onward')
    (out/'results.json').write_text(json.dumps(list(joined.values()),indent=2))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
