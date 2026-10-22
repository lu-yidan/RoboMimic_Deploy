"""Check reference-only prefix equality and the original zero-action replay."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.continue_vector_full_cycle_training import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('experiment',type=Path);args=parser.parse_args();p=args.experiment
    rows=json.loads((p/'independent_comparison.json').read_text())['results']
    for r in rows:
        assert r['integrity_passed'] and sha(p/r['label']/'independent_audit.json')==r['audit_sha256']
    parent=Path('outputs/bar_geometry_blind98304_v1')
    old_audit=json.loads((parent/'independent_training_audit.json').read_text())
    assert old_audit['passed'] and old_audit['experiment_summary_sha256']==sha(parent/'summary.json')
    old_summary=json.loads((parent/'summary.json').read_text())
    old_zero=next(x for x in old_summary['zero'] if x['case']=='foot_z_plus10')
    old_path=parent/'zero_foot_z_plus10.npz'
    assert sha(old_path)==old_zero['trajectory_sha256']
    with np.load(old_path) as a,np.load(p/'original_bias-3/trajectory.npz') as b:
        old_match={k:bool(np.array_equal(a[k],b[k])) for k in a.files if k in b.files}
    assert len(old_match)>=6 and all(old_match.values()),old_match
    pairs=[]
    for bias in ('+0','-3'):
        left=p/f'clearance_bias{bias}';right=p/f'adapted_bias{bias}'
        lm=json.loads((left/'manifest.json').read_text());rm=json.loads((right/'manifest.json').read_text())
        assert lm['sources']==rm['sources'] and lm['offsets']==rm['offsets'] and lm['normal_bias_y_deg']==rm['normal_bias_y_deg']
        plan_manifest=json.loads((Path(rm['reference']).parent/'manifest.json').read_text())
        start=plan_manifest['blend_window_s'][0]
        with np.load(left/'trajectory.npz') as a,np.load(right/'trajectory.npz') as b:
            na=int(np.searchsorted(a['time_s'],start,side='left'));nb=int(np.searchsorted(b['time_s'],start,side='left'))
            assert na==nb
            matches={k:bool(np.array_equal(a[k][:na],b[k][:nb])) for k in ('qpos','qvel','time_s','torques','actions')}
        assert all(matches.values()),matches
        pairs.append(dict(bias=bias,reference_blend_start_s=start,matched_prefix_samples=na,exact=matches,
            note='Actor observations/rewards depend on the reference and are excluded. State/torque prefix is compared only before blending.'))
    result=dict(original_zero_replay_exact=old_match,original_zero_sha256=sha(old_path),prefix_pairs=pairs,
        comparison_sha256=sha(p/'independent_comparison.json'),source_sha256=sha(__file__),
        scope='Within-bias reference comparison. Adapted endpoint and duration differ from original policy task; no pooled seven-case success claim.')
    (p/'paired_checks.json').write_text(json.dumps(result,indent=2));(p/'paired_checks_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
