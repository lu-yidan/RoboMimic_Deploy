"""Pair exactly the same nine cases across completed controller ablations."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    sources=['contact_bias_segments_v1','robust_contact_segments_v1','robust_tracking_segments_v1','axis_contact_segments_v1','contact_memory_segments_v1']
    keys=None;base=None;protocol=None;table=[]
    for folder in sources:
        root=ROOT/'outputs'/folder;manifest=json.loads((root/'manifest.json').read_text())
        rows=[r for r in json.loads((root/'results.json').read_text()) if r['point_bias_x_m']==0 and r['normal_bias_y_deg'] in (-3.,0.,3.)]
        cases={(r['case'],r['normal_bias_y_deg']):r for r in rows};assert len(cases)==9
        signature=dict(prefixes=manifest['prefixes'],core={k:manifest['sources'][k] for k in ['training/load_transfer_residual_env.py','training/foot_residual_env.py','training/climb_cycle.py','training/climb_env.py','training/wall_support_env.py','training/climb_robustness.py']})
        if protocol is None:protocol=signature;keys=set(cases);base=cases
        assert signature==protocol and set(cases)==keys
        assert all(r['integrity_passed'] for r in rows)
        table.append(dict(source=folder,successes=sum(r['success'] for r in rows),total=9,
            by_bias_deg={str(bias):sum(cases[(name,bias)]['success'] for name in ['nominal','row01_z_+5mm','row01_x_-10mm']) for bias in (-3.,0.,3.)},
            added_vs_original=[list(k) for k in sorted(keys) if cases[k]['success'] and not base[k]['success']],
            lost_vs_original=[list(k) for k in sorted(keys) if not cases[k]['success'] and base[k]['success']],
            results_sha256=hashlib.sha256((root/'results.json').read_bytes()).hexdigest()))
    out=ROOT/'outputs/contact_estimator_comparison_v1';out.mkdir(exist_ok=False)
    result=dict(table=table,matching_protocol=protocol,
        scope='Same nine recorded-prefix 49..61.56 s segments, zero residual, no learning. Exact topology/force feedback, known uncertainty axis for axis variants, and exact geometric gap for memory remain assumptions. Not a full-cycle or real-sensor robustness result.')
    (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(table,indent=2))


if __name__=='__main__':main()
