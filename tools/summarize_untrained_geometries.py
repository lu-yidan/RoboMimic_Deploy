"""Require every declared case and audit; report paired gains and regressions."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    protocol=read(a.run/'protocol.json');parent=ROOT/'outputs/biased_load_transfer_ppo_32768_v1';original=read(parent/'manifest.json')
    assert not set(protocol['cases'])&set(original['cases'])
    assert hashlib.sha256((parent/'policy.zip').read_bytes()).hexdigest()==protocol['policy_sha256']
    assert hashlib.sha256((ROOT/'outputs/constant_residual_selection_v1.json').read_bytes()).hexdigest()==protocol['selection_sha256']
    methods={m:{} for m in protocol['methods']};prefixes=None;proof={}
    for angle in protocol['angles']:
        for method in protocol['methods']:
            path=a.run/f'{method}__normal_{angle:+g}deg';manifest=read(path/'manifest.json');rows=read(path/'results.json');audit=read(path/'independent_audit.json')
            assert audit['integrity_passed'] and len(audit['results'])==len(protocol['cases'])
            assert len(rows)==len(protocol['cases']) and {r['case'] for r in rows}==set(protocol['cases'])
            if prefixes is None:prefixes=manifest['prefixes']
            assert manifest['prefixes']==prefixes
            for source in ['training/contact_load_transfer_residual_env.py','training/contact_foot_residual_env.py','training/climb_cycle.py','training/contact_frame_controller.py','training/biased_contact_frame_controller.py','training/climb_robustness.py','training/wall_support_env.py']:
                assert manifest['sources'][source]==original['sources'][source],source
            assert manifest['policy_sha256']==protocol['policy_sha256']
            for row in rows:
                assert row['normal_bias_y_deg']==angle and row['method']==method
                methods[method][row['case'],angle]=row
            proof[path.name]=hashlib.sha256((path/'results.json').read_bytes()).hexdigest()
    expected={(c,float(angle)) for c in protocol['cases'] for angle in protocol['angles']}
    assert all(set(rows)==expected for rows in methods.values())
    counts={}
    for method,rows in methods.items():
        counts[method]=dict(successes=sum(r['success'] for r in rows.values()),count=len(rows),by_angle={str(angle):sum(r['success'] for (case,bias),r in rows.items() if bias==angle) for angle in protocol['angles']},added_vs_zero=[list(k) for k in sorted(expected) if rows[k]['success'] and not methods['zero'][k]['success']],lost_vs_zero=[list(k) for k in sorted(expected) if not rows[k]['success'] and methods['zero'][k]['success']])
    scenarios=[dict(case=case,angle=angle,methods={m:dict(success=rows[case,angle]['success'],reason=rows[case,angle]['reason'],time_s=rows[case,angle]['time_s']) for m,rows in methods.items()}) for case,angle in sorted(expected)]
    result=dict(integrity_passed=True,counts=counts,scenarios=scenarios,source_hashes=proof,scope=protocol['scope'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.run/'paired_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(counts,indent=2))

if __name__=='__main__':main()
