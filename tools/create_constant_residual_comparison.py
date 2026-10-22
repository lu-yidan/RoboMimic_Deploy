"""Select the smallest tested successful action, then freeze a15-scenario check."""
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=root/'outputs/residual_authority_probe_v1/audited_summary.json'
data=json.loads(source.read_text())
candidates=[r for r in data['results'] if r['runtime_success'] and r['audit_passed']]
def score(row):
    action=row['constant_action']
    return sum(v*v for v in action),next(i for i,v in enumerate(action) if v),tuple(action)
chosen=min(candidates,key=score);action=chosen['constant_action']
selection=dict(selected_trial=chosen['trial'],constant_action=action,residual_scale_rad=.015,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),selection_rule='Among audited successful actions choose minimum squared norm, then lowest active joint index, then lexicographic action. Selection uses one known hardcase only. Subsequent15cases are not a statistical success probability.')
path=root/'outputs/constant_residual_selection_v1.json';assert not path.exists();path.write_text(json.dumps(selection,indent=2))
s=(root/'tools/run_early_contact_bias_suite.py').read_text(encoding='utf-8')
s=s.replace('action=np.zeros(6)',f'action=np.array({action!r},dtype=float)')
s=s.replace('tools/run_early_contact_bias_suite.py','tools/run_constant_residual_comparison.py')
s=s.replace('tools/audit_contact_load_transfer.py','tools/audit_constant_residual.py')
s=s.replace("('normal_plus3deg',0.,3.)]", "('normal_plus3deg',0.,3.),('normal_minus1p5deg',0.,-1.5),('normal_plus1p5deg',0.,1.5)]")
s=s.replace('manifest=dict(suite=',f"manifest=dict(constant_action={action!r},selection={selection!r},suite=")
s=s.replace('zero residual;', 'selected bounded constant residual;')
p=root/'tools/run_constant_residual_comparison.py';assert not p.exists();p.write_text(s,encoding='utf-8')
print(json.dumps(selection,indent=2))
