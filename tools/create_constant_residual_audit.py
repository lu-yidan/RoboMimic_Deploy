"""Check recorded diagnostic actions rather than pretending they are zero."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'tools/audit_contact_load_transfer.py').read_text(encoding='utf-8')
s=s.replace("else 'zero residual' in manifest['contract']", "else bool(z['actions'].shape==(len(z['time_s'])-1,6) and np.isfinite(z['actions']).all() and np.max(abs(z['actions']))<=1.+1e-6 and np.allclose(z['actions'],manifest['constant_action'],atol=0,rtol=0))")
s=s.replace('Zero-residual diagnostic source contract only; no action samples or policy-action audit claim','Recorded constant actions match the declared bounded vector; no learned policy')
p=root/'tools/audit_constant_residual.py';assert not p.exists();p.write_text(s,encoding='utf-8')
