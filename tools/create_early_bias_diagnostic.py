"""Create a matched nine-case normal-bias diagnostic on validated new prefixes."""
from pathlib import Path
root = Path(__file__).resolve().parents[1]
s = (root / 'tools/run_contact_bias_suite.py').read_text(encoding='utf-8')
s = s.replace('training.load_transfer_residual_env', 'training.contact_load_transfer_residual_env')
s = s.replace("'training/load_transfer_residual_env.py'", "'training/contact_load_transfer_residual_env.py','training/contact_frame_controller.py'")
s = s.replace("'training/foot_residual_env.py'", "'training/contact_foot_residual_env.py'")
s = s.replace('tools/audit_load_transfer.py', 'tools/audit_contact_load_transfer.py')
s = s.replace('tools/run_contact_bias_suite.py', 'tools/run_early_contact_bias_suite.py')
start = s.index("    biases=[")
end = s.index('    sources={}', start)
s = s[:start] + "    biases=[('zero',0.,0.),('normal_minus3deg',0.,-3.),('normal_plus3deg',0.,3.)]\n" + s[end:]
s = s.replace('fixed position/normal calibration biases', 'fixed normal calibration biases; validated pre-reset history; per-physics-step native joint positions')
p = root / 'tools/run_early_contact_bias_suite.py'
assert not p.exists()
p.write_text(s, encoding='utf-8')
