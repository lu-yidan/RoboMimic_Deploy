"""Isolate contact memory from angular uncertainty cone tightening."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'tools/run_early_memory_suite.py').read_text(encoding='utf-8')
s=s.replace('normal_bias_y_deg=angle)', 'normal_bias_y_deg=angle,normal_uncertainty_deg=0.)')
s=s.replace('tools/run_early_memory_suite.py', 'tools/run_early_memory_only_suite.py')
s=s.replace('normal_uncertainty_deg=3.', 'normal_uncertainty_deg=0.')
s=s.replace('axis uncertainty plus <=20 ms contact memory', 'no angular-uncertainty tightening; <=20 ms contact memory')
s=s.replace('known world-Y uncertainty axis and ideal force feedback retained', 'exact geometric gap and ideal force feedback retained; no angle-interval robustness claim')
p=root/'tools/run_early_memory_only_suite.py';assert not p.exists();p.write_text(s,encoding='utf-8')
