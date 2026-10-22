"""Create a classical contact-memory comparison on the same new prefixes."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'tools/run_contact_memory_suite.py').read_text(encoding='utf-8')
s=s.replace('training.load_transfer_residual_env','training.contact_load_transfer_residual_env')
s=s.replace("'training/load_transfer_residual_env.py'", "'training/contact_load_transfer_residual_env.py','training/contact_frame_controller.py'")
s=s.replace("'training/foot_residual_env.py'", "'training/contact_foot_residual_env.py'")
s=s.replace('tools/audit_load_transfer.py','tools/audit_contact_load_transfer.py')
s=s.replace('tools/run_contact_memory_suite.py','tools/run_early_memory_suite.py')
s=s.replace("('normal_plus3deg',0.,3.)]", "('normal_plus3deg',0.,3.),('normal_minus1p5deg',0.,-1.5),('normal_plus1p5deg',0.,1.5)]")
s=s.replace('zero residual; axis uncertainty', 'zero residual; validated pre-reset history and native position checks every physics step; axis uncertainty')
p=root/'tools/run_early_memory_suite.py';assert not p.exists();p.write_text(s,encoding='utf-8')
