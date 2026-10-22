"""Create a zero-action diagnostic matching the isolated contact environment."""
from pathlib import Path
root = Path(__file__).resolve().parents[1]
s = (root / 'tools/check_load_transfer.py').read_text(encoding='utf-8')
s = s.replace('training.load_transfer_residual_env', 'training.contact_load_transfer_residual_env')
s = s.replace("'training/load_transfer_residual_env.py','training/foot_residual_env.py'", "'training/contact_load_transfer_residual_env.py','training/contact_foot_residual_env.py','training/contact_frame_controller.py'")
s = s.replace("'tools/check_load_transfer.py'", "'tools/check_contact_load_transfer.py'")
s = s.replace('zero residual baseline', 'zero residual baseline; exact contact geometry controller; per-physics-step native joint bounds')
p = root / 'tools/check_contact_load_transfer.py'
assert not p.exists()
p.write_text(s, encoding='utf-8')
