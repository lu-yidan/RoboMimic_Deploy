"""Create an isolated, reviewable gain-reset ablation; never alter live runners."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = root / 'deploy_mujoco/run_early_contact_cycle.py'
dst = src.with_name('run_early_contact_reset_cycle.py')
old = '                transfer_entered=False\n                controller_cycle=current_cycle'
new = '                transfer_entered=False\n                transfer_controller.leg_kp=normal_gain\n                controller_cycle=current_cycle'
s = src.read_text(encoding='utf-8')
assert s.count(old) == 1
assert not dst.exists()
s = s.replace(old, new).replace('deploy_mujoco/run_early_contact_cycle.py', 'deploy_mujoco/run_early_contact_reset_cycle.py')
dst.write_text(s, encoding='utf-8')
src = root / 'tools/run_early_contact_suite.py'
dst = src.with_name('run_early_contact_reset_suite.py')
assert not dst.exists()
s = src.read_text(encoding='utf-8').replace('deploy_mujoco.run_early_contact_cycle', 'deploy_mujoco.run_early_contact_reset_cycle')
s = s.replace('early actual-contact controller with baseline contact establishment gate', 'early actual-contact controller with baseline contact establishment gate and explicit cycle-start contact-controller gain reset')
dst.write_text(s, encoding='utf-8')
