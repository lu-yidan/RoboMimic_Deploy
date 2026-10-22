"""Create a separate runner; preserve early-controller sources and checkpoints."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'deploy_mujoco/run_early_contact_reset_cycle.py').read_text(encoding='utf-8')
s=s.replace('from training.contact_frame_controller import ContactFrameController','from training.window_contact_controller import WindowContactController as ContactFrameController')
s=s.replace("extra_sources=[", "extra_sources=['training/window_contact_controller.py','training/contact_memory_controller.py','training/biased_contact_frame_controller.py',")
s=s.replace('deploy_mujoco/run_early_contact_reset_cycle.py','deploy_mujoco/run_window_memory_cycle.py')
old="transfer_controller=transfer_class(env,joint_kp=normal_gain,torque_weight=float(cfg.get('torque_weight',0.)),leg_kp=normal_gain) if use_transfer_frame else None"
new="transfer_controller=transfer_class(env,joint_kp=normal_gain,torque_weight=float(cfg.get('torque_weight',0.)),leg_kp=normal_gain,normal_bias_y_deg=float(cfg.get('window_normal_bias_y_deg',-3.)),memory_enabled=bool(cfg.get('window_memory_enabled',False))) if use_transfer_frame else None"
assert old in s;s=s.replace(old,new)
s=s.replace('                if missing:\n', '                if missing and not transfer_controller.bias_window_active():\n')
s=s.replace("    (out/'contact_gate_events.json').write_text", "    (out/'memory_events.json').write_text(json.dumps(transfer_controller.memory_events,indent=2))\n    (out/'contact_gate_events.json').write_text")
p=root/'deploy_mujoco/run_window_memory_cycle.py';assert not p.exists();p.write_text(s,encoding='utf-8')
