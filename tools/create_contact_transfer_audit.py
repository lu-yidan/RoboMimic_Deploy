"""Preserve legacy audit while adding native positions to the new contract."""
from pathlib import Path
root = Path(__file__).resolve().parents[1]
s = (root / 'tools/audit_load_transfer.py').read_text(encoding='utf-8')
s = s.replace("z=np.load(a.run/(f'{label}_{name}.npz' if evaluation else name+'.npz'))", "z=dict(np.load(a.run/(f'{label}_{name}.npz' if evaluation else name+'.npz')))")
s = s.replace("complete=abs(z['time_s'][-1]-61.56)<1e-7", """complete=abs(z['time_s'][-1]-61.56)<1e-7
            native_margin=float(np.min(np.minimum(z['qpos'][:,e.qadr]-m.jnt_range[e.joints,0],m.jnt_range[e.joints,1]-z['qpos'][:,e.qadr])))
            native_ok=native_margin>=-1e-7""")
s = s.replace('success_geometry)', 'success_geometry and native_ok)')
s = s.replace('runtime_success=row[\'success\'],terminal_geometry_passed=success_geometry,', "runtime_success=row['success'],terminal_geometry_passed=success_geometry,native_position_passed=native_ok,minimum_native_joint_margin_rad=native_margin,")
s = s.replace("scope='50 Hz sampled geometry/torques;", "scope='50 Hz sampled geometry/torques/native positions; runtime checks positions every physics step;")
s = s.replace("result=dict(integrity_passed=", "result=dict(audit_source_sha256=sha(Path(__file__)),integrity_passed=")
s = s.replace("(a.run/'independent_audit.json').write_text", "(a.run/'independent_audit_source.py').write_bytes(Path(__file__).read_bytes())\n    (a.run/'independent_audit.json').write_text")
p = root / 'tools/audit_contact_load_transfer.py'
assert not p.exists()
p.write_text(s, encoding='utf-8')
