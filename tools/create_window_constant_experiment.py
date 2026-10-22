"""Create isolated full-cycle constant residual experiment; never alter predecessors."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT/'deploy_mujoco/run_window_memory_cycle.py').read_text(encoding='utf-8')
source = source.replace('deploy_mujoco/run_window_memory_cycle.py', 'deploy_mujoco/run_window_constant_cycle.py')
source = source.replace("    contact_gate_events=[]", "    residual=np.zeros(6)\n    residual_records=[]\n    contact_gate_events=[]")
needle = "        qref,vref,active,phase=reference(float(d.time))"
replacement = needle + '''
        # Same action/filter as isolated load training; smoothly decay after window.
        old_residual=residual.copy()
        action=np.array([-.25,0.,0.,0.,0.,0.]) if 49.<=d.time<61.56 else np.zeros(6)
        residual+=.05*(.015*action-residual)
        qref[env.qadr[6:12]]+=residual
        vref[env.vadr[6:12]]+=(residual-old_residual)/.005
        residual_records.append(dict(time_s=float(d.time),action=action.tolist(),offset_rad=residual.tolist()))'''
assert source.count(needle)==1
source=source.replace(needle,replacement)
needle="        np.savez_compressed(out/'online_reference.npz'"
assert needle in source
# Residual record saved even on allocation failure; qref records retain applied targets.
needle="    (out/'contact_log.json').write_text"
assert needle in source
source=source.replace(needle,"    (out/'constant_residual_records.json').write_text(json.dumps(residual_records))\n    np.savez_compressed(out/'constant_online_reference.npz',qpos=online_qrefs,qvel=online_vrefs,active=online_active,time_s=online_times)\n"+needle)
source=source.replace('        if landing:\n            online_qrefs.append', '        if True:\n            online_qrefs.append')
target=ROOT/'deploy_mujoco/run_window_constant_cycle.py'
assert not target.exists();target.write_text(source,encoding='utf-8')

driver=(ROOT/'tools/run_window_memory_experiment.py').read_text(encoding='utf-8')
driver=driver.replace('deploy_mujoco.run_window_memory_cycle','deploy_mujoco.run_window_constant_cycle')
driver=driver.replace("name=('memory__' if memory else 'raw__')+case", "name='constant__'+case")
driver=driver.replace('memory_options=[False,True]', 'memory_options=[False],constant_action=[-.25,0,0,0,0,0],residual_scale_rad=.015,filter_alpha=.05,filter_dt_s=.005')
driver=driver.replace('for memory in [False,True]', 'for memory in [False]')
driver=driver.replace('Not full-cycle sensor-bias robustness; single-cycle nominal reference only.', 'Constant action selected on one known hardcase, ramps to zero after61.56s via same filter; not learned. Not full-cycle sensor-bias robustness; single-cycle nominal reference only.')
target=ROOT/'tools/run_window_constant_experiment.py'
assert not target.exists();target.write_text(driver,encoding='utf-8')
print('Created isolated runner and three-case driver')
