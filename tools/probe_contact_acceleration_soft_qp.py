"""Inspect a new acceleration/force QP on recorded bottleneck states only."""
import json
from pathlib import Path
import mujoco
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from training.contact_acceleration_soft_probe import solve_probe
from tools.continue_vector_full_cycle_training import sha


def main():
    out=Path('outputs/contact_acceleration_qp_soft_probe_v1');out.mkdir(exist_ok=False);rows=[];values={}
    for label,run in [('original','highrod_height_mismatch_v1'),('support','support_retarget_calibration_v1')]:
        p=Path('outputs')/run/'true_height_12.5mm';meta=json.loads((p/'manifest.json').read_text());audit=json.loads((p/'independent_audit.json').read_text())
        assert audit['integrity_passed'] and audit['trajectory_sha256']==sha(p/'trajectory.npz')
        env=FullCycleResidualEnv(plan=meta['reference'],cases={meta['case']:meta['offsets']});env.reset(options={'case':meta['case']});e=env.env;d=mujoco.MjData(e.model)
        try:
            with np.load(p/'trajectory.npz') as z:
                for request in (59.,59.4,59.8,60.,60.2):
                    i=int(np.argmin(abs(z['time_s']-request)));t=float(z['time_s'][i]);qref,vref,active,phase=env.reference(t);assert phase=='lf_unload'
                    d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];mujoco.mj_forward(e.model,d)
                    bodies=[env.contact.ik.bodies[key] for key,on in zip(('lf','rf','lh','rh'),active) if on]
                    r=solve_probe(e,d,qref,vref,bodies);key=f'{label}_{request:g}';record=dict(label=label,time_s=t,trajectory_sha256=sha(p/'trajectory.npz'))
                    for name,value in r.items():
                        if isinstance(value,np.ndarray):values[key+'_'+name]=value
                        else:record[name]=value
                    rows.append(record);print(json.dumps(record),flush=True)
        finally:env.close()
    np.savez_compressed(out/'solutions.npz',**values)
    sources={}
    for file in (__file__,'training/contact_acceleration_soft_probe.py'):
        file=Path(file);dest=out/file.name;dest.write_bytes(file.read_bytes());sources[str(file)]=sha(file)
    result=dict(rows=rows,sources=sources,solutions_sha256=sha(out/'solutions.npz'),scope='Offline oracle-contact soft sticking-QP feasibility (hard normal separation acceleration, unchanged force/motor bounds; no force-separation complementarity) on ten previously recorded states. No runtime controller change, no dynamic success or training claim. Friction and motor limits retained; current bias-memory estimation not reproduced.')
    (out/'summary.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
