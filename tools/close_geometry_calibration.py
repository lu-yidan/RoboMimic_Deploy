"""Close the declared fixed-residual calibration without selecting failed runs."""
import json
from pathlib import Path
import mujoco
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha


def main():
    root=Path('outputs/geometric_swing_amplitude_calibration_v1')
    rows=[]
    for name in ('geometric_swing_residual_plus10_v2','geometric_swing_half_plus10_v1','geometric_swing_threequarter_plus10_v1'):
        p=Path('outputs')/name
        m=json.loads((p/'manifest.json').read_text());s=json.loads((p/'summary.json').read_text());a=json.loads((p/'independent_audit.json').read_text())
        assert a['integrity_passed'] and a['summary_sha256']==sha(p/'summary.json') and a['trajectory_sha256']==sha(p/'trajectory.npz')
        pm=json.loads((Path(m['parent'])/'manifest.json').read_text())
        env=FullCycleResidualEnv(plan=pm['reference'],cases={m['case']:m['offsets']})
        env.reset(options={'case':m['case']});e=env.env;d=mujoco.MjData(e.model)
        with np.load(p/'trajectory.npz') as z:d.qpos[:]=z['qpos'][-1]
        mujoco.mj_kinematics(e.model,d);mujoco.mj_collision(e.model,d)
        phase=env.reference(s['time_s'])[3]
        contacts=[dict(geoms=[e.model.geom(c.geom1).name,e.model.geom(c.geom2).name],distance_m=float(c.dist)) for c in d.contact]
        rows.append(dict(output=str(p),amplitude=m.get('amplitude',1.),accepted=a['independently_accepted'],reason=s['reason'],time_s=s['time_s'],phase=phase,peaks=s['peaks'],terminal_contact_geometry=contacts,audit_sha256=sha(p/'independent_audit.json')))
        env.close()
    accepted=[r for r in rows if r['accepted']]
    result=dict(results=rows,selected=min(accepted,key=lambda r:r['amplitude'])['output'] if accepted else None,
                decision='select_success' if accepted else 'reject_all',protocol_sha256=sha(root/'protocol.json'),source_sha256=sha(__file__),
                scope='Known-case calibration. Terminal contact geometry is reconstructed; forces remain runtime logs. Integrity is not task success.')
    (root/'selection.json').write_text(json.dumps(result,indent=2));(root/'selection_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
