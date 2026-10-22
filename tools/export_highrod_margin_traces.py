"""Export audited right-ankle and all-joint margin histories for plotting."""
import json
from pathlib import Path
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha


def main():
    p=Path('outputs/highrod_height_mismatch_v1');rows=[]
    for r in json.loads((p/'results.json').read_text()):
        run=p/r['label'];m=json.loads((run/'manifest.json').read_text());a=json.loads((run/'independent_audit.json').read_text())
        assert a['integrity_passed'] and a['trajectory_sha256']==sha(run/'trajectory.npz')
        env=FullCycleResidualEnv(plan=m['reference']);e=env.env
        joint=e.model.joint('right_ankle_pitch_joint').id;qadr=e.model.jnt_qposadr[joint];vadr=e.model.jnt_dofadr[joint]
        with np.load(run/'trajectory.npz') as z:
            q=z['qpos'];v=z['qvel'];t=z['time_s']
            margins=np.minimum(q[:,e.qadr]-e.model.jnt_range[e.joints,0],e.model.jnt_range[e.joints,1]-q[:,e.qadr]).min(axis=1)
            cost=.01*np.clip((.02-margins[1:])/.02,0,1)**2
            keep=(t>=45)&(t<=65)
            rows.append(dict(true_height_mm=m['offsets']['rung_01'][2]*1000,accepted=a['independently_accepted'],time_s=t[keep].tolist(),
                right_ankle_pitch_rad=q[keep,qadr].tolist(),right_ankle_velocity_rad_s=v[keep,vadr].tolist(),
                minimum_all_joint_margin_rad=margins[keep].tolist(),lower_limit_rad=float(e.model.jnt_range[joint,0]),
                terminal_time_s=r['time_s'],prospective_cost_sum=float(cost.sum()),prospective_cost_nonzero_steps=int((cost>0).sum()),
                trajectory_sha256=sha(run/'trajectory.npz')))
        env.close()
    result=dict(rows=sorted(rows,key=lambda r:r['true_height_mm']),source_sha256=sha(__file__),
        scope='Recorded50Hz state traces, including terminal saved state. Prospective penalty computed offline at action ends; not a trained result. Native safety was checked at2000Hz during original runs.')
    (p/'margin_traces.json').write_text(json.dumps(result,indent=2));print(json.dumps([{k:r[k] for k in ('true_height_mm','prospective_cost_sum','prospective_cost_nonzero_steps')} for r in rows]))


if __name__=='__main__':main()
