"""Reconstruct saved reference tracking near the audited native-limit failure."""
import json
from pathlib import Path
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha


def main():
    root=Path('outputs/highrod_height_mismatch_v1'); rows=[]
    for h in (5.,7.5,10.,12.5,15.):
        p=root/f'true_height_{h:g}mm'; m=json.loads((p/'manifest.json').read_text())
        a=json.loads((p/'independent_audit.json').read_text())
        assert a['integrity_passed'] and a['trajectory_sha256']==sha(p/'trajectory.npz')
        env=FullCycleResidualEnv(plan=m['reference'],cases={m['case']:m['offsets']})
        env.reset(options={'case':m['case']});e=env.env
        with np.load(p/'trajectory.npz') as z:
            q=z['qpos'];v=z['qvel'];t=z['time_s'];tau=z['torques']
        for time in (52.2,57.,59.,60.,min(61.,float(t[-1]))):
            i=min(int(np.searchsorted(t,time)),len(t)-1); ref,vel,active,phase=env.reference(float(t[i]))
            k=[e.model.joint(j).name for j in e.joints].index('right_ankle_pitch_joint')
            qa,va=e.qadr[k],e.vadr[k];err=q[i,e.qadr]-ref[e.qadr]
            indices=np.argsort(abs(err))[-5:][::-1]
            rows.append(dict(height_mm=h,time_s=float(t[i]),phase=phase,active=active.tolist(),
                right_ankle=dict(actual_rad=float(q[i,qa]),reference_rad=float(ref[qa]),
                    reference_lower_margin_rad=float(ref[qa]-e.model.jnt_range[e.joints[k],0]),
                    actual_lower_margin_rad=float(q[i,qa]-e.model.jnt_range[e.joints[k],0]),
                    velocity_rad_s=float(v[i,va]),reference_velocity_rad_s=float(vel[va]),
                    pd_acceleration_rad_s2=float(100*(ref[qa]-q[i,qa])+20*(vel[va]-v[i,va]))),
                base_error_m=(q[i,:3]-ref[:3]).tolist(),
                largest_tracking_errors=[dict(joint=e.model.joint(e.joints[j]).name,error_rad=float(err[j]),
                    torque_nm=float(tau[max(0,i-1),j]),torque_limit_nm=e.limits[j].tolist()) for j in indices]))
        env.close()
    result=dict(rows=rows,trajectory_hashes={str(p):sha(p) for p in root.glob('true_height_*/trajectory.npz')},
        source_sha256=sha(__file__),scope='50Hz saved-state reconstruction; PD target acceleration is not realized acceleration. No counterfactual contact force reconstruction.')
    (root/'tracking_diagnostic.json').write_text(json.dumps(result,indent=2))
    (root/'tracking_diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps([r for r in rows if r['time_s']>=60],indent=2))


if __name__=='__main__':main()
