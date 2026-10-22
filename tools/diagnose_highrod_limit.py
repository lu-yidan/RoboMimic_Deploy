"""Identify native joint-limit failures from audited mismatch trajectories."""
import json
from pathlib import Path
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha


def main():
    root=Path('outputs/highrod_height_mismatch_v1');rows=[]
    for r in json.loads((root/'results.json').read_text()):
        if r['reason']!='native_joint_position_violation':continue
        p=root/r['label'];m=json.loads((p/'manifest.json').read_text());a=json.loads((p/'independent_audit.json').read_text())
        assert a['integrity_passed'] and a['trajectory_sha256']==sha(p/'trajectory.npz')
        env=FullCycleResidualEnv(plan=m['reference'],cases={m['case']:m['offsets']});env.reset(options={'case':m['case']});e=env.env
        with np.load(p/'trajectory.npz') as z:
            q=z['qpos'][-1,e.qadr];lower=q-e.model.jnt_range[e.joints,0];upper=e.model.jnt_range[e.joints,1]-q
            k=int(np.argmin(np.minimum(lower,upper)));j=e.joints[k];lower_side=lower[k]<upper[k]
            row=dict(case=m['case'],time_s=r['time_s'],phase=env.reference(r['time_s'])[3],joint=e.model.joint(j).name,
                bound='lower' if lower_side else 'upper',position_rad=float(q[k]),limit_rad=float(e.model.jnt_range[j,0 if lower_side else 1]),
                velocity_rad_s=float(z['qvel'][-1,e.vadr[k]]),violation_rad=float(-min(lower[k],upper[k])),
                recorded_torque_nm=float(z['torques'][-1,k]),motor_range_nm=e.limits[k].tolist(),trajectory_sha256=sha(p/'trajectory.npz'))
            rows.append(row)
        env.close()
    result=dict(failures=rows,source_sha256=sha(__file__),scope='Terminal saved-state diagnosis. Motor torque is last controller command; no counterfactual recovery or hardware claim.')
    (root/'native_limit_diagnostic.json').write_text(json.dumps(result,indent=2));(root/'native_limit_diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
