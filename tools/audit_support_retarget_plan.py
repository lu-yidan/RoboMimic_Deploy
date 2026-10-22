"""Independently reconstruct target shifts and reference bounds from saved poses."""
import json
from pathlib import Path
import mujoco
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha


def main():
    p=Path('outputs/support_retarget_plan_v1');m=json.loads((p/'manifest.json').read_text())
    assert sha(p/'plan.npz')==m['plan_sha256'] and sha(m['parent'])==m['parent_sha256']
    base=FullCycleResidualEnv(plan=m['parent']);e=base.env;d=mujoco.MjData(e.model)
    bodies=[e.model.body(n).id for n in ('hook','right_hook','left_ankle_roll_link','right_ankle_roll_link')]
    with np.load(p/'plan.npz') as z:plan={k:z[k].copy() for k in z.files}
    maximum=dict(position_m=0.,rotation_rad=0.,joint_delta_rad=0.,base_shift_m=0.,outside_pose_delta=0.)
    try:
        for i,t in enumerate(plan['time']):
            q0,_,active,phase=base.reference(float(t));q=plan['qpos'][i]
            # Stored knot labels describe the interval ending at the knot;
            # evaluate phase and masks at interval midpoints separately below.
            u=float(np.clip((t-57.)/2.,0,1));v=float(np.clip((65.-t)/2.,0,1));gain=min(u*u*(3-2*u),v*v*(3-2*v))
            d.qpos[:]=q0;mujoco.mj_kinematics(e.model,d);positions=d.xpos[bodies].copy();rotations=d.xmat[bodies].copy();positions[3,2]+=.01*gain
            d.qpos[:]=q;mujoco.mj_kinematics(e.model,d)
            maximum['position_m']=max(maximum['position_m'],float(np.linalg.norm(d.xpos[bodies]-positions,axis=1).max()))
            for R,S in zip(rotations.reshape(-1,3,3),d.xmat[bodies].reshape(-1,3,3)):
                angle=np.arccos(np.clip((np.trace(R.T@S)-1)/2,-1,1));maximum['rotation_rad']=max(maximum['rotation_rad'],float(angle))
            maximum['joint_delta_rad']=max(maximum['joint_delta_rad'],float(abs(q[e.qadr]-q0[e.qadr]).max()))
            maximum['base_shift_m']=max(maximum['base_shift_m'],float(np.linalg.norm(q[:3]-q0[:3])))
            np.testing.assert_array_equal(q[3:7],q0[3:7])
            if t<=57. or t>=65.:maximum['outside_pose_delta']=max(maximum['outside_pose_delta'],float(abs(q-q0).max()))
            if i:
                _,_,aa,pp=base.reference(float((plan['time'][i-1]+t)/2))
                np.testing.assert_array_equal(plan['active'][i],aa);assert str(plan['phase'][i])==pp
        passed=maximum['position_m']<1e-5 and maximum['rotation_rad']<1e-4 and maximum['joint_delta_rad']<=.15+1e-10 and maximum['base_shift_m']<=.03+1e-10 and maximum['outside_pose_delta']<1e-12
        np.testing.assert_array_equal(plan['qpos'][0],base.plan['qpos'][0]);np.testing.assert_array_equal(plan['qpos'][-1],base.plan['qpos'][-1]);assert plan['time'][-1]==base.horizon
        result=dict(passed=bool(passed),maxima=maximum,plan_sha256=sha(p/'plan.npz'),manifest_sha256=sha(p/'manifest.json'),dense_audit_sha256=sha(p/'audit.json'),source_sha256=sha(__file__),scope='Independent saved-pose retarget checks, not physical execution.')
        (p/'independent_audit.json').write_text(json.dumps(result,indent=2));(p/'independent_audit_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result,indent=2))
        if not passed:raise SystemExit(1)
    finally:base.close()


if __name__=='__main__':main()
