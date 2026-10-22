"""Offline fixed-material-point acceleration under the controller PD target.

This tests compatibility with an ideal stationary contact, not actual force or
MuJoCo's compliant-contact feasibility. Never modifies an executing rollout.
"""
import json
from pathlib import Path
import mujoco
import numpy as np
from tools.continue_vector_full_cycle_training import sha
from training.strict_full_cycle_residual_env import FullCycleResidualEnv


def main():
    output=Path('outputs/contact_acceleration_diagnostic_v1');output.mkdir(exist_ok=False);rows=[];arrays={}
    for label,root in [('original','highrod_height_mismatch_v1'),('support','support_retarget_calibration_v1')]:
        p=Path('outputs')/root/'true_height_12.5mm';meta=json.loads((p/'manifest.json').read_text());audit=json.loads((p/'independent_audit.json').read_text())
        assert audit['integrity_passed'] and audit['trajectory_sha256']==sha(p/'trajectory.npz')
        env=FullCycleResidualEnv(plan=meta['reference'],cases={meta['case']:meta['offsets']});env.reset(options={'case':meta['case']});e=env.env;m=e.model
        d=mujoco.MjData(m);plus=mujoco.MjData(m);minus=mujoco.MjData(m);body=m.body('right_ankle_roll_link').id;records=[];convergence=[]
        try:
            with np.load(p/'trajectory.npz') as z:
                ids=np.flatnonzero((z['time_s']>=59)&(z['time_s']<=60.2))
                for i in ids:
                    t=float(z['time_s'][i]);q=z['qpos'][i];v=z['qvel'][i];qref,vref,active,phase=env.reference(t)
                    assert phase=='lf_unload' and active[1] and not np.any(z['actions'][i])
                    d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_collision(m,d)
                    desired=np.zeros(m.nv);desired[:3]=80*(qref[:3]-q[:3])+18*(vref[:3]-v[:3])
                    rotation=np.zeros(3);mujoco.mju_quat2Vel(rotation,q[3:7],1.);desired[3:6]=-80*rotation-18*v[3:6]
                    desired[e.vadr]=100*(qref[e.qadr]-q[e.qadr])+20*(vref[e.vadr]-v[e.vadr])
                    for c in d.contact:
                        a,b=int(c.geom1),int(c.geom2);ba,bb=int(m.geom_bodyid[a]),int(m.geom_bodyid[b])
                        if not ((ba==body and m.geom(b).name=='rung_01') or (bb==body and m.geom(a).name=='rung_01')):continue
                        local=d.xmat[body].reshape(3,3).T@(c.pos-d.xpos[body]);J=np.zeros((3,m.nv));mujoco.mj_jac(m,d,J,None,c.pos,body)
                        derivatives=[]
                        for epsilon in (1e-5,5e-6):
                            jac=[]
                            for scratch,sign in ((plus,1),(minus,-1)):
                                scratch.qpos[:]=q;mujoco.mj_integratePos(m,scratch.qpos,v,sign*epsilon);mujoco.mj_kinematics(m,scratch);mujoco.mj_comPos(m,scratch)
                                point=scratch.xpos[body]+scratch.xmat[body].reshape(3,3)@local;value=np.zeros((3,m.nv));mujoco.mj_jac(m,scratch,value,None,point,body);jac.append(value)
                            derivatives.append((jac[0]-jac[1])@v/(2*epsilon))
                        convergence.append(float(np.linalg.norm(derivatives[0]-derivatives[1])))
                        acceleration=J@desired+derivatives[-1];normal=c.frame[:3]*(-1 if ba==body else 1)
                        records.append([t,float(np.linalg.norm(acceleration)),float(normal@acceleration),float(np.linalg.norm(J@v))])
            a=np.asarray(records);assert len(a)>0 and max(convergence)<1e-6
            arrays[label]=a;rows.append(dict(label=label,contact_samples=len(a),desired_contact_acceleration_rms_m_s2=float(np.sqrt(np.mean(a[:,1]**2))),normal_acceleration_min_m_s2=float(a[:,2].min()),normal_acceleration_max_m_s2=float(a[:,2].max()),point_speed_max_m_s=float(a[:,3].max()),finite_difference_convergence_m_s2=max(convergence),trajectory_sha256=sha(p/'trajectory.npz'),audit_sha256=sha(p/'independent_audit.json')))
        finally:env.close()
    np.savez_compressed(output/'diagnostic.npz',**arrays)
    result=dict(rows=rows,source_sha256=sha(__file__),data_sha256=sha(output/'diagnostic.npz'),columns=['time_s','desired_acceleration_norm_m_s2','desired_normal_acceleration_m_s2','material_point_speed_m_s'],scope=__doc__)
    (output/'summary.json').write_text(json.dumps(result,indent=2));(output/'diagnostic_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result,indent=2))


if __name__=='__main__':main()
