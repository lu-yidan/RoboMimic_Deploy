"""Rebuild physical constraints and finite-difference material accelerations."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('experiment',type=Path);args=parser.parse_args();p=args.experiment;summary=json.loads((p/'summary.json').read_text());rows=[]
    assert summary['solutions_sha256']==sha(p/'solutions.npz')
    with np.load(p/'solutions.npz') as solutions:
        for row in summary['rows']:
            if not row['passed']:continue
            run='highrod_height_mismatch_v1' if row['label']=='original' else 'support_retarget_calibration_v1'
            source=Path('outputs')/run/'true_height_12.5mm';meta=json.loads((source/'manifest.json').read_text())
            assert sha(source/'trajectory.npz')==row['trajectory_sha256']
            env=FullCycleResidualEnv(plan=meta['reference'],cases={meta['case']:meta['offsets']});env.reset(options={'case':meta['case']});e=env.env;m=e.model;d=mujoco.MjData(m)
            try:
                with np.load(source/'trajectory.npz') as z:
                    i=int(np.argmin(abs(z['time_s']-row['time_s'])));q=z['qpos'][i].copy();v=z['qvel'][i].copy()
                key=f"{row['label']}_{row['time_s']:.6g}";a=solutions[key+'_acceleration'];forces=solutions[key+'_force'].reshape(-1,3);torque=solutions[key+'_torque']
                d.qpos[:]=q;d.qvel[:]=v;mujoco.mj_forward(m,d);qref,vref,active,_=env.reference(row['time_s'])
                bodies={env.contact.ik.bodies[k] for k,on in zip(('lf','rf','lh','rh'),active) if on}
                mass=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,mass,d.qM);required=mass@a+d.qfrc_bias
                required[e.vadr]+=.3*np.tanh((vref[e.vadr]+5*(qref[e.qadr]-q[e.qadr]))/.02)
                normal_min=np.inf;force_violation=0.;difference=0.;velocity=[];accelerations=[];k=0
                for c in d.contact:
                    ba,bb=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
                    if ba in bodies and bb==0:body,other,sign=ba,int(c.geom2),-1.
                    elif bb in bodies and ba==0:body,other,sign=bb,int(c.geom1),1.
                    else:continue
                    name=m.geom(other).name or ''
                    if not(name.startswith('rung_') and len(name)==7 and name[-2:].isdigit()):continue
                    frame=sign*c.frame.reshape(3,3);f=forces[k];k+=1;normal=float(frame[0]@f);mu=min(.7,float(c.friction[0]),float(c.friction[1]))
                    force_violation=max(force_violation,-normal,normal-350.)
                    for theta in np.arange(8)*np.pi/4+np.pi/8:force_violation=max(force_violation,float((np.cos(theta)*frame[1]+np.sin(theta)*frame[2])@f-mu*np.cos(np.pi/8)*normal))
                    J=np.zeros((3,m.nv));mujoco.mj_jac(m,d,J,None,c.pos,body);required-=J.T@f;speed=J@v
                    local=d.xmat[body].reshape(3,3).T@(c.pos-d.xpos[body]);position0=d.xpos[body]+d.xmat[body].reshape(3,3)@local;second=[]
                    for epsilon in (2e-4,1e-4):
                        positions=[]
                        for sign_step in (1,-1):
                            scratch=mujoco.MjData(m);scratch.qpos[:]=q
                            mujoco.mj_integratePos(m,scratch.qpos,v+.5*sign_step*epsilon*a,sign_step*epsilon);mujoco.mj_kinematics(m,scratch)
                            positions.append(scratch.xpos[body]+scratch.xmat[body].reshape(3,3)@local)
                        second.append((positions[0]-2*position0+positions[1])/epsilon**2)
                    difference=max(difference,float(np.linalg.norm(second[0]-second[1])));acceleration=second[-1]
                    accelerations.extend(acceleration);velocity.extend(speed);normal_min=min(normal_min,float(frame[0]@(acceleration+20*speed)))
                assert k==len(forces)
                required[e.vadr]-=torque
                accel_error=float(np.max(abs(np.asarray(accelerations)-solutions[key+'_solved_contact_acceleration'])))
                checks=dict(full_dynamics=float(abs(required).max())<1e-3,motor_bounds=bool(np.all(torque>=e.limits[:,0]-1e-6) and np.all(torque<=e.limits[:,1]+1e-6)),force_cones=force_violation<1e-3,normal_separation=normal_min>=-1e-4,acceleration_reconstruction=accel_error<1e-4,finite_difference_convergence=difference<1e-4)
                rows.append(dict(label=row['label'],time_s=row['time_s'],passed=all(checks.values()),checks=checks,max_dynamics_residual=float(abs(required).max()),acceleration_reconstruction_error=accel_error,finite_difference_convergence=difference,normal_separation_min=normal_min,force_cone_violation=force_violation))
            finally:env.close()
    result=dict(passed=bool(rows) and all(r['passed'] for r in rows),rows=rows,summary_sha256=sha(p/'summary.json'),source_sha256=sha(__file__),scope='Independent full dynamics, torque/friction limits and contact acceleration reconstructed from position second differences. Offline feasibility only; no force-separation complementarity or closed-loop claim.')
    (p/'independent_audit.json').write_text(json.dumps(result,indent=2));(p/'audit_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
