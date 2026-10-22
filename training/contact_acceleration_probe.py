"""Offline whole-body allocation candidate; not installed in any runtime controller."""
import mujoco
import numpy as np
import osqp
from scipy import sparse


def material_jacobian_rate(model,data,body,point,epsilon=1e-5):
    local=data.xmat[body].reshape(3,3).T@(point-data.xpos[body]);values=[]
    for sign in (1,-1):
        scratch=mujoco.MjData(model);scratch.qpos[:]=data.qpos
        mujoco.mj_integratePos(model,scratch.qpos,data.qvel,sign*epsilon)
        mujoco.mj_kinematics(model,scratch);mujoco.mj_comPos(model,scratch)
        moved=scratch.xpos[body]+scratch.xmat[body].reshape(3,3)@local
        jacobian=np.zeros((3,model.nv));mujoco.mj_jac(model,scratch,jacobian,None,moved,body);values.append(jacobian)
    return (values[0]-values[1])@data.qvel/(2*epsilon)


def solve_probe(env,data,qref,vref,bodies,contact_damping=20.):
    """Jointly choose acceleration and force, enforcing ideal sticking contacts.

    Actual simulated contact geometry only. Hard sticking is a modelling
    hypothesis; this output is never applied to MuJoCo by this module.
    """
    m=env.model;nv=m.nv;mass=np.zeros((nv,nv));mujoco.mj_fullM(m,mass,data.qM)
    desired=np.zeros(nv);desired[:3]=80*(qref[:3]-data.qpos[:3])+18*(vref[:3]-data.qvel[:3])
    rot=np.zeros(3);mujoco.mju_quat2Vel(rot,data.qpos[3:7],1.);desired[3:6]=-80*rot-18*data.qvel[3:6]
    desired[env.vadr]=100*(qref[env.qadr]-data.qpos[env.qadr])+20*(vref[env.vadr]-data.qvel[env.vadr])
    bias=data.qfrc_bias.copy();bias[env.vadr]+=.3*np.tanh((vref[env.vadr]+5*(qref[env.qadr]-data.qpos[env.qadr]))/.02)
    jacobians=[];frames=[];frictions=[];rates=[];groups=[]
    for c in data.contact:
        a,b=int(c.geom1),int(c.geom2);ba,bb=int(m.geom_bodyid[a]),int(m.geom_bodyid[b])
        if ba in bodies and bb==0:body,other,sign=ba,b,-1.
        elif bb in bodies and ba==0:body,other,sign=bb,a,1.
        else:continue
        name=m.geom(other).name or ''
        if not(name.startswith('rung_') and len(name)==7 and name[-2:].isdigit()):continue
        J=np.zeros((3,nv));mujoco.mj_jac(m,data,J,None,c.pos,body)
        jacobians.append(J);frames.append(sign*c.frame.reshape(3,3));frictions.append(min(.7,float(c.friction[0]),float(c.friction[1])))
        rates.append(material_jacobian_rate(m,data,body,c.pos));groups.append(body)
    if not jacobians or set(groups)!=set(bodies):raise ValueError('Not all scheduled supports have actual contacts')
    J=np.vstack(jacobians);rate=np.concatenate(rates);nf=J.shape[0];n=nv+nf
    weights=np.r_[np.full(6,10.),np.ones(nv-6),np.full(nf,1e-4)]
    P=sparse.diags(weights,format='csc');objective=np.r_[-weights[:nv]*desired,np.zeros(nf)]
    dynamic=np.hstack([mass,-J.T]);contact=np.hstack([J,np.zeros((nf,nf))])
    target=-rate-contact_damping*(J@data.qvel)
    rows=[dynamic[:6],dynamic[env.vadr],contact];lower=[-bias[:6],env.limits[:,0]-bias[env.vadr],target];upper=[-bias[:6],env.limits[:,1]-bias[env.vadr],target]
    for i,(frame,mu) in enumerate(zip(frames,frictions)):
        normal=np.zeros((1,n));normal[0,nv+3*i:nv+3*i+3]=frame[0];rows.append(normal);lower.append(np.zeros(1));upper.append(np.full(1,350.))
        cone=np.zeros((8,n))
        for k,angle in enumerate(np.arange(8)*np.pi/4+np.pi/8):cone[k,nv+3*i:nv+3*i+3]=np.cos(angle)*frame[1]+np.sin(angle)*frame[2]-mu*np.cos(np.pi/8)*frame[0]
        rows.append(cone);lower.append(np.full(8,-np.inf));upper.append(np.zeros(8))
    A=np.vstack(rows);lo=np.concatenate(lower);hi=np.concatenate(upper)
    solver=osqp.OSQP();solver.setup(P=P,q=objective,A=sparse.csc_matrix(A),l=lo,u=hi,verbose=False,eps_abs=1e-6,eps_rel=1e-7,max_iter=50000,polishing=True)
    solution=solver.solve(raise_error=False)
    if solution.info.status_val!=1:return dict(passed=False,status=solution.info.status,contact_count=len(frames))
    x=solution.x;a=x[:nv];f=x[nv:];torque=(dynamic@x+bias)[env.vadr]
    violation=float(max(0.,np.max(lo-A@x),np.max(A@x-hi)))
    return dict(passed=bool(np.isfinite(x).all() and violation<1e-3),status=solution.info.status,contact_count=len(frames),
        max_constraint_residual=violation,desired_acceleration=desired,acceleration=a,force=f,torque=torque,
        desired_contact_acceleration=J@desired+rate,solved_contact_acceleration=J@a+rate,
        contact_velocity=J@data.qvel,acceleration_change_norm=float(np.linalg.norm(a-desired)),
        base_equilibrium_error=float(np.max(abs((dynamic@x+bias)[:6]))))
