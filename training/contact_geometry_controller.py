"""Oracle contact-position allocation experiment; not a deployable sensor model.

Same fixed tracking acceleration, friction cones, force caps and torque limits as
baseline. Only Jacobian application points follow actual MuJoCo rung contacts.
"""
import mujoco
import numpy as np
from scipy import sparse
import osqp
from training.climb_cycle import LimbIK,solve_force_qp_fallback


class ContactGeometryController:
    def __init__(self,env,joint_kp=100.,torque_weight=0.,leg_kp=None):
        self.e=env;self.m=env.model;self.d=env.data;self.cache={}
        self.M=np.zeros((self.m.nv,self.m.nv));self.ik=LimbIK(env)
        self.joint_kp=float(joint_kp);self.joint_kd=2*np.sqrt(self.joint_kp)
        self.solver_retries=0
        self.solver_fallbacks=0
        self.torque_weight=float(torque_weight)
        if not np.isfinite(self.torque_weight) or self.torque_weight<0:
            raise ValueError('torque_weight must be finite and nonnegative')
        self.leg_kp=self.joint_kp if leg_kp is None else float(leg_kp)
        if not np.isfinite(self.leg_kp) or self.leg_kp<=0:raise ValueError('leg_kp must be finite and positive')

    def command(self,qref,vref,active):
        e,m,d=self.e,self.m,self.d
        mujoco.mj_forward(m,d)
        mujoco.mj_fullM(m,self.M,d.qM)
        qacc=np.zeros(m.nv)
        qacc[:3]=80*(qref[:3]-d.qpos[:3])+18*(vref[:3]-d.qvel[:3])
        rot=np.zeros(3);mujoco.mju_quat2Vel(rot,d.qpos[3:7],1.)
        qacc[3:6]=-80*rot-18*d.qvel[3:6]
        qacc[e.vadr]=self.joint_kp*(qref[e.qadr]-d.qpos[e.qadr])+self.joint_kd*(vref[e.vadr]-d.qvel[e.vadr])
        leg_q=e.qadr[:12];leg_v=e.vadr[:12]
        qacc[leg_v]=self.leg_kp*(qref[leg_q]-d.qpos[leg_q])+2*np.sqrt(self.leg_kp)*(vref[leg_v]-d.qvel[leg_v])
        required=self.M@qacc+d.qfrc_bias
        # Reuse the research project's bounded Coulomb-friction estimate.
        # This is a fixed 0.3 Nm model assumption, not measured contact torque.
        required[e.vadr] += .3*np.tanh((vref[e.vadr]+5*(qref[e.qadr]-d.qpos[e.qadr]))/.02)
        jacobians=[];goals=[];counts=[]
        for index,key in enumerate(['lf','rf','lh','rh']):
            if not active[index]:continue
            body=self.ik.bodies[key];points=[]
            for contact in d.contact:
                a,b=int(contact.geom1),int(contact.geom2)
                ba,bb=int(m.geom_bodyid[a]),int(m.geom_bodyid[b])
                if ba==body and bb==0:external=b
                elif bb==body and ba==0:external=a
                else:continue
                name=m.geom(external).name
                if name.startswith('rung_') and len(name)==7 and name[-2:].isdigit():
                    points.append(contact.pos.copy())
            if not points:raise RuntimeError('No measured rung contact for scheduled support '+key)
            counts.append(len(points))
            load=65. if index>=2 else (e.robot_weight_n-65.*sum(active[2:]))/sum(active[:2])
            for point in points:
                J=np.zeros((3,m.nv));mujoco.mj_jac(m,d,J,None,point,body)
                jacobians.append(J.T);goals.extend([0,0,load/len(points)])
        J=np.hstack(jacobians);n=J.shape[1];count=n//3
        cone=np.zeros((8*count,n));normal=np.zeros((count,n))
        for i in range(count):
            normal[i,3*i+2]=1.
            for k,angle in enumerate(np.arange(8)*np.pi/4+np.pi/8):
                cone[8*i+k,3*i:3*i+3]=[np.cos(angle),np.sin(angle),-.7*np.cos(np.pi/8)]
        A=np.vstack([J[:6],J[e.vadr],normal,cone]);nr=len(A)
        lower=np.r_[required[:6],required[e.vadr]-e.limits[:,1],np.zeros(count),np.full(8*count,-np.inf)]
        upper=np.r_[required[:6],required[e.vadr]-e.limits[:,0],np.full(count,350.),np.zeros(8*count)]
        key=(tuple(active),tuple(counts));values=A.ravel(order='F')
        objective=-np.asarray(goals)
        if self.torque_weight:
            scale=np.maximum(abs(e.limits[:,0]),abs(e.limits[:,1]))
            B=J[e.vadr]/scale[:,None];b=required[e.vadr]/scale
            H=np.eye(n)+self.torque_weight*(B.T@B)
            objective=objective-self.torque_weight*(B.T@b)
            rr,cc=np.triu_indices(n)
            P=sparse.csc_matrix((H[rr,cc],(rr,cc)),shape=(n,n))
        else:H=np.eye(n);P=sparse.eye(n,format='csc')
        if key not in self.cache:
            solver=osqp.OSQP();matrix=sparse.csc_matrix((values,np.tile(np.arange(nr),n),np.arange(n+1)*nr),shape=(nr,n))
            solver.setup(P=P,q=objective,A=matrix,l=lower,u=upper,
                         verbose=False,eps_abs=1e-5,eps_rel=1e-6,max_iter=5000,polishing=True,warm_starting=True)
            self.cache[key]=solver
        else:
            solver=self.cache[key];solver.update(Ax=values,q=objective,l=lower,u=upper)
            if self.torque_weight:solver.update(Px=P.data)
        solution=solver.solve(raise_error=False)
        if solution.info.status_val in (2,7):
            # Retry the identical QP to the same accuracy; never relax its bounds.
            self.solver_retries+=1
            solver.update_settings(max_iter=50000)
            solution=solver.solve(raise_error=False)
            solver.update_settings(max_iter=5000)
        if solution.info.status_val in (2,7):
            f=solve_force_qp_fallback(H,objective,A,lower,upper)
            self.solver_fallbacks+=1
            solver.warm_start(x=f)
        elif solution.info.status_val!=1:
            raise RuntimeError('Contact force allocation '+solution.info.status)
        else:f=solution.x
        if not np.isfinite(f).all():raise RuntimeError('Nonfinite contact allocation')
        torque=required[e.vadr]-J[e.vadr]@f
        residual=float(np.max(abs(J[:6]@f-required[:6])))
        feasibility=float(max(0.,np.max(lower-A@f),np.max(A@f-upper)))
        if feasibility>1e-3 or residual>1e-3 or np.max(torque-e.limits[:,1])>1e-3 or np.max(e.limits[:,0]-torque)>1e-3:
            raise RuntimeError('Inverse-dynamics residual or motor bound violation')
        # Remove only solver roundoff already verified above; retain hard motor bounds.
        torque=np.clip(torque,e.limits[:,0],e.limits[:,1])
        return torque,dict(contact_point_counts=counts,contact_source="Privileged MuJoCo contact positions",base_equilibrium_residual=residual,force_plan=f.tolist(),torque_ratio=float(np.max(abs(torque)/e.limits[:,1])),solver_retries=self.solver_retries,solver_fallbacks=self.solver_fallbacks)
