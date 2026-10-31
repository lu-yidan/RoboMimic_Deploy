"""Experimental hook admission for later cycles; original first cycle unchanged.

The copied fixed-contact QP changes only its world-vertical force upper bounds.
These are estimated support forces, not measured surface-normal force limits.
Native motor limits,
equilibrium equalities, friction cones and solver acceptance stay unchanged.
This module is not installed by the adopted baseline.
"""
import hashlib
import inspect
import textwrap
import mujoco
import numpy as np
from scipy import sparse
import osqp
from training.climb_cycle import CycleTorqueController,solve_force_qp_fallback

ORIGINAL_COMMAND_SHA256 = '1216c24c48b295f6ff50b560bc0088de617d32a159607e2321211847ff7bb0fd'

def capped_command(self,qref,vref,active,normal_caps):
    normal_caps=np.asarray(normal_caps,dtype=float)
    if normal_caps.shape!=(4,) or not np.isfinite(normal_caps).all() or np.any(normal_caps<0) or np.any(normal_caps>350):
        raise ValueError("Expected four per-contact normal caps within 0..350 N")
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
    jacobians=[];goals=[]
    for index,key in enumerate(['lf','rf','lh','rh']):
        if not active[index]:continue
        body=self.ik.bodies[key]
        for dy in [-.025,.025] if index<2 else [-.009,.009]:
            local=np.array([.09,0,-.035] if index<2 else [0,0,.022])
            point=d.xpos[body]+d.xmat[body].reshape(3,3)@local+np.array([0,dy,0])
            J=np.zeros((3,m.nv));mujoco.mj_jac(m,d,J,None,point,body)
            jacobians.append(J.T);goals.extend([0,0,65./2 if index>=2 else (e.robot_weight_n-65.*sum(active[2:]))/(2*sum(active[:2]))])
    J=np.hstack(jacobians);n=J.shape[1];count=n//3
    cone=np.zeros((8*count,n));normal=np.zeros((count,n))
    for i in range(count):
        normal[i,3*i+2]=1.
        for k,angle in enumerate(np.arange(8)*np.pi/4+np.pi/8):
            cone[8*i+k,3*i:3*i+3]=[np.cos(angle),np.sin(angle),-.7*np.cos(np.pi/8)]
    A=np.vstack([J[:6],J[e.vadr],normal,cone]);nr=len(A)
    lower=np.r_[required[:6],required[e.vadr]-e.limits[:,1],np.zeros(count),np.full(8*count,-np.inf)]
    upper=np.r_[required[:6],required[e.vadr]-e.limits[:,0],np.repeat(normal_caps[np.asarray(active,dtype=bool)],2),np.zeros(8*count)]
    key=tuple(active);values=A.ravel(order='F')
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
    return torque,dict(base_equilibrium_residual=residual,force_plan=f.tolist(),torque_ratio=float(np.max(abs(torque)/e.limits[:,1])),solver_retries=self.solver_retries,solver_fallbacks=self.solver_fallbacks)


def install_hook_admission(base,probe_force_n=15.,confirm_force_n=5.,confirm_seconds=.05,ramp_seconds=.5,timeout_seconds=1.5):
    if not (0<confirm_force_n<probe_force_n<100 and .01<=confirm_seconds<=.2 and .1<=ramp_seconds<=1. and confirm_seconds+ramp_seconds<timeout_seconds<=2.):
        raise ValueError('Invalid bounded probe configuration')
    source=textwrap.dedent(inspect.getsource(CycleTorqueController.command))
    if hashlib.sha256(source.encode()).hexdigest()!=ORIGINAL_COMMAND_SHA256:
        raise ValueError('Original QP changed; copied capped-QP must be reviewed')
    original=base.baseline.command
    state=dict(previous=np.ones(4,dtype=bool),hooks={},trace=[])
    labels={2:'left_hook',3:'right_hook'}
    def measured(group,rod):
        e=base.env;metrics=e.contact_metrics();normal=0.
        for ci,c in enumerate(e.data.contact):
            ba,bb=[int(e.model.geom_bodyid[g]) for g in (c.geom1,c.geom2)]
            if (ba in e.robot_bodies)==(bb in e.robot_bodies):continue
            body=ba if ba in e.robot_bodies else bb
            if e.contact_groups.get(body)==group:
                wrench=np.zeros(6);mujoco.mj_contactForce(e.model,e.data,ci,wrench);normal+=max(0.,float(wrench[0]))
        loaded=[c for c in metrics['contacts'] if c['group']==group and np.linalg.norm(c['force_on_robot_world_n'])>1.]
        return dict(normal_n=normal,vertical_n=float(metrics[group+'_vertical_n']),
            intended=bool(loaded) and all(c['external_geom']==rod for c in loaded))
    def command(qref,vref,active):
        t=float(base.env.data.time);current=min(int(np.searchsorted(base.ends,t,side='right')),len(base.ends)-1)
        active=np.asarray(active,dtype=bool);caps=np.full(4,350.);measurements={};phases={}
        if current==0:
            state['previous']=active.copy()
            return original(qref,vref,active)
        for index,group in labels.items():
            if not active[index]:
                state['hooks'].pop(index,None)
                continue
            if not state['previous'][index]:
                state['hooks'][index]=dict(start=t,stable_start=None,confirmed=None)
            hook=state['hooks'].get(index)
            if hook is None:continue
            measurement=measured(group,f'rung_{current+5:02d}');measurements[group]=measurement
            if hook['confirmed'] is None:
                ok=measurement['intended'] and min(measurement['normal_n'],measurement['vertical_n'])>confirm_force_n
                if ok:
                    if hook['stable_start'] is None:hook['stable_start']=t
                    if t-hook['stable_start']>=confirm_seconds-1e-8:hook['confirmed']=t
                else:hook['stable_start']=None
                if hook['confirmed'] is None and t-hook['start']>timeout_seconds:
                    raise RuntimeError('Hook contact was not confirmed within probe timeout: '+group)
            if hook['confirmed'] is None:
                caps[index]=probe_force_n/2.;phases[group]='probe'
            else:
                u=float(np.clip((t-hook['confirmed'])/ramp_seconds,0,1));weight=u*u*(3-2*u)
                caps[index]=(probe_force_n+(700.-probe_force_n)*weight)/2.
                phases[group]='ramp' if u<1 else 'admitted'
        state['previous']=active.copy()
        limited=bool(np.any(caps[np.asarray(active)]<350.-1e-10))
        result=capped_command(base.baseline,qref,vref,active,caps) if limited else original(qref,vref,active)
        torque,details=result
        if limited:
            f=np.array(details['force_plan']).reshape(-1,3);limits=np.repeat(caps[active],2)
            if np.max(f[:,2]-limits)>1e-3:raise RuntimeError('Probe QP exceeded declared normal cap')
            state['trace'].append(dict(time_s=t,cycle=current+1,phase=base.last_phase,active=active.tolist(),
                caps_per_contact_n=caps.tolist(),measurements=measurements,states=phases,
                force_plan=f.tolist(),torque_nm=torque.tolist(),base_equilibrium_residual=details['base_equilibrium_residual'],
                hooks={labels[k]:dict(v) for k,v in state['hooks'].items()}))
        return result
    base.baseline.command=command
    return state
