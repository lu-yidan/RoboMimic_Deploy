"""Kinematic references and torque-only control for a complete rung cycle.

All reference optimization uses scratch MjData. Live states are never replaced.
The force allocation follows the existing research project's constrained
inverse-dynamics formulation, extended to both hooks and sole contact patches.
"""
import mujoco
import numpy as np
from scipy.optimize import least_squares, linprog, minimize, LinearConstraint
from scipy.interpolate import PchipInterpolator
from scipy import sparse
import osqp

from training.hand_transfer import smooth


class LimbIK:
    def __init__(self, env):
        self.env=env;self.m=env.model;self.d=mujoco.MjData(self.m)
        self.slices={'lf':slice(0,6),'rf':slice(6,12),'lh':slice(15,22),'rh':slice(22,29)}
        self.bodies={k:self.m.body(n).id for k,n in [('lf','left_ankle_roll_link'),('rf','right_ankle_roll_link'),('lh','hook'),('rh','right_hook')]}
        self.leg_pairs={}
        for key,side in [('lf','left'),('rf','right')]:
            geoms=[g for g in range(self.m.ngeom) if self.m.geom_contype[g] and
                   self.m.body(self.m.geom_bodyid[g]).name.startswith(side+'_') and
                   any(n in self.m.body(self.m.geom_bodyid[g]).name for n in ['hip','knee'])]
            self.leg_pairs[key]=[(g,self.m.geom(f'rung_{i:02d}').id) for g in geoms for i in range(5)]

    def contact_leg(self,q,key,point,yaw,pitch_weight=.0005):
        """Constrain sole tangency to a horizontal rod, leaving foot pitch free."""
        sl=self.slices[key];qa=self.env.qadr[sl];js=self.env.joints[sl];bid=self.bodies[key]
        prior=q[qa].copy();local=np.array([.09,0,-.035])
        def fun(x):
            self.d.qpos[:]=q;self.d.qpos[qa]=x;mujoco.mj_kinematics(self.m,self.d)
            R=self.d.xmat[bid].reshape(3,3);normal=R[:,2]
            contact=self.d.xpos[bid]+R@local
            angle=np.arctan2(R[1,0],R[0,0]);angle_error=np.arctan2(np.sin(angle-yaw),np.cos(angle-yaw))
            distances=[mujoco.mj_geomDistance(self.m,self.d,a,b,.013,None) for a,b in self.leg_pairs[key]]
            return np.r_[contact-point-.0148*normal,.2*normal[1],.08*angle_error,
                         .0001*(x-prior),pitch_weight*normal[0],4.*np.maximum(0.,.012-np.array(distances))]
        fit=least_squares(fun,np.clip(prior,self.m.jnt_range[js,0]+1e-8,self.m.jnt_range[js,1]-1e-8),
                          bounds=(self.m.jnt_range[js,0],self.m.jnt_range[js,1]),max_nfev=120,
                          ftol=1e-9,xtol=1e-9,gtol=1e-9)
        residual=fun(fit.x);result=q.copy();result[qa]=fit.x
        return result,float(np.linalg.norm(residual[:5]))

    def pose(self,q,targets,regularization=1e-5,orientations=None):
        q=q.copy();errors={}
        for key,target in targets.items():
            sl=self.slices[key];qa=self.env.qadr[sl];va=self.env.vadr[sl]
            joints=self.env.joints[sl];bid=self.bodies[key];prior=q[qa].copy()
            desired_quat=np.array([1.,0,0,0]) if orientations is None else orientations[key]
            inverse_quat=desired_quat*np.array([1,-1,-1,-1])
            def fun(x):
                self.d.qpos[:]=q;self.d.qpos[qa]=x;mujoco.mj_kinematics(self.m,self.d)
                difference=np.zeros(4);mujoco.mju_mulQuat(difference,self.d.xquat[bid],inverse_quat)
                rot=np.zeros(3);mujoco.mju_quat2Vel(rot,difference,1.)
                return np.r_[self.d.xpos[bid]-target,.2*rot,regularization*(x-prior)]
            def jac(x):
                fun(x);mujoco.mj_comPos(self.m,self.d)
                jp=np.zeros((3,self.m.nv));jr=np.zeros_like(jp)
                mujoco.mj_jacBody(self.m,self.d,jp,jr,bid)
                return np.vstack([jp[:,va],.2*jr[:,va],regularization*np.eye(len(qa))])
            fit=least_squares(fun,np.clip(prior,self.m.jnt_range[joints,0]+1e-8,self.m.jnt_range[joints,1]-1e-8),
                              jac=jac,bounds=(self.m.jnt_range[joints,0],self.m.jnt_range[joints,1]),max_nfev=80,
                              ftol=1e-9,xtol=1e-9,gtol=1e-9)
            residual=fun(fit.x);q[qa]=fit.x
            errors[key]=float(np.linalg.norm(residual[:6]))
        return q,errors

    def collisions(self,q):
        self.d.qpos[:]=q;mujoco.mj_forward(self.m,self.d)
        result=[]
        for c in self.d.contact:
            a,b=int(c.geom1),int(c.geom2);ba,bb=int(self.m.geom_bodyid[a]),int(self.m.geom_bodyid[b])
            ina,inb=ba in self.env.robot_bodies,bb in self.env.robot_bodies
            if ina==inb:continue
            body=ba if ina else bb;other=b if ina else a
            if body not in self.env.contact_groups:
                result.append((self.m.body(body).name,self.m.geom(other).name,float(c.dist)))
        return result


def rung(z):
    return np.array([.6+np.tan(np.deg2rad(5.))*z-.35,0.,z])


def plan_joint_swing(ik,qstart,key,point,yaw,seed=41):
    """Bidirectional joint-space search for the airborne leg; no live state edits."""
    e,m,d=ik.env,ik.m,ik.d;sl=ik.slices[key];qa=e.qadr[sl];js=e.joints[sl]
    rng=np.random.default_rng(seed);low=m.jnt_range[js,0];high=m.jnt_range[js,1]
    side='left' if key=='lf' else 'right'
    bodies={b for b in range(m.nbody) if m.body(b).name.startswith(side+'_') and any(s in m.body(b).name for s in ['hip','knee','ankle'])}
    geoms=[g for g in range(m.ngeom) if m.geom_contype[g] and m.geom_bodyid[g] in bodies]
    bars=[m.geom(f'rung_{i:02d}').id for i in range(5)]
    evaluations=0
    def valid(x):
        nonlocal evaluations
        evaluations+=1;d.qpos[:]=qstart;d.qpos[qa]=x;mujoco.mj_kinematics(m,d)
        if abs(yaw)<np.deg2rad(45):
            R=d.xmat[ik.bodies[key]].reshape(3,3)
            actual_yaw=np.arctan2(R[1,0],R[0,0])
            if abs(actual_yaw)>abs(yaw)+np.deg2rad(20):return False
        for g in geoms:
            for b in bars:
                if mujoco.mj_geomDistance(m,d,g,b,.0015,None)<.0015:return False
        mujoco.mj_collision(m,d)
        for c in d.contact:
            if c.dist<-.0005 and (int(m.geom_bodyid[c.geom1]) in bodies or int(m.geom_bodyid[c.geom2]) in bodies):return False
        return True
    def edge(a,b):
        return all(valid((1-u)*a+u*b) for u in np.linspace(0,1,max(2,int(np.max(abs(b-a))/.018)+2))[1:])
    goal=None
    for attempt in range(48):
        q=qstart.copy()
        if attempt:
            q[qa]=np.clip(q[qa]+rng.normal(0,.9,6),low+1e-5,high-1e-5)
            q[qa[3]]=rng.uniform(.5,2.3)
        candidate,error=ik.contact_leg(q,key,point,yaw)
        if error<.001 and valid(candidate[qa]):goal=candidate[qa].copy();break
    if goal is None:raise RuntimeError('No collision-free swing endpoint for '+key)
    start=qstart[qa].copy()
    if not valid(start):raise RuntimeError('Swing start lacks collision clearance for '+key)
    if edge(start,goal):path=[start,goal]
    else:
        trees=[([start],[-1]),([goal],[-1])];joined=None
        def extend(tree,target):
            nodes,parents=tree;i=int(np.argmin([np.linalg.norm(n-target) for n in nodes]));delta=target-nodes[i]
            norm=np.linalg.norm(delta);new=target.copy() if norm<.18 else nodes[i]+.18*delta/norm
            if not edge(nodes[i],new):return None,False
            nodes.append(new);parents.append(i);return len(nodes)-1,norm<.18
        for iteration in range(5000):
            a=iteration%2;b=1-a
            sample=trees[b][0][rng.integers(len(trees[b][0]))] if rng.random()<.25 else np.clip(
                (1-rng.random())*start+rng.random()*goal+rng.normal(0,.65,6),low,high)
            ia,_=extend(trees[a],sample)
            if ia is None:continue
            for _ in range(60):
                ib,reached=extend(trees[b],trees[a][0][ia])
                if ib is None:break
                if reached:joined=(a,ia,b,ib);break
            if joined:break
        if joined is None:raise RuntimeError('Joint swing search exhausted for '+key)
        def ancestry(tree,i):
            nodes,parents=tree;chain=[]
            while i>=0:chain.append(nodes[i]);i=parents[i]
            return chain[::-1]
        a,ia,b,ib=joined;ca=ancestry(trees[a],ia);cb=ancestry(trees[b],ib)
        path=ca+cb[::-1] if a==0 else cb+ca[::-1]
        for _ in range(120):
            if len(path)<3:break
            i,j=sorted(rng.choice(len(path),2,replace=False))
            if j>i+1 and edge(path[i],path[j]):path=path[:i+1]+path[j:]
    print(key,'joint swing nodes',len(path),'collision queries',evaluations,flush=True)
    return np.array(path)


def plan_contact_cycle(env,nodes=13,prep_z=1.115,yaw_deg=120.,prep_x=.17,unload_x=None,waist_pitch=0.,shift_y=0.,shift_z=None,shift_waist=None,curved_transfer=False,transfer_seconds=None,closed_hold_seconds=4.):
    """Use sole-to-rod tangency instead of prescribing an infeasible flat ankle."""
    ik=LimbIK(env);targets=initial_targets(env)
    targets['lf'][1]=.2;targets['rf'][1]=-.2
    q=env.initial.copy();q[:3]=[.1,0,1.07];q,err=ik.pose(q,targets)
    initial=q.copy()
    targets['lf']=rung(.3)+[0,.2,0];targets['rf']=rung(.3)+[0,-.2,0]
    angles={'lf':0.,'rf':0.}
    knots=[q.copy()];times=[0.];labels=['settle'];masks=[np.ones(4,bool)];errors=[max(err.values())];collisions=[]
    def seg(label,duration,key=None,point=None,root=None,yaw=None,flat=False,waist=None):
        nonlocal q,targets
        start=times[-1];old={k:v.copy() for k,v in targets.items()};root0=q[:3].copy();a0=angles.get(key,0.);waist0=q[env.qadr[14]]
        mask=np.ones(4,bool)
        if key is not None:mask[['lf','rf','lh','rh'].index(key)]=False
        local_nodes=max(nodes,81) if label.endswith('straighten') else nodes
        if yaw_deg<100 and label=='lf_unload':local_nodes=max(nodes,121)
        for u in np.linspace(0,1,local_nodes)[1:]:
            w=smooth(u);desired={k:v.copy() for k,v in old.items()}
            if key is not None:desired[key]=(1-w)*old[key]+w*point
            if yaw is not None:angles[key]=(1-w)*a0+w*yaw
            q[:3]=root0 if root is None else (1-w)*root0+w*root
            if curved_transfer and label=='lf_unload':
                # Delay forward motion until the raised stance knee clears the rung.
                q[0]=float(PchipInterpolator([.99,1.04,1.10,1.16,1.23],[-.10,-.10,-.065,-.06,-.02])(q[2]))
            if waist is not None:q[env.qadr[14]]=(1-w)*waist0+w*waist
            es=[]
            for fk in ['lf','rf']:
                q,e=ik.contact_leg(q,fk,desired[fk],angles[fk],pitch_weight=.05 if flat else .0005);es.append(e)
            q,he=ik.pose(q,{k:desired[k] for k in ['lh','rh']});es+=list(he.values())
            cs=ik.collisions(q)
            if cs:collisions.append(dict(time=start+duration*u,phase=label,contacts=cs))
            knots.append(q.copy());times.append(start+duration*u);labels.append(label);masks.append(mask.copy());errors.append(max(es))
        targets=desired
        print(label,'IK',max(errors[-nodes+1:]),'collisions',len([c for c in collisions if c['phase']==label]),flush=True)
    def swing(key,point,yaw):
        nonlocal q,targets
        path=plan_joint_swing(ik,q,key,point,yaw)
        qa=env.qadr[ik.slices[key]];mask=np.ones(4,bool);mask[['lf','rf','lh','rh'].index(key)]=False
        for a,b in zip(path[:-1],path[1:]):
            start=times[-1];duration=float(np.ceil(max(2.,float(np.max(abs(b-a)))*2.5)/.02)*.02)
            for u in np.linspace(0,1,max(nodes,int(duration*10)))[1:]:
                q[qa]=(1-smooth(u))*a+smooth(u)*b
                knots.append(q.copy());times.append(start+duration*u);labels.append(key+'_swing');masks.append(mask.copy());errors.append(0.)
        targets[key]=point.copy();angles[key]=yaw
    seg('settle',3.)
    for key in ['rh','lh']:
        p=targets[key].copy();up=np.array([.25*np.tan(np.deg2rad(5)),0,.25])
        for label,duration,offset in [('lift',2,[0,0,.05]),('retreat',2,[-.07,0,.05]),
            ('raise',3,up+[-.07,0,.05]),('approach',2,up+[0,0,.05]),('seat',2,up)]:
            seg(key+'_'+label,duration,key,p+offset)
        seg(key+'_load',2.)
    seg('step_preparation',6. if waist_pitch else 4.,root=np.array([prep_x,0,prep_z]),waist=waist_pitch)
    for key,sign in [('rf',-1),('lf',1)]:
        p=targets[key].copy();end=rung(.55)+[0,sign*.2,0]
        if key=='lf':
            if shift_y or shift_z is not None:
                seg('weight_transfer',4.,root=np.array([prep_x,shift_y,prep_z if shift_z is None else shift_z]),waist=waist_pitch if shift_waist is None else shift_waist)
            seg(key+'_unload',transfer_seconds if transfer_seconds is not None else (6. if yaw_deg<100 else 3.),key,p+[0,0,min(.2,1.23-prep_z)],root=np.array([prep_x if unload_x is None else unload_x,shift_y,1.23]));p=targets[key].copy()
        else:seg(key+'_unload',2.,key,p+[0,0,.008]);p=targets[key].copy()
        swing(key,end+[0,0,.015],sign*np.deg2rad(yaw_deg))
        seg(key+'_land',2.,key,end)
        seg(key+'_load',2.)
    seg('body_rise',8. if waist_pitch else 5.,root=np.array([.14,0,1.34]),waist=0.)
    for key in (['rf','lf'] if abs(yaw_deg)>1e-6 else []):
        landing=targets[key].copy()
        seg(key+'_straighten_unload',2.,key,landing+[0,0,.015])
        seg(key+'_straighten',5.,key,landing+[0,0,.015],yaw=0.,root=np.array([.1218721659,0,1.34]))
        seg(key+'_reseat',2.,key,landing)
        seg(key+'_reload',2.)
    seg('final_center',3.,root=np.array([.1218721659,0,1.32]),flat=True)
    seg('final_hold',4.,flat=True)
    plan=dict(qpos=np.array(knots),time=np.array(times),phase=np.array(labels),active=np.array(masks),
                initial=initial,errors=np.array(errors),collisions=collisions,initial_collisions=ik.collisions(initial))
    return close_cycle_posture(env,plan,hold_seconds=closed_hold_seconds)


def close_cycle_posture(env,plan,hold_seconds=4.):
    """Close arm redundancy smoothly, retaining all four contact poses."""
    ik=LimbIK(env);start=plan['qpos'][-1].copy();goal=plan['qpos'][0].copy()
    goal[:3]=start[:3];ik.d.qpos[:]=goal;mujoco.mj_forward(ik.m,ik.d)
    hands={k:ik.d.xpos[ik.bodies[k]].copy() for k in ['lh','rh']}
    states=[];times=[];errors=[];labels=[];q=start.copy();t0=float(plan['time'][-1])
    for u in np.linspace(0,1,121)[1:]:
        prior=(1-smooth(u))*start+smooth(u)*goal
        q,es=ik.pose(prior,hands,regularization=.001)
        states.append(q);times.append(t0+6*u);errors.append(max(es.values()));labels.append('posture_closure')
    for u in np.linspace(0,hold_seconds,round(hold_seconds/.05)+1)[1:]:
        states.append(q.copy());times.append(t0+6+u);errors.append(errors[-1]);labels.append('closed_hold')
    result=dict(plan)
    for key,values in [('qpos',states),('time',times),('errors',errors),('phase',labels),('active',np.ones((len(states),4),bool))]:
        result[key]=np.concatenate([plan[key],np.asarray(values)])
    return result


def initial_targets(env, hand_row=4):
    result={}
    for k,name in [('lf','left_ankle_roll_link'),('rf','right_ankle_roll_link')]:
        result[k]=env.data.xpos[env.model.body(name).id].copy()
    for k,y in [('lh',.2),('rh',-.2)]:
        result[k]=rung(.3+.25*hand_row)+[0,y,-.0068]
    return result


def solve_force_qp_fallback(H,objective,A,lower,upper):
    """Solve the identical convex QP from a feasible LP seed, without relaxation."""
    n=len(objective)
    equality=np.isfinite(lower)&np.isfinite(upper)&(abs(upper-lower)<1e-10)
    up=np.flatnonzero(~equality&np.isfinite(upper));low=np.flatnonzero(~equality&np.isfinite(lower))
    seed=linprog(np.zeros(n),A_ub=np.vstack([A[up],-A[low]]),b_ub=np.r_[upper[up],-lower[low]],
                 A_eq=A[equality],b_eq=upper[equality],bounds=[(None,None)]*n,method='highs')
    if not seed.success:raise RuntimeError('Contact force allocation fallback has no feasible LP seed: '+seed.message)
    scale=max(1.,float(np.diag(H).max()))
    constraints=[LinearConstraint(A[equality],lower[equality],upper[equality]),
                 LinearConstraint(A[~equality],lower[~equality],upper[~equality])]
    result=minimize(lambda f:(.5*f@H@f+objective@f)/scale,seed.x,
                    jac=lambda f:(H@f+objective)/scale,method='SLSQP',constraints=constraints,
                    options=dict(ftol=1e-9,maxiter=200))
    violation=float(max(0.,np.max(lower-A@result.x),np.max(A@result.x-upper)))
    if not result.success or not np.isfinite(result.x).all() or violation>1e-5:
        raise RuntimeError('Contact force allocation fallback failed: '+result.message)
    return result.x


class CycleTorqueController:
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
        upper=np.r_[required[:6],required[e.vadr]-e.limits[:,0],np.full(count,350.),np.zeros(8*count)]
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
