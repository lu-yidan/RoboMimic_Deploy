"""Experimental clearance-aware planner; nominal planner/controller remain untouched.

Derived from training.climb_cycle on 2026-10-17. Changes are restricted to
kinematic planning: stance clearance, swing clearance, lift and body-rise targets.
"""
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
from training.hand_transfer import smooth
from training.climb_cycle import LimbIK as NominalLimbIK, rung, initial_targets, close_cycle_posture


class LimbIK(NominalLimbIK):
    def __init__(self,env,leg_clearance=.012,swing_clearance=.0015,swing_goal_seeds=None):
        super().__init__(env)
        if not (.001<=leg_clearance<=.04 and .001<=swing_clearance<=.03):
            raise ValueError('Clearance outside declared research range')
        self.leg_clearance=float(leg_clearance)
        self.swing_clearance=float(swing_clearance)
        self.swing_goal_seeds={} if swing_goal_seeds is None else swing_goal_seeds

    def contact_leg(self,q,key,point,yaw,pitch_weight=.0005):
        """Constrain sole tangency to a horizontal rod, leaving foot pitch free."""
        sl=self.slices[key];qa=self.env.qadr[sl];js=self.env.joints[sl];bid=self.bodies[key]
        prior=q[qa].copy();local=np.array([.09,0,-.035])
        def fun(x):
            self.d.qpos[:]=q;self.d.qpos[qa]=x;mujoco.mj_kinematics(self.m,self.d)
            R=self.d.xmat[bid].reshape(3,3);normal=R[:,2]
            contact=self.d.xpos[bid]+R@local
            angle=np.arctan2(R[1,0],R[0,0]);angle_error=np.arctan2(np.sin(angle-yaw),np.cos(angle-yaw))
            distances=[mujoco.mj_geomDistance(self.m,self.d,a,b,self.leg_clearance+.001,None) for a,b in self.leg_pairs[key]]
            return np.r_[contact-point-.0148*normal,.2*normal[1],.08*angle_error,
                         .0001*(x-prior),pitch_weight*normal[0],4.*np.maximum(0.,self.leg_clearance-np.array(distances))]
        fit=least_squares(fun,np.clip(prior,self.m.jnt_range[js,0]+1e-8,self.m.jnt_range[js,1]-1e-8),
                          bounds=(self.m.jnt_range[js,0],self.m.jnt_range[js,1]),max_nfev=120,
                          ftol=1e-9,xtol=1e-9,gtol=1e-9)
        residual=fun(fit.x);result=q.copy();result[qa]=fit.x
        return result,float(np.linalg.norm(residual[:5]))


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
                if mujoco.mj_geomDistance(m,d,g,b,ik.swing_clearance,None)<ik.swing_clearance:return False
        mujoco.mj_collision(m,d)
        for c in d.contact:
            if c.dist<-.0005 and (int(m.geom_bodyid[c.geom1]) in bodies or int(m.geom_bodyid[c.geom2]) in bodies):return False
        return True
    def edge(a,b):
        return all(valid((1-u)*a+u*b) for u in np.linspace(0,1,max(2,int(np.max(abs(b-a))/.018)+2))[1:])
    goal=None
    for attempt in range(48):
        q=qstart.copy()
        if attempt==0 and key in ik.swing_goal_seeds:
            q[qa]=np.asarray(ik.swing_goal_seeds[key])[qa]
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


def plan_contact_cycle(env,nodes=13,prep_z=1.115,yaw_deg=120.,prep_x=.17,unload_x=None,waist_pitch=0.,shift_y=0.,shift_z=None,shift_waist=None,curved_transfer=False,transfer_seconds=None,closed_hold_seconds=4.,leg_clearance=.012,swing_clearance=.0015,foot_lift=.008,approach_lift=.015,initialize_stance=False,body_rise_x=.14,body_rise_z=1.34,left_unload_lift=.2,swing_goal_seeds=None,initial_x=.1):
    """Use sole-to-rod tangency instead of prescribing an infeasible flat ankle."""
    ik=LimbIK(env,leg_clearance,swing_clearance,swing_goal_seeds);targets=initial_targets(env)
    targets['lf'][1]=.2;targets['rf'][1]=-.2
    q=env.initial.copy();q[:3]=[initial_x,0,1.07];q,err=ik.pose(q,targets)
    if initialize_stance:
        for fk,sign in [('lf',1),('rf',-1)]:
            q,error=ik.contact_leg(q,fk,rung(.3)+[0,sign*.2,0],0.)
            err[fk]=error
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
            seg(key+'_unload',transfer_seconds if transfer_seconds is not None else (6. if yaw_deg<100 else 3.),key,p+[0,0,min(left_unload_lift,1.23-prep_z)],root=np.array([prep_x if unload_x is None else unload_x,shift_y,1.23]));p=targets[key].copy()
        else:seg(key+'_unload',2.,key,p+[0,0,foot_lift]);p=targets[key].copy()
        swing(key,end+[0,0,approach_lift],sign*np.deg2rad(yaw_deg))
        seg(key+'_land',2.,key,end)
        seg(key+'_load',2.)
    seg('body_rise',8. if waist_pitch else 5.,root=np.array([body_rise_x,0,body_rise_z]),waist=0.)
    for key in (['rf','lf'] if abs(yaw_deg)>1e-6 else []):
        landing=targets[key].copy()
        seg(key+'_straighten_unload',2.,key,landing+[0,0,.015])
        seg(key+'_straighten',5.,key,landing+[0,0,.015],yaw=0.,root=np.array([.1218721659,0,1.34]))
        seg(key+'_reseat',2.,key,landing)
        seg(key+'_reload',2.)
    seg('final_center',3.,root=np.array([initial_x+.0218721659,0,1.32]),flat=True)
    seg('final_hold',4.,flat=True)
    plan=dict(qpos=np.array(knots),time=np.array(times),phase=np.array(labels),active=np.array(masks),
                initial=initial,errors=np.array(errors),collisions=collisions,initial_collisions=ik.collisions(initial))
    return close_cycle_posture(env,plan,hold_seconds=closed_hold_seconds)

