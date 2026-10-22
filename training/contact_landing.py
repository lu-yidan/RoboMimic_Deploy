"""One-cycle left-foot landing search with privileged contact confirmation."""
import mujoco
import numpy as np
from training.climb_cycle import LimbIK
from training.climb_robustness import ReleaseInterlock


class ContactLanding:
    def __init__(self,env):
        self.e=env;self.ik=LimbIK(env);self.guard=ReleaseInterlock(env)
        self.guard.expected=['rung_01','rung_01','rung_05','rung_05']
        self.mode='waiting';self.depth=0.;self.confirmed_at=None;self.previous=np.zeros(6)
        self.records=[];self.maximum_ik_error=0.

    def update(self,q,v,active,phase,controller):
        e=self.e;t=float(e.data.time);self.guard.sample(t)
        q=q.copy();v=v.copy();active=np.asarray(active,dtype=bool).copy()
        metrics=e.contact_metrics()
        contact=any(c['group']=='left_foot' and c['external_geom']=='rung_01' for c in metrics['contacts'])
        if phase=='lf_load' and self.mode=='waiting':self.mode='searching'
        if self.mode!='waiting':
            if self.mode=='searching':
                if contact:self.mode='confirming'
                else:self.depth=min(.015,self.depth+.02*.005)
            if self.mode=='confirming':
                if not contact:self.mode='searching'
                elif self.guard.history and t-self.guard.history[0][0]>=.25-1e-7 and all(r[1][0] for r in self.guard.history):
                    self.mode='ramping';self.confirmed_at=t
            if self.mode=='ramping' and t-self.confirmed_at>=.5:self.mode='loaded'
            if self.mode in ('ramping','loaded') and not contact:
                raise RuntimeError('Confirmed left landing contact lost')
            active[0]=contact
            cap=10. if self.confirmed_at is None else 10.+340.*min(1.,(t-self.confirmed_at)/.5)
            controller.normal_caps={'lf':cap}
            if phase!='lf_load' and self.mode!='loaded':
                raise RuntimeError('Landing not confirmed before body rise')
        if self.depth:
            self.ik.d.qpos[:]=q;mujoco.mj_kinematics(e.model,self.ik.d);bid=self.ik.bodies['lf']
            target=self.ik.d.xpos[bid].copy();target[2]-=self.depth
            quat=self.ik.d.xquat[bid].copy()
            corrected,errors=self.ik.pose(q,{'lf':target},orientations={'lf':quat})
            self.maximum_ik_error=max(self.maximum_ik_error,errors['lf'])
            if errors['lf']>.002:raise RuntimeError('Online landing IK error exceeds 2 mm')
            delta=corrected[e.qadr[:6]]-q[e.qadr[:6]]
            q=corrected;v[e.vadr[:6]]+=(delta-self.previous)/.005;self.previous=delta
        self.records.append(dict(time_s=t,phase=phase,mode=self.mode,depth_m=self.depth,lf_contact=contact,
            lf_valid=self.guard.history[-1][1][0],
            lf_force_n=metrics['left_foot_vertical_n'],lf_normal_cap_n=getattr(controller,'normal_caps',{}).get('lf',350.)))
        return q,v,active
