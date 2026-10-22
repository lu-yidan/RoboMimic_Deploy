"""Frozen76-observation policy in a continuous cycle; no simulator reset."""
import numpy as np
from training.climb_robustness import GROUPS


def load_observation(env,qref,time_s,previous_action):
    gravity=env.data.xmat[env.pelvis].reshape(3,3).T@np.array([0.,0.,-1.])
    phase=2*np.pi*np.clip((time_s-49.)/(61.56-49.),0.,1.)
    metrics=env.contact_metrics()
    return np.r_[env.data.qpos[env.qadr]-qref[env.qadr],env.data.qvel[env.vadr],gravity,
        env.data.qpos[:3]-qref[:3],np.sin(phase),np.cos(phase),
        [metrics[g+'_vertical_n']/env.robot_weight_n for g in GROUPS],previous_action].astype(np.float32)


class WindowResidualPolicy:
    def __init__(self,env,model):
        self.env=env;self.model=model;self.previous_action=np.zeros(6)
        self.action=np.zeros(6);self.residual=np.zeros(6);self.window_steps=0
        self.policy_records=[];self.control_records=[]

    def command(self,qref,vref,time_s):
        active=49.<=time_s<61.56
        if active:
            if self.window_steps%4==0:
                obs=load_observation(self.env,qref,time_s,self.previous_action)
                action=np.asarray(self.model.predict(obs,deterministic=True)[0],dtype=float)
                if action.shape!=(6,) or not np.isfinite(action).all():raise ValueError('Invalid policy action')
                self.action=np.clip(action,-1.,1.)
                self.policy_records.append(dict(time_s=float(time_s),observation=obs.tolist(),action=self.action.tolist()))
            self.window_steps+=1
            if self.window_steps%4==0:self.previous_action=self.action.copy()
        else:self.action=np.zeros(6)
        old=self.residual.copy();self.residual+=.05*(.015*self.action-self.residual)
        q=qref.copy();v=vref.copy()
        q[self.env.qadr[6:12]]+=self.residual
        v[self.env.vadr[6:12]]+=(self.residual-old)/.005
        self.control_records.append(dict(time_s=float(time_s),action=self.action.tolist(),offset_rad=self.residual.tolist()))
        return q,v
