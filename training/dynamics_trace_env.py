"""Read-only instrumentation of original controller targets and realized dynamics."""
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv


class CommandTrace:
    def __init__(self,controller,owner):self.controller=controller;self.owner=owner
    def __getattr__(self,key):return getattr(self.controller,key)
    def command(self,qref,vref,active):
        o=self.owner;e=o.env;d=e.data;c=self.controller
        kp=np.full(29,c.joint_kp);kp[:12]=c.leg_kp
        kd=np.full(29,c.joint_kd);kd[:12]=2*np.sqrt(c.leg_kp)
        o.command_record=dict(start=float(d.time),velocity=d.qvel[e.vadr].copy(),
            desired=kp*(qref[e.qadr]-d.qpos[e.qadr])+kd*(vref[e.vadr]-d.qvel[e.vadr]),
            reference_error=qref[e.qadr]-d.qpos[e.qadr],active=active.copy(),samples=0)
        return c.command(qref,vref,active)


class DynamicsTraceEnv(FullCycleResidualEnv):
    def reset(self,**kwargs):
        self.dynamics_records=[];self.command_record=None
        return super().reset(**kwargs)

    def select_controller(self,active,phase):
        return CommandTrace(super().select_controller(active,phase),self)

    def physics_check(self,qref,phase,controller):
        reason=super().physics_check(qref,phase,controller)
        r=self.command_record;r['samples']+=1
        if r['samples']==10 or reason!='running':
            d=self.env.data;e=self.env;dt=float(d.time)-r['start']
            self.dynamics_records.append(dict(time_s=float(d.time),phase=phase,dt_s=dt,
                desired_acceleration=r['desired'].copy(),
                average_acceleration=(d.qvel[e.vadr]-r['velocity'])/dt,
                reference_error=r['reference_error'].copy(),active=r['active'].copy(),
                torque=d.ctrl[e.motors].copy(),constraint_force=d.qfrc_constraint[e.vadr].copy()))
        return reason
