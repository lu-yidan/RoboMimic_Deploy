"""Single right-hook transfer: a bounded residual on the verified torque controller.

This research contract is distinct from the old 100D/29D deployment policy.
Actor sees nominal tracking error, proprioception and ideal contact force, never
the randomized true bar coordinates. Contact identity is privileged guard data.
"""
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.climb_cycle import CycleTorqueController
from training.climb_robustness import apply_rung_offsets, ReleaseInterlock, GROUPS

ROOT=Path(__file__).resolve().parents[1]


class RungResidualEnv(gym.Env):
    def __init__(self, randomization_m=.005):
        super().__init__()
        self.env=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2')
        self.plan=np.load(ROOT/'outputs/compact_cycle_plan_v13/plan.npz')
        self.nominal_geoms=self.env.model.geom_pos.copy()
        self.nominal_sites=self.env.model.site_pos.copy()
        self.randomization_m=float(randomization_m)
        self.action_space=spaces.Box(-1.,1.,(7,),dtype=np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,(77,),dtype=np.float32)
        self.max_steps=800
        self.finished=True

    def reference(self,t):
        times=self.plan['time'];qs=self.plan['qpos'];e=self.env
        j=min(max(np.searchsorted(times,t,side='right')-1,0),len(times)-2)
        dt=times[j+1]-times[j];u=np.clip((t-times[j])/dt,0,1)
        q=(1-u)*qs[j]+u*qs[j+1];v=np.zeros(e.model.nv)
        v[:3]=(qs[j+1,:3]-qs[j,:3])/dt
        v[e.vadr]=(qs[j+1,e.qadr]-qs[j,e.qadr])/dt
        return q,v,self.plan['active'][j+1]

    def observation(self):
        e=self.env;q,_,_=self.reference(e.data.time)
        metrics=e.contact_metrics()
        gravity=e.data.xmat[e.pelvis].reshape(3,3).T@np.array([0.,0.,-1.])
        phase=2*np.pi*min(e.data.time/16.,1.)
        return np.r_[e.data.qpos[e.qadr]-q[e.qadr],e.data.qvel[e.vadr],gravity,
                     e.data.qpos[:3]-q[:3],np.sin(phase),np.cos(phase),
                     [metrics[g+'_vertical_n']/e.robot_weight_n for g in GROUPS],self.previous_action].astype(np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);e=self.env
        e.model.geom_pos[:]=self.nominal_geoms;e.model.site_pos[:]=self.nominal_sites
        e.reset();e.data.qpos[:]=self.plan['qpos'][0];e.initial=e.data.qpos.copy()
        delta=np.asarray((options or {}).get('offset',[self.np_random.uniform(-self.randomization_m,self.randomization_m),0.,self.np_random.uniform(-self.randomization_m,self.randomization_m)]),dtype=float)
        self.variant=apply_rung_offsets(e,{'rung_05':delta.tolist()})
        self.controller=CycleTorqueController(e,torque_weight=10000.)
        self.guard=ReleaseInterlock(e);self.previous_action=np.zeros(7)
        self.residual=np.zeros(7);self.steps=0;self.finished=False
        self.other_peak=0.;self.self_peak=0.;self.flight_peak=0.;self.base_peak=0.;self.torque_peak=0.
        self.last_torque=np.zeros(29)
        return self.observation(),dict(offset_m=delta.tolist(),contract='rung-residual-v1-77obs-7act')

    def step(self,action):
        if self.finished:raise RuntimeError('Reset required')
        action=np.asarray(action,dtype=float)
        if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError('Invalid residual action')
        action=np.clip(action,-1,1);e=self.env;m=e.model;d=e.data
        reason='running';self.steps+=1
        for _ in range(4):
            q,v,active=self.reference(float(d.time))
            self.guard.sample(float(d.time))
            if not self.guard.allow(float(d.time),active):reason='release_interlock';break
            # Smooth 50 Hz actions at 200 Hz, capped at 0.01 rad per right-arm joint.
            ramp=float(np.clip((d.time-3.)/.5,0.,1.))
            old=self.residual.copy();self.residual+=.05*(.01*ramp*action-self.residual)
            q[e.qadr[22:29]]+=self.residual;v[e.vadr[22:29]]+=(self.residual-old)/.005
            try:torque,info=self.controller.command(q,v,active)
            except RuntimeError as exc:reason='allocation_failure: '+str(exc);break
            self.last_torque=torque.copy();self.torque_peak=max(self.torque_peak,info['torque_ratio'])
            d.ctrl[e.motors]=torque
            for _ in range(10):
                if np.any(d.qfrc_applied) or np.any(d.xfrc_applied):raise AssertionError('External force forbidden')
                mujoco.mj_step(m,d);mujoco.mj_kinematics(m,d)
                self.base_peak=max(self.base_peak,float(np.linalg.norm(d.qpos[:3]-q[:3])))
                for ci,c in enumerate(d.contact):
                    ba,bb=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
                    ia,ib=ba in e.robot_bodies,bb in e.robot_bodies
                    if not (ia or ib):continue
                    wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench);force=float(np.linalg.norm(wrench[:3]))
                    if ia and ib:self.self_peak=max(self.self_peak,force)
                    elif (ba if ia else bb) not in e.contact_groups:self.other_peak=max(self.other_peak,force)
                    elif 5.005<=d.time<12.-1e-7 and (ba if ia else bb)==m.body('right_hook').id:
                        self.flight_peak=max(self.flight_peak,force)
                if not np.isfinite(d.qpos).all() or np.any(d.warning.number):reason='physics_failure';break
                if self.base_peak>.03:reason='base_tracking_loss';break
                if max(self.other_peak,self.self_peak)>1.:reason='unintended_contact';break
                if self.flight_peak>1.:reason='swing_contact';break
            if reason!='running':break
        self.previous_action=action.copy()
        terminal=reason!='running';timeout=self.steps>=self.max_steps and not terminal
        success=False
        if timeout:
            self.guard.sample(float(d.time))
            history=self.guard.history
            success=bool(history and d.time-history[0][0]>=.25-1e-7 and all(all(v) for _,v in history))
            reason='success' if success else 'unconfirmed_terminal_support'
        self.finished=terminal or timeout
        q,_,_=self.reference(d.time)
        error=float(np.mean((d.qpos[e.qadr]-q[e.qadr])**2))
        reward=.02-2*error-.001*float(np.mean(action**2))
        if self.finished:reward+=20. if success else -20.
        info=dict(success=success,reason=reason,time_s=float(d.time),base_error_peak_m=self.base_peak,
                  other_contact_peak_n=self.other_peak,self_contact_peak_n=self.self_peak,
                  flight_contact_peak_n=self.flight_peak,
                  torque_peak_ratio=self.torque_peak,offsets=self.variant['offsets'],
                  scope='one right-hook transfer; ideal simulation force feedback; no full-cycle learned-policy claim')
        return self.observation(),float(reward),terminal,timeout,info

    def close(self):self.env.close()
