"""One right-foot transfer from a recorded dynamic prefix, with ideal force feedback.

Smoke-training distribution is a declared finite set of geometry errors, not a
continuous randomization or held-out robustness claim. Reset restores qpos/qvel;
solver warm-start state is reset, so continuity must be checked separately.
"""
import json
import hashlib
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from training.rung_residual_env import RungResidualEnv, ROOT
from training.wall_support_env import WallSupportEnv
from training.contact_frame_controller import ContactFrameController
from training.climb_robustness import apply_rung_offsets, ReleaseInterlock, GROUPS


class FootResidualEnv(RungResidualEnv):
    def __init__(self, suite, cases=None):
        gym.Env.__init__(self)
        self.suite=Path(suite)
        protocol=json.loads((self.suite/'protocol.json').read_text())
        self.plan=np.load(protocol['reference'])
        reference_hash=hashlib.sha256(Path(protocol['reference']).read_bytes()).hexdigest()
        self.cases=cases or ['nominal','row01_x_-5mm','row01_x_+5mm','row01_z_-5mm','row01_z_+5mm']
        self.prefixes={}
        for name in self.cases:
            run=self.suite/name
            manifest=json.loads((run/'manifest.json').read_text())
            if manifest['plan_sha256']!=reference_hash:
                raise ValueError('Prefix reference differs from training reference')
            with np.load(run/'trajectory.npz') as z:
                i=int(np.argmin(abs(z['time_s']-34.5)))
                if abs(z['time_s'][i]-34.5)>1e-6:
                    raise ValueError('Recorded prefix must reach 34.5 s')
                q=z['qpos'][i].copy();v=z['qvel'][i].copy()
            variant=json.loads((run/'scene_variant.json').read_text())
            self.prefixes[name]=(q,v,variant['offsets'])
        self.env=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2')
        self.nominal_geoms=self.env.model.geom_pos.copy()
        self.nominal_sites=self.env.model.site_pos.copy()
        self.nominal_bvh=self.env.model.bvh_aabb.copy()
        self.action_space=spaces.Box(-1.,1.,(6,),dtype=np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,(76,),dtype=np.float32)
        self.start_time=34.5;self.end_time=49.56
        self.max_steps=int(round((self.end_time-self.start_time)/.02))
        self.finished=True

    def observation(self):
        e=self.env;q,_,_=self.reference(e.data.time)
        metrics=e.contact_metrics()
        gravity=e.data.xmat[e.pelvis].reshape(3,3).T@np.array([0.,0.,-1.])
        phase=2*np.pi*np.clip((e.data.time-self.start_time)/(self.end_time-self.start_time),0.,1.)
        return np.r_[e.data.qpos[e.qadr]-q[e.qadr],e.data.qvel[e.vadr],gravity,
                     e.data.qpos[:3]-q[:3],np.sin(phase),np.cos(phase),
                     [metrics[g+'_vertical_n']/e.robot_weight_n for g in GROUPS],self.previous_action].astype(np.float32)

    def reset(self,*,seed=None,options=None):
        gym.Env.reset(self,seed=seed)
        name=(options or {}).get('case',self.cases[int(self.np_random.integers(len(self.cases)))])
        q,v,offsets=self.prefixes[name];e=self.env
        e.model.geom_pos[:]=self.nominal_geoms;e.model.site_pos[:]=self.nominal_sites
        e.model.bvh_aabb[:]=self.nominal_bvh
        e.reset();self.variant=apply_rung_offsets(e,offsets)
        e.data.qpos[:]=q;e.data.qvel[:]=v;e.data.time=self.start_time
        e.initial=self.plan['qpos'][0].copy();mujoco.mj_forward(e.model,e.data)
        self.controller=ContactFrameController(e,torque_weight=10000.)
        self.guard=ReleaseInterlock(e);self.guard.expected=['rung_00','rung_00','rung_05','rung_05']
        self.previous_action=np.zeros(6);self.residual=np.zeros(6)
        self.steps=0;self.finished=False;self.case=name
        self.other_peak=0.;self.self_peak=0.;self.flight_peak=0.;self.base_peak=0.;self.torque_peak=0.
        self.last_torque=np.zeros(29)
        self.joint_minimum=float(np.min(np.minimum(e.data.qpos[e.qadr]-e.model.jnt_range[e.joints,0],e.model.jnt_range[e.joints,1]-e.data.qpos[e.qadr])))
        if not np.all(e.model.jnt_limited[e.joints]):raise ValueError('Expected bounded joints')
        if self.joint_minimum < -1e-7:raise ValueError('Recorded initial state violates native joint bounds')
        return self.observation(),dict(case=name,contract='contact-foot-residual-v1-76obs-6act',
            prefix='Recorded qpos/qvel at 34.5 s; solver warm start reset; 0.5 s before residual activation')

    def step(self,action):
        if self.finished:raise RuntimeError('Reset required')
        action=np.asarray(action,dtype=float)
        if action.shape!=(6,) or not np.isfinite(action).all():raise ValueError('Invalid residual action')
        action=np.clip(action,-1,1);e=self.env;m=e.model;d=e.data
        reason='running';self.steps+=1
        for _ in range(4):
            q,v,active=self.reference(float(d.time))
            self.guard.sample(float(d.time))
            if not self.guard.allow(float(d.time),active):reason='release_interlock';break
            old=self.residual.copy();ramp=float(np.clip((d.time-35.)/.5,0.,1.))
            self.residual+=.05*(.015*ramp*action-self.residual)
            q[e.qadr[6:12]]+=self.residual;v[e.vadr[6:12]]+=(self.residual-old)/.005
            try:torque,info=self.controller.command(q,v,active)
            except RuntimeError as exc:reason='allocation_failure: '+str(exc);break
            self.last_torque=torque.copy();self.torque_peak=max(self.torque_peak,info['torque_ratio'])
            d.ctrl[e.motors]=torque
            for _ in range(10):
                if np.any(d.qfrc_applied) or np.any(d.xfrc_applied):raise AssertionError('External force forbidden')
                mujoco.mj_step(m,d);mujoco.mj_kinematics(m,d)
                margin=float(np.min(np.minimum(d.qpos[e.qadr]-m.jnt_range[e.joints,0],m.jnt_range[e.joints,1]-d.qpos[e.qadr])))
                self.joint_minimum=min(self.joint_minimum,margin)
                if margin < -1e-7:reason='native_joint_position_violation';break
                self.base_peak=max(self.base_peak,float(np.linalg.norm(d.qpos[:3]-q[:3])))
                for ci,c in enumerate(d.contact):
                    ba,bb=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
                    ia,ib=ba in e.robot_bodies,bb in e.robot_bodies
                    if not (ia or ib):continue
                    wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench);force=float(np.linalg.norm(wrench[:3]))
                    if ia and ib:self.self_peak=max(self.self_peak,force)
                    elif (ba if ia else bb) not in e.contact_groups:self.other_peak=max(self.other_peak,force)
                    elif 37.005<=d.time<45.56-1e-7 and (ba if ia else bb)==m.body('right_ankle_roll_link').id:
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
            self.guard.sample(float(d.time));h=self.guard.history
            success=bool(h and d.time-h[0][0]>=.25-1e-7 and all(all(v) for _,v in h))
            reason='success' if success else 'unconfirmed_terminal_support'
        self.finished=terminal or timeout
        q,_,_=self.reference(d.time)
        error=float(np.mean((d.qpos[e.qadr]-q[e.qadr])**2))
        reward=.02-2*error-.001*float(np.mean(action**2))
        if self.finished:reward+=20. if success else -20.
        info=dict(minimum_native_joint_margin_rad=self.joint_minimum,controller_assumption='Exact MuJoCo contact points, normals, topology and ideal forces',success=success,reason=reason,time_s=float(d.time),case=self.case,
                  base_error_peak_m=self.base_peak,other_contact_peak_n=self.other_peak,
                  self_contact_peak_n=self.self_peak,flight_contact_peak_n=self.flight_peak,
                  torque_peak_ratio=self.torque_peak,offsets=self.variant['offsets'],
                  scope='One right-foot transfer; finite reset set; ideal contact feedback; no full-cycle learning claim')
        return self.observation(),float(reward),terminal,timeout,info
