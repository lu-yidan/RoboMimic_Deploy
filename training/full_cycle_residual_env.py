"""Full-cycle motor-only residual learning with an audited scripted controller.

This new contract intentionally does not load the old 76-observation checkpoint.
Actor observations include ideal support forces; the controller uses privileged
MuJoCo contact geometry. This is residual learning, not autonomous route planning.
"""
import hashlib
import json
from collections import deque
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from training.wall_support_env import WallSupportEnv
from training.climb_cycle import CycleTorqueController
from training.periodic_force_history_controller import PeriodicForceHistoryController
from training.repeated_rung_errors import apply_rung_offsets, GROUPS
from training.cycle_clock import cycle_index
from training.hand_transfer import smooth

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = 'full-cycle-residual-v1-136obs-26act'


class FullCycleResidualEnv(gym.Env):
    """50 Hz actor, 200 Hz bounded torque allocation, 2000 Hz MuJoCo.

    Episodes start at physical time zero. Each completed cycle is checked over
    its final two seconds. Failure terminates; reaching the declared horizon
    terminates too (finite-horizon task, with remaining time in observation).
    """

    def __init__(self, plan=None, cases=None, normal_bias_y_deg=-3., memory_enabled=True):
        super().__init__()
        self.plan_path = Path(plan or ROOT/'outputs/clearance_back20_leg14_plan_v1/plan.npz')
        self.plan_hash = hashlib.sha256(self.plan_path.read_bytes()).hexdigest()
        audit = json.loads((self.plan_path.parent/'audit.json').read_text())
        manifest = json.loads((self.plan_path.parent/'manifest.json').read_text())
        if not audit['passed'] or manifest['plan_sha256'] != self.plan_hash:
            raise ValueError('An audited, hash-matched reference is required')
        with np.load(self.plan_path) as data:
            self.plan = {key: data[key].copy() for key in data.files}
        self.ends = np.asarray(self.plan.get('cycle_ends_s', [self.plan['time'][-1]]), dtype=float)
        if not np.all(np.diff(np.r_[0., self.ends]) > 61.56) or abs(self.ends[-1]-self.plan['time'][-1]) > 1e-8:
            raise ValueError('Invalid complete cycle boundaries')
        self.horizon = float(self.ends[-1])
        if abs(self.horizon/.02-round(self.horizon/.02)) > 1e-6:
            raise ValueError('Horizon must align with policy steps')
        self.cases = cases or {'nominal': {}}
        self.normal_bias = float(normal_bias_y_deg)
        if not np.isfinite(self.normal_bias):
            raise ValueError('Nonfinite normal bias')
        self.memory_enabled = bool(memory_enabled)
        self.env = WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2')
        self.nominal_geometry = tuple(x.copy() for x in
            (self.env.model.geom_pos, self.env.model.site_pos, self.env.model.bvh_aabb))
        self.indices = np.r_[np.arange(12), np.arange(15, 29)]
        self.phases = sorted(set(map(str, self.plan['phase'])))
        # 29 joint error + 29 velocity + gravity3 + base error3 + base velocity6
        # + nominal velocity29 + active4 + phase clock3 + forces4 + residual26.
        self.observation_space = spaces.Box(-np.inf, np.inf, (136,), dtype=np.float32)
        self.action_space = spaces.Box(-1., 1., (26,), dtype=np.float32)
        self.contract = CONTRACT
        self.finished = True

    def reference(self, t):
        times, qs, e = self.plan['time'], self.plan['qpos'], self.env
        j = min(max(int(np.searchsorted(times, t, side='right')-1), 0), len(times)-2)
        dt = times[j+1]-times[j]
        u = float(np.clip((t-times[j])/dt, 0., 1.))
        q = (1-u)*qs[j]+u*qs[j+1]
        v = np.zeros(e.model.nv)
        v[:3] = (qs[j+1, :3]-qs[j, :3])/dt
        v[e.vadr] = (qs[j+1, e.qadr]-qs[j, e.qadr])/dt
        return q, v, self.plan['active'][j+1].copy(), str(self.plan['phase'][j+1])

    def observation(self):
        e, d = self.env, self.env.data
        q, v, active, _ = self.reference(float(d.time))
        i = min(cycle_index(float(d.time), self.ends), len(self.ends)-1)
        start = 0. if i == 0 else self.ends[i-1]
        angle = 2*np.pi*(d.time-start)/(self.ends[i]-start)
        metrics = e.contact_metrics()
        gravity = d.xmat[e.pelvis].reshape(3, 3).T@np.array([0., 0., -1.])
        return np.r_[d.qpos[e.qadr]-q[e.qadr], d.qvel[e.vadr], gravity,
                     d.qpos[:3]-q[:3], d.qvel[:6], v[e.vadr], active,
                     np.sin(angle), np.cos(angle), max(0., 1.-d.time/self.horizon),
                     [metrics[g+'_vertical_n']/e.robot_weight_n for g in GROUPS],
                     self.residual/.015].astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        names = list(self.cases)
        self.case = (options or {}).get('case', names[int(self.np_random.integers(len(names)))])
        if self.case not in self.cases:
            raise ValueError('Unknown declared geometry case')
        e, m, d = self.env, self.env.model, self.env.data
        for destination, original in zip((m.geom_pos, m.site_pos, m.bvh_aabb), self.nominal_geometry):
            destination[:] = original
        e.reset()
        self.variant = apply_rung_offsets(e, self.cases[self.case])
        e.initial = self.plan['qpos'][0].copy()
        d.qpos[:] = e.initial
        mujoco.mj_forward(m, d)
        self.baseline = CycleTorqueController(e, joint_kp=100., leg_kp=100., torque_weight=10000.)
        self.contact = PeriodicForceHistoryController(e, joint_kp=100., leg_kp=100.,
            torque_weight=10000., normal_bias_y_deg=self.normal_bias,
            memory_enabled=self.memory_enabled, cycle_ends=self.ends)
        self.controller_cycle = 0
        self.transfer_entered = False
        self.last_phase = None
        self.phase_start = 0.
        self.phase_gain = 100.
        self.residual = np.zeros(26)
        self.previous_action = np.zeros(26)
        self.last_torque = np.zeros(29)
        self.steps = 0
        self.finished = False
        self.peaks = dict(other=0., self=0., flight=0., base=0., torque=0.)
        self.minimum_margin = float('inf')
        self.tail = deque()
        self.cycle_results = []
        self.warning_start = d.warning.number.copy()
        self.last_mode = 'fixed_contact_points'
        return self.observation(), self.info('reset', False)

    def select_controller(self, active, phase):
        e, m, d = self.env, self.env.model, self.env.data
        current = cycle_index(float(d.time), self.ends)
        if current != self.controller_cycle:
            self.transfer_entered = False
            self.contact.leg_kp = 100.
            self.controller_cycle = current
        self.transfer_entered |= phase in ('step_preparation', 'weight_transfer', 'lf_unload', 'lf_swing', 'lf_land')
        controller = self.contact if self.transfer_entered else self.baseline
        if controller is self.contact:
            mujoco.mj_forward(m, d)
            available = set()
            for c in d.contact:
                a, b = int(c.geom1), int(c.geom2)
                ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
                for body, external, world in ((ba, b, bb), (bb, a, ba)):
                    name = m.geom(external).name
                    if world == 0 and name.startswith('rung_') and len(name) == 7 and name[-2:].isdigit():
                        available.add(body)
            missing = any(enabled and self.contact.ik.bodies[key] not in available
                          for enabled, key in zip(active, ('lf', 'rf', 'lh', 'rh')))
            if missing and not self.contact.bias_window_active():
                controller = self.baseline
        if phase != self.last_phase:
            self.last_phase, self.phase_start, self.phase_gain = phase, float(d.time), controller.leg_kp
        target = 300. if phase in ('final_center', 'final_hold', 'posture_closure', 'closed_hold') else 100.
        weight = smooth(float(np.clip((d.time-self.phase_start)/3., 0., 1.)))
        controller.leg_kp = (1-weight)*self.phase_gain+weight*target
        self.last_mode = 'contact_frame' if controller is self.contact else 'fixed_contact_points'
        return controller

    def physics_check(self, qref, phase, controller):
        e, m, d = self.env, self.env.model, self.env.data
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or np.any(d.warning.number > self.warning_start):
            return 'physics_failure'
        margin = float(np.min(np.minimum(d.qpos[e.qadr]-m.jnt_range[e.joints, 0],
                                       m.jnt_range[e.joints, 1]-d.qpos[e.qadr])))
        self.minimum_margin = min(self.minimum_margin, margin)
        self.peaks['base'] = max(self.peaks['base'], float(np.linalg.norm(d.qpos[:3]-qref[:3])))
        for ci, c in enumerate(d.contact):
            ba, bb = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            ina, inb = ba in e.robot_bodies, bb in e.robot_bodies
            if not (ina or inb):
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(m, d, ci, wrench)
            force = float(np.linalg.norm(wrench[:3]))
            if ina and inb:
                self.peaks['self'] = max(self.peaks['self'], force)
                continue
            body = ba if ina else bb
            if body not in e.contact_groups:
                self.peaks['other'] = max(self.peaks['other'], force)
            prefix = phase[:2]
            clear = phase.endswith(('_swing', '_straighten')) or (prefix in ('lh', 'rh') and phase.split('_', 1)[1] in ('retreat', 'raise', 'approach'))
            if clear and body == controller.ik.bodies.get(prefix):
                self.peaks['flight'] = max(self.peaks['flight'], force)
        if margin < -1e-7:
            return 'native_joint_position_violation'
        if self.peaks['base'] >= .03:
            return 'base_tracking_loss'
        if max(self.peaks['other'], self.peaks['self'], self.peaks['flight']) >= 1.:
            return 'unintended_or_swing_contact'
        return 'running'

    def check_endpoint(self):
        e, d = self.env, self.env.data
        metrics = e.contact_metrics()
        index = len(self.cycle_results)+1
        expected = {g: f'rung_{(4 if "hook" in g else 0)+index:02d}' for g in GROUPS}
        loaded = [c for c in metrics['contacts'] if np.linalg.norm(c['force_on_robot_world_n']) > 1.]
        valid = (abs(d.qpos[2]-e.initial[2]-.25*index) < .01
                 and min(metrics[g+'_vertical_n'] for g in GROUPS) > 10.
                 and bool(loaded) and all(c['group'] in expected and c['external_geom'] == expected[c['group']] for c in loaded)
                 and np.max(abs(d.qpos[e.qadr]-e.initial[e.qadr])) < .04)
        self.tail.append((float(d.time), bool(valid)))
        while self.tail and self.tail[0][0] <= d.time-2.+1e-7:
            self.tail.popleft()
        if abs(d.time-self.ends[index-1]) < 1e-7:
            passed = len(self.tail) >= 99 and all(v for _, v in self.tail)
            self.cycle_results.append(dict(cycle=index, time_s=float(d.time), passed=passed, samples=len(self.tail)))
            self.tail.clear()
            return passed
        return True

    def info(self, reason, success):
        return dict(contract=self.contract, case=self.case, reason=reason, success=bool(success),
                    time_s=float(self.env.data.time), peaks=self.peaks.copy(),
                    minimum_native_joint_margin_rad=self.minimum_margin,
                    cycle_results=list(self.cycle_results), controller_mode=self.last_mode,
                    scope='Full-cycle residual on a scripted reference; exact contact geometry and ideal force feedback')

    def step(self, action):
        if self.finished:
            raise RuntimeError('Reset required')
        action = np.asarray(action, dtype=float)
        if action.shape != (26,) or not np.isfinite(action).all():
            raise ValueError('Expected finite 26D residual action')
        action = np.clip(action, -1., 1.)
        e, m, d = self.env, self.env.model, self.env.data
        reason = 'running'
        for _ in range(4):
            q, v, active, phase = self.reference(float(d.time))
            controller = self.select_controller(active, phase)
            previous = self.residual.copy()
            self.residual += .05*(.015*action-self.residual)
            q[e.qadr[self.indices]] += self.residual
            v[e.vadr[self.indices]] += (self.residual-previous)/.005
            try:
                torque, details = controller.command(q, v, active)
            except RuntimeError as exc:
                reason = 'allocation_failure: '+str(exc)
                break
            self.last_torque = torque.copy()
            self.peaks['torque'] = max(self.peaks['torque'], details['torque_ratio'])
            d.ctrl[e.motors] = torque
            for _ in range(10):
                if np.any(d.qfrc_applied) or np.any(d.xfrc_applied):
                    raise AssertionError('External applied force forbidden')
                mujoco.mj_step(m, d)
                mujoco.mj_kinematics(m, d)
                reason = self.physics_check(q, phase, controller)
                if reason != 'running':
                    break
            if reason != 'running':
                break
        self.steps += 1
        if reason == 'running' and not self.check_endpoint():
            reason = 'cycle_endpoint_failure'
        success = reason == 'running' and len(self.cycle_results) == len(self.ends)
        if success:
            reason = 'success'
        self.finished = reason != 'running'
        q, _, _, _ = self.reference(float(d.time))
        error = np.mean((d.qpos[e.qadr]-q[e.qadr])**2)
        reward = .02-2.*error-.001*np.mean(action**2)-.001*np.mean((action-self.previous_action)**2)
        if self.finished:
            reward += 20. if success else -20.
        self.previous_action = action.copy()
        return self.observation(), float(reward), self.finished, False, self.info(reason, success)

    def close(self):
        self.env.close()
