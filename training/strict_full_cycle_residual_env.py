"""Align training success with the independent endpoint acceptance checks.

The v1 environment remains frozen for running experiments and provenance.
Only terminal acceptance changes; control, observation and physical stepping do
not change. Desired endpoint poses are those of the audited nominal reference.
"""
from collections import deque
import mujoco
import numpy as np
from training.full_cycle_residual_env import FullCycleResidualEnv as BaseEnv

CONTRACT = 'full-cycle-residual-v2-strict-endpoint-136obs-26act'


class FullCycleResidualEnv(BaseEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contract = CONTRACT
        e = self.env
        self.endpoint_bodies = [e.model.body(name).id for name in
            ('hook', 'right_hook', 'left_ankle_roll_link', 'right_ankle_roll_link')]
        scratch = mujoco.MjData(e.model)
        self.desired_poses = []
        for end in self.ends:
            i = int(np.argmin(abs(self.plan['time']-end)))
            scratch.qpos[:] = self.plan['qpos'][i]
            mujoco.mj_kinematics(e.model, scratch)
            self.desired_poses.append(scratch.xpos[self.endpoint_bodies].copy())

    def reset(self, **kwargs):
        self.strict_tail = deque()
        return super().reset(**kwargs)

    def strict_sample(self, index):
        e, d = self.env, self.env.data
        metrics = dict(
            effector_error_m=float(np.linalg.norm(d.xpos[self.endpoint_bodies]-self.desired_poses[index], axis=1).max()),
            tilt_rad=float(np.arccos(np.clip(d.xmat[e.pelvis].reshape(3, 3)[2, 2], -1., 1.))),
            linear_speed_m_s=float(np.linalg.norm(d.qvel[:3])),
            angular_speed_rad_s=float(np.linalg.norm(d.qvel[3:6])),
            joint_speed_rad_s=float(abs(d.qvel[6:]).max()))
        passed = (metrics['effector_error_m'] < .01 and metrics['tilt_rad'] < np.deg2rad(3.)
                  and metrics['linear_speed_m_s'] < .01 and metrics['angular_speed_rad_s'] < .1
                  and metrics['joint_speed_rad_s'] < .1)
        return bool(passed), metrics

    def check_endpoint(self):
        index = len(self.cycle_results)
        t = float(self.env.data.time)
        passed, metrics = self.strict_sample(index)
        # Match the independent audit's declared final 2 s interval exactly.
        if self.ends[index]-2. < t <= self.ends[index]+1e-7:
            self.strict_tail.append((t, passed, metrics))
        base_passed = super().check_endpoint()
        if len(self.cycle_results) > index:
            strict_passed = len(self.strict_tail) >= 99 and all(item[1] for item in self.strict_tail)
            peaks = {key: max(item[2][key] for item in self.strict_tail) for key in metrics} if self.strict_tail else {}
            result = self.cycle_results[-1]
            result.update(base_endpoint_passed=bool(base_passed), strict_endpoint_passed=bool(strict_passed),
                          strict_sample_count=len(self.strict_tail), strict_peaks=peaks,
                          passed=bool(base_passed and strict_passed))
            self.strict_tail.clear()
            return result['passed']
        return base_passed
