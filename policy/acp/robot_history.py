"""Bounded bridge samples for ACP observations at the localization time.

No interpolation or extrapolation: use a fresh, uniquely received SDK tick.
The latest bridge sample remains the source of controls and PD limiting.
"""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RobotSample:
    tick: int
    captured_at: float
    q: np.ndarray
    dq: np.ndarray
    gyro: np.ndarray
    imu_quat_wxyz: np.ndarray


class RobotHistory:
    def __init__(self, max_age=.15, max_skew=.04):
        self.max_age, self.max_skew = float(max_age), float(max_skew)
        if not 0 < self.max_age <= .5 or not 0 <= self.max_skew <= self.max_age:
            raise ValueError('Invalid ACP robot history timing limits')
        self.samples = deque(maxlen=64)
        self.failed = ''

    def add(self, state, stamp, now):
        if self.failed:
            raise ValueError(self.failed)
        tick = state.tick
        if isinstance(tick, bool) or not isinstance(tick, (int, np.integer)) or tick <= 0:
            raise ValueError('Invalid ACP bridge history tick')
        if not math.isfinite(stamp) or not 0 <= now-stamp <= self.max_age:
            raise ValueError('ACP bridge history receipt stale or future')
        if self.samples:
            previous = self.samples[-1]
            if tick < previous.tick or (tick != previous.tick and stamp <= previous.captured_at):
                self.failed = 'ACP bridge tick/time reset; restart policy'
                raise ValueError(self.failed)
            if tick == previous.tick:
                return  # Re-forwarded ticks cannot refresh the stored capture time.
        values = []
        for name, size in (('q', 29), ('dq', 29), ('imu_gyro', 3), ('imu_quat_wxyz', 4)):
            value = np.array(getattr(state, name), dtype=np.float32, copy=True)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError('Invalid ACP bridge history '+name)
            if name == 'imu_quat_wxyz':
                norm = np.linalg.norm(value)
                if abs(norm-1.) > .05:
                    raise ValueError('Invalid ACP bridge history quaternion norm')
                value /= norm
            value.setflags(write=False)
            values.append(value)
        self.samples.append(RobotSample(int(tick), float(stamp), *values))

    def nearest(self, stamp, now):
        if self.failed:
            raise ValueError(self.failed)
        if (not math.isfinite(stamp) or not 0 <= now-stamp <= self.max_age
                or not self.samples or not 0 <= now-self.samples[-1].captured_at <= self.max_age):
            raise ValueError('ACP robot history or localization stale/unavailable')
        sample = min(self.samples, key=lambda item: abs(item.captured_at-stamp))
        if (abs(sample.captured_at-stamp) > self.max_skew
                or not 0 <= now-sample.captured_at <= self.max_age):
            raise ValueError('ACP localization has no synchronized robot history sample')
        return sample
