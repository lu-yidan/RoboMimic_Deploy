"""Offline bounded whole-body correction for explicitly supplied hand targets."""
import mujoco
import numpy as np
from scipy.optimize import least_squares


def refine_wholebody_hand_targets(env, plan, *, offset_m, windows, continuation=False):
    offset = np.asarray(offset_m, dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all() or abs(offset).max() > .002000000001:
        raise ValueError('Expected finite XYZ shift no larger than 2 mm per axis')
    if set(windows) != {'lh', 'rh'}:
        raise ValueError('Both hand windows must be declared')
    times = plan['time']
    gains = {}
    for key, group in (('lh', 2), ('rh', 3)):
        start, end = windows[key]
        if not np.isfinite([start, end]).all() or not times[0] <= start < end <= times[-1]:
            raise ValueError('Invalid hand ramp')
        first, last = [int(np.argmin(abs(times-t))) for t in (start, end)]
        if max(abs(times[first]-start), abs(times[last]-end)) > 1e-7:
            raise ValueError('Hand ramp must use existing knots')
        if np.any(plan['active'][first:last+1, group]):
            raise ValueError('Hand ramp must be unloaded')
        u = np.clip((times-times[first])/(times[last]-times[first]), 0., 1.)
        gains[key] = u**3*(10+u*(-15+6*u))
    result = {k: v.copy() for k, v in plan.items()}
    if not np.any(offset):
        return result, dict(identity=True, records=[])
    m, d = env.model, mujoco.MjData(env.model)
    bodies = [m.body(n).id for n in ('hook', 'right_hook', 'left_ankle_roll_link', 'right_ankle_roll_link')]
    records = []
    previous = np.zeros(3 + len(env.qadr))
    for index in np.flatnonzero((gains['lh'] > 0) | (gains['rh'] > 0)):
        original = plan['qpos'][index]
        d.qpos[:] = original
        mujoco.mj_kinematics(m, d)
        targets, orientations = d.xpos[bodies].copy(), d.xquat[bodies].copy()
        targets[0] += gains['lh'][index]*offset
        targets[1] += gains['rh'][index]*offset
        lower = np.r_[np.full(3, -.003/np.sqrt(3)),
                       np.maximum(-.15, m.jnt_range[env.joints, 0]-original[env.qadr])]
        upper = np.r_[np.full(3, .003/np.sqrt(3)),
                       np.minimum(.15, m.jnt_range[env.joints, 1]-original[env.qadr])]
        def errors(x):
            d.qpos[:] = original
            d.qpos[:3] += x[:3]
            d.qpos[env.qadr] += x[3:]
            mujoco.mj_kinematics(m, d)
            rotations = []
            for body, orientation in zip(bodies, orientations):
                difference, vector = np.zeros(4), np.zeros(3)
                mujoco.mju_mulQuat(difference, d.xquat[body], orientation*[1, -1, -1, -1])
                mujoco.mju_quat2Vel(vector, difference, 1.)
                rotations.append(vector)
            return d.xpos[bodies]-targets, np.asarray(rotations)
        def objective(x):
            position, rotation = errors(x)
            anchor = previous if continuation else np.zeros_like(x)
            return np.r_[position.ravel(), .2*rotation.ravel(), 1e-5*(x-anchor)]
        seed = previous if continuation else np.zeros(len(lower))
        fit = least_squares(objective, np.clip(seed, lower+1e-10, upper-1e-10),
            bounds=(lower, upper), max_nfev=400, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        position, rotation = errors(fit.x)
        pe = float(np.linalg.norm(position, axis=1).max())
        re = float(np.linalg.norm(rotation, axis=1).max())
        if pe >= 2e-5 or re >= 1e-4:
            raise ValueError(f'Whole-body correction failed at {times[index]}: {pe}, {re}')
        result['qpos'][index] = d.qpos.copy()
        previous = fit.x.copy()
        result['errors'][index] = max(float(result['errors'][index]), pe, .2*re)
        records.append(dict(index=int(index), time_s=float(times[index]), position_error_m=pe,
            rotation_error_rad=re, root_norm_m=float(np.linalg.norm(fit.x[:3])),
            maximum_joint_delta_rad=float(abs(fit.x[3:]).max()), evaluations=fit.nfev))
    return result, dict(identity=False, offset_m=offset.tolist(), windows=windows, continuation=continuation, records=records,
                       scope='Independent offline IK knots; no collision or dynamic acceptance')
