"""Experimental offline hand-reference shift. Never installed by the controller.

A supplied displacement ramps during an unloaded interval and remains fixed
through the declared plan horizon. The caller must independently associate the
target rod and audit the full interpolated path before any dynamics experiment.
"""
import numpy as np
import mujoco
from training.climb_cycle import LimbIK


def retarget_hand_reference(env, plan, *, key, offset_m, start_s, end_s):
    offset = np.asarray(offset_m, dtype=float)
    if key not in ('lh', 'rh') or offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError('Expected a hand and a finite XYZ displacement')
    if np.max(abs(offset)) > .002000000001:
        raise ValueError('Prototype restricted to 2 mm per axis')
    times = np.asarray(plan['time'])
    if not np.isfinite([start_s, end_s]).all() or not times[0] <= start_s < end_s <= times[-1]:
        raise ValueError('Invalid ramp interval')
    first, last = [int(np.argmin(abs(times - t))) for t in (start_s, end_s)]
    if max(abs(times[first] - start_s), abs(times[last] - end_s)) > 1e-7:
        raise ValueError('Ramp boundaries must be existing plan knots')
    group = 2 if key == 'lh' else 3
    if np.any(plan['active'][first:last + 1, group]):
        raise ValueError('Displacement ramp must occur while the hand is unloaded')
    result = {k: v.copy() for k, v in plan.items()}
    u = np.clip((times - times[first]) / (times[last] - times[first]), 0., 1.)
    weights = u**3 * (10 + u * (-15 + 6*u))
    diagnostics = []
    if not np.any(offset):
        return result, dict(identity=True, changed_nodes=0, diagnostics=[])
    ik = LimbIK(env)
    data = mujoco.MjData(env.model)
    body = ik.bodies[key]
    qa = env.qadr[ik.slices[key]]
    other = np.ones(env.model.nq, dtype=bool)
    other[qa] = False
    for index in np.flatnonzero(weights > 0):
        original = plan['qpos'][index]
        data.qpos[:] = original
        mujoco.mj_kinematics(env.model, data)
        position, orientation = data.xpos[body].copy(), data.xquat[body].copy()
        target = position + weights[index] * offset
        changed, _ = ik.pose(original, {key: target}, orientations={key: orientation})
        # This local prototype must not trade a millimetre target shift for a
        # large joint-space branch change near a poorly conditioned posture.
        if float(abs(changed[qa] - original[qa]).max()) > .15:
            raise ValueError(f'Local hand correction exceeds 0.15 rad at {times[index]}')
        data.qpos[:] = changed
        mujoco.mj_kinematics(env.model, data)
        difference = np.zeros(4)
        mujoco.mju_mulQuat(difference, data.xquat[body], orientation * [1, -1, -1, -1])
        rotation = np.zeros(3)
        mujoco.mju_quat2Vel(rotation, difference, 1.)
        error = float(np.linalg.norm(data.xpos[body] - target))
        angle = float(np.linalg.norm(rotation))
        if error >= 2e-5 or angle >= 1e-4:
            raise ValueError(f'Hand IK failed at {times[index]}: {error}, {angle}')
        assert np.array_equal(changed[other], original[other])
        result['qpos'][index] = changed
        result['errors'][index] = max(float(result['errors'][index]), error, .2 * angle)
        diagnostics.append(dict(index=int(index), time_s=float(times[index]),
            weight=float(weights[index]), position_error_m=error, orientation_error_rad=angle))
    assert np.array_equal(result['qpos'][times <= times[first]], plan['qpos'][times <= times[first]])
    assert all(np.array_equal(v, result[k]) for k, v in plan.items() if k not in ('qpos', 'errors'))
    return result, dict(identity=False, hand=key, offset_m=offset.tolist(),
        interval_s=[float(times[first]), float(times[last])], changed_nodes=len(diagnostics),
        diagnostics=diagnostics, scope='Offline knots only; support schedule unchanged; requires dense path and dynamics audits')
