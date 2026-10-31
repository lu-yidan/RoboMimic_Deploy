"""Deterministic Cartesian lift refinement for a frozen whole-cycle plan.

This is distinct from clearance_cycle's initial unload foot_lift: it lifts only
the airborne left-foot path. Timing, base, other limbs and endpoint stay fixed.
"""
import mujoco
import numpy as np


def refine_swing_foot_lift(env, plan, *, foot_lift=.008, swing_start=53.7,
                          swing_end=82.26, envelope='sine', taper_start=77.56,
                          taper_end=80.66, joint_margin=.018):
    if not (0 <= foot_lift <= .02 and 0 <= swing_start < swing_end <= plan['time'][-1]):
        raise ValueError('Invalid swing lift or interval')
    if envelope not in ('sine', 'landing_taper'):
        raise ValueError('Unknown swing lift envelope')
    if envelope == 'landing_taper' and not swing_start < taper_start < taper_end <= swing_end:
        raise ValueError('Taper must lie inside the swing gap')
    m = env.model
    leg = [i for i,j in enumerate(env.joints) if m.joint(int(j)).name.startswith('left_')
           and any(k in m.joint(int(j)).name for k in ('hip','knee','ankle'))]
    if len(leg) != 6:
        raise ValueError('Expected six left-leg joints')
    qa, va = np.asarray(env.qadr)[leg], np.asarray(env.vadr)[leg]
    lo, hi = m.jnt_range[np.asarray(env.joints)[leg]].T
    foot = m.body('left_ankle_roll_link').id
    d = mujoco.MjData(m)
    def position(q):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        return d.xpos[foot].copy()
    # Snap endpoints to actual knots to reproduce the diagnostic's floating-point
    # times exactly; never move the first or last airborne knot.
    times = plan['time']
    first, last = [int(np.argmin(abs(times-t))) for t in (swing_start,swing_end)]
    if max(abs(times[first]-swing_start),abs(times[last]-swing_end)) > 1e-7:
        raise ValueError('Swing interval endpoints must be plan knots')
    start, end = float(times[first]), float(times[last])
    if plan['active'].shape != (len(times),4) or np.any(plan['active'][first:last+1,0]):
        raise ValueError('Lift interval must keep the left foot inactive')
    result = {k:v.copy() for k,v in plan.items()}
    weights = np.zeros(len(times))
    residuals = []
    for i in range(first+1,last):
        t = float(times[i])
        weight = float(np.sin(np.pi*(t-start)/(end-start)))
        if envelope == 'landing_taper':
            u = float(np.clip((t-taper_start)/(taper_end-taper_start),0,1))
            weight *= 1-u*u*u*(10+u*(-15+6*u))
        weights[i] = weight
        if foot_lift == 0 or weight <= 1e-9:
            continue
        q = plan['qpos'][i].copy()
        target = position(q)+np.array([0.,0.,foot_lift*weight])
        for _ in range(10):
            error = target-position(q)
            if np.linalg.norm(error) < 2e-5:
                break
            jac = np.zeros((3,m.nv))
            mujoco.mj_jacBody(m,d,jac,None,foot)
            jac = jac[:,va]
            dq = jac.T @ np.linalg.inv(jac @ jac.T+1e-6*np.eye(3)) @ error
            q[qa] = np.clip(q[qa]+dq.ravel(),lo+joint_margin,hi-joint_margin)
        residual = float(np.linalg.norm(target-position(q)))
        if residual >= 2e-5:
            raise ValueError(f'Lift IK failed at {t}: {residual}')
        result['qpos'][i] = q
        result['errors'][i] = max(float(result['errors'][i]),residual)
        residuals.append(residual)
    untouched = np.ones(m.nq,dtype=bool); untouched[qa] = False
    checks = dict(other_dofs_unchanged=np.array_equal(result['qpos'][:,untouched],plan['qpos'][:,untouched]),
                  outside_window_unchanged=np.array_equal(result['qpos'][weights<=1e-9],plan['qpos'][weights<=1e-9]),
                  timing_and_contacts_unchanged=all(np.array_equal(result[k],v) for k,v in plan.items() if k not in ('qpos','errors')),
                  endpoints_unchanged=np.array_equal(result['qpos'][[0,-1]],plan['qpos'][[0,-1]]))
    if not all(checks.values()):
        raise AssertionError(checks)
    return result, dict(foot_lift_m=foot_lift,swing_interval_s=[start,end],envelope=envelope,
                       taper_start_s=taper_start if envelope=='landing_taper' else None,
                       taper_end_s=taper_end if envelope=='landing_taper' else None,
                       joint_margin_rad=joint_margin,checks=checks,
                       maximum_ik_residual_m=max(residuals,default=0.),
                       changed_nodes=int(np.any(result['qpos']!=plan['qpos'],axis=1).sum()))
