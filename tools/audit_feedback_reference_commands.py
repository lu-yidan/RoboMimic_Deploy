"""Reconstruct feedback observations, decisions, holds and controller references."""
import numpy as np


def feedback_checks(manifest, arrays, policy):
    trace = arrays['adaptive_reference_trace']
    obs, actions = arrays['feedback_observations'], arrays['feedback_actions']
    indices = arrays['feedback_teacher_indices']
    count = len(arrays['actions'])
    expected_indices = np.arange(round(manifest['suffix_start_seconds']/.02), count,
                                 manifest['feedback_hold_steps'], dtype=np.int64)
    schema = (obs.shape == (len(indices), 153) and actions.shape == (len(indices), 5)
              and indices.shape == expected_indices.shape and np.array_equal(indices, expected_indices)
              and trace.ndim == 2 and trace.shape[1] == 21 and len(trace) > 0
              and arrays['feedback_time_s'].shape == indices.shape)
    checks = dict(feedback_schema=bool(schema))
    if not schema:
        return checks, {}
    scales = np.array([.003, .003, .003, .00015, .001])
    assert manifest['adaptive_reference_scales'] == scales.tolist()
    previous_control = indices*4-1
    valid = bool(np.all(previous_control >= 0) and np.all(previous_control < len(trace)))
    checks['feedback_prior_control_indices'] = valid
    if not valid:
        return checks, {}
    expected_obs = np.c_[arrays['observations'][indices], trace[previous_control, 11:16]/scales].astype(np.float32)
    replayed = np.asarray([policy.predict(row, deterministic=True)[0] for row in obs], dtype=np.float32).reshape(-1, 5)
    action_error = float(abs(replayed-actions).max()) if len(actions) else 0.
    seed = np.asarray(manifest['adaptive_reference_command'], dtype=float)
    mapped = np.clip(seed+manifest['feedback_delta_scale']*actions.astype(float), -1., 1.)
    commands = np.broadcast_to(seed, (len(trace), 5)).copy()
    control_indices = np.rint(trace[:, 0]/.005).astype(np.int64)
    held_index = np.searchsorted(indices*4, control_indices, side='right')-1
    active = held_index >= 0
    commands[active] = mapped[held_index[active]]
    geometry = arrays['observations'][0, 138]
    gain = float(np.clip((float(geometry)-.625)/.125, 0., 1.))
    u = np.clip((trace[:, 0]-99.66)/3., 0., 1.)
    ramp = u*u*(3-2*u)
    target = commands*scales*gain*ramp[:, None]
    residual = np.empty_like(target)
    previous = np.zeros(5)
    for i, value in enumerate(target):
        previous = previous+.05*(value-previous)
        residual[i] = previous
    velocity = np.diff(np.vstack([np.zeros(5), residual]), axis=0)/.005
    fixed = arrays['waist_reference_trace']
    waist = np.zeros((len(trace), 3))
    if len(fixed):
        fixed_indices = np.rint(fixed[:, 0]/.005).astype(np.int64)
        if not (np.all(fixed_indices >= 0) and np.all(fixed_indices < len(trace))):
            checks['feedback_fixed_trace_indices'] = False
            return checks, {}
        waist[fixed_indices] = fixed[:, 4:7]
    total_waist = waist+residual[:, :3]
    total_height = -gain*manifest['controller_modifier']['body_drop_m']*ramp+residual[:, 3]
    checks.update(
        feedback_finite=bool(all(np.isfinite(x).all() for x in (obs, actions, trace))),
        feedback_observations=bool(np.array_equal(obs, expected_obs)),
        feedback_time=bool(np.array_equal(arrays['feedback_time_s'], arrays['time_s'][indices])),
        feedback_action_bounds=bool(not len(actions) or abs(actions).max() <= 1.),
        feedback_actor_replay=action_error == 0.,
        feedback_command_hold=bool(np.array_equal(trace[:, 1:6], commands)),
        feedback_controller_start=abs(float(trace[0, 0])) < 1e-12,
        feedback_controller_cadence=bool(np.allclose(np.diff(trace[:, 0]), .005, atol=1e-7, rtol=0)),
        feedback_controller_coverage=bool(-1e-7 <= float(arrays['time_s'][-1]-trace[-1, 0]) <= .0050001),
        feedback_target=bool(np.allclose(trace[:, 6:11], target, atol=1e-15, rtol=0)),
        feedback_filter=bool(np.allclose(trace[:, 11:16], residual, atol=1e-15, rtol=0)),
        feedback_velocity=bool(np.allclose(trace[:, 16:21], velocity, atol=1e-13, rtol=0)),
        feedback_geometry_gain=gain == float(arrays['controller_geometry_gain']),
        feedback_foreaft_bound=bool(abs(residual[:, 4]).max() <= .001+1e-12),
        feedback_total_waist_bound=bool(abs(total_waist).max() <= .015+1e-12),
        feedback_total_body_bound=bool(total_height.max() <= 1e-12 and total_height.min() >= -.015-1e-12),
        feedback_inactive_before_closure=bool(np.array_equal(residual[trace[:, 0] <= 99.66],
                                           np.zeros_like(residual[trace[:, 0] <= 99.66]))))
    differences = dict(action=action_error, target=float(abs(trace[:, 6:11]-target).max()),
                       filter=float(abs(trace[:, 11:16]-residual).max()),
                       velocity=float(abs(trace[:, 16:21]-velocity).max()))
    return checks, differences
