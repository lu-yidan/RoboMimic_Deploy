"""Save and audit an offline two-hand displacement candidate, never promote it."""
import argparse
from pathlib import Path
import mujoco
import numpy as np
from training.compiled_rung_geometry import CompiledPhysicalV2CycleEnv
from training.hand_target_reference import retarget_hand_reference
from tools.plan_clearance_cycle import audit_plan
from tools.probe_hand_reference_path import read, write, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path('outputs/v2_swing_two_reference_20261029_v1/plan.npz')
    parent = read(source.parent / 'manifest.json')
    assert sha(source) == parent['plan_sha256'] and read(source.parent / 'audit.json')['passed']
    offset = [.002, 0., .002]
    windows = {'rh': [123.66, 125.66], 'lh': [136.66, 138.66]}
    protocol = dict(parent=str(source), parent_sha256=sha(source), offset_m=offset,
        windows_s=windows, geometry={'rung_06': offset}, other_rods='nominal, not the T12 foot-offset scene',
        candidate_selection='Predeclared positive X/Z corner for an implementation pilot; no dynamic grid outcome selection',
        source_hashes={p: sha(p) for p in ('training/hand_target_reference.py',
            'training/climb_cycle.py', 'training/compiled_rung_geometry.py',
            'tools/build_hand_reference_candidate.py', 'tools/plan_clearance_cycle.py')},
        dense_sample_dt_s=.005, interpolated_position_tolerance_m=.00002,
        interpolated_rotation_tolerance_rad=.0001, additional_joint_speed_tolerance_rad_s=.1,
        scope='Offline reference and collision samples only. Not a formal plan; no prefix/policy/dynamic acceptance, no image association, no training.')
    write(args.output / 'protocol.json', protocol)
    base = CompiledPhysicalV2CycleEnv(plan=source, cases={'target_shift': {'rung_06': offset}})
    try:
        base.reset(options={'case': 'target_shift'})
        e = base.env
        original = {k: v.copy() for k, v in base.plan.items()}
        result = original
        details = {}
        for key, (start, end) in windows.items():
            result, details[key] = retarget_hand_reference(e, result, key=key,
                                      offset_m=offset, start_s=start, end_s=end)
        np.savez_compressed(args.output / 'candidate_knots.npz', **result)
        write(args.output / 'node_diagnostics.json', details)
        # The same compiled shifted geometry is used for original and candidate.
        baseline_audit = audit_plan(e, original, dt=.005)
        write(args.output / 'original_geometry_audit.json', baseline_audit)
        candidate_audit = audit_plan(e, result, dt=.005)
        write(args.output / 'candidate_geometry_audit.json', candidate_audit)
        data = mujoco.MjData(e.model)
        bodies = [e.model.body(name).id for name in ('right_hook', 'hook')]
        max_position, max_rotation, samples = 0., 0., 0
        times = original['time']
        for t in np.arange(123.66, times[-1] + 1e-8, .005):
            index = min(max(int(np.searchsorted(times, t, side='right') - 1), 0), len(times) - 2)
            u = float(np.clip((t - times[index]) / (times[index+1] - times[index]), 0., 1.))
            q0 = (1-u)*original['qpos'][index] + u*original['qpos'][index+1]
            q1 = (1-u)*result['qpos'][index] + u*result['qpos'][index+1]
            data.qpos[:] = q0
            mujoco.mj_kinematics(e.model, data)
            positions, rotations = data.xpos[bodies].copy(), data.xmat[bodies].copy()
            data.qpos[:] = q1
            mujoco.mj_kinematics(e.model, data)
            for hand, body, position, rotation in zip(windows, bodies, positions, rotations):
                start, end = windows[hand]
                fraction = float(np.clip((t-start)/(end-start), 0., 1.))
                weight = fraction**3*(10 + fraction*(-15+6*fraction))
                max_position = max(max_position, float(np.linalg.norm(data.xpos[body] - position - weight*np.asarray(offset))))
                angle = float(np.arccos(np.clip((np.trace(rotation.reshape(3, 3).T @ data.xmat[body].reshape(3, 3)) - 1)/2, -1., 1.)))
                max_rotation = max(max_rotation, angle)
            samples += 1
        velocity_delta = np.diff(result['qpos'][:, e.qadr] - original['qpos'][:, e.qadr], axis=0) / np.diff(times)[:, None]
        prefix_exact = all(np.array_equal(result[k], v) for k, v in original.items() if k not in ('qpos', 'errors')) and np.array_equal(result['qpos'][times <= 123.66], original['qpos'][times <= 123.66])
        max_speed = float(abs(velocity_delta).max())
        checks = dict(prefix_and_schedule_exact=prefix_exact,
            dense_collision_and_limits=candidate_audit['passed'],
            interpolated_position=max_position < .00002,
            interpolated_orientation=max_rotation < .0001,
            additional_joint_speed=max_speed < .1,
            original_sources_unchanged=all(sha(p) == h for p, h in protocol['source_hashes'].items()))
        summary = dict(passed=all(checks.values()), checks=checks,
            interpolated_samples=samples, max_interpolated_position_error_m=max_position,
            max_interpolated_rotation_error_rad=max_rotation,
            maximum_additional_joint_speed_rad_s=max_speed,
            original_collision_failures=len(baseline_audit['failures']),
            candidate_collision_failures=len(candidate_audit['failures']),
            candidate_sha256=sha(args.output / 'candidate_knots.npz'),
            protocol_sha256=sha(args.output / 'protocol.json'), scope=protocol['scope'])
        write(args.output / 'summary.json', summary)
        print(summary, flush=True)
    finally:
        base.close()


if __name__ == '__main__':
    main()
