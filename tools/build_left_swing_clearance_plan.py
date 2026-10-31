"""Offline, fixed-amplitude left-foot lift; never edits a running physical state."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from training.climb_cycle import LimbIK
from tools.compare_audited_closure_candidates import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--failure', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    manifest = json.loads((args.failure/'manifest.json').read_text())
    assert sha(args.parent) == manifest['reference_sha256']
    cases = {'nominal': {}, 'foot_z_plus15': {'rung_01': [0., 0., .015]},
             'fresh_boundary_03': manifest['evaluation_cases']['fresh_boundary_03']}
    env = FullCycleResidualEnv(plan=args.parent, cases=cases)
    env.reset(options={'case': 'nominal'})
    robot = env.env
    ik = LimbIK(robot)
    plan = {k: v.copy() for k, v in env.plan.items()}
    original = plan['qpos'].copy()
    start, end, amplitude = 75., 80.66, .005
    records = []
    try:
        for i, time in enumerate(plan['time']):
            if not start < time < end:
                continue
            q = original[i]
            ik.d.qpos[:] = q
            mujoco.mj_kinematics(ik.m, ik.d)
            target = ik.d.xpos[ik.bodies['lf']].copy()
            orientation = ik.d.xquat[ik.bodies['lf']].copy()
            u = (float(time)-start)/(end-start)
            lift = amplitude*64*u**3*(1-u)**3
            target[2] += lift
            result, errors = ik.pose(q, {'lf': target}, orientations={'lf': orientation})
            assert errors['lf'] < 1e-6, errors
            plan['qpos'][i] = result
            records.append(dict(time_s=float(time), requested_lift_m=lift, ik_error=errors['lf']))
        qa = robot.qadr[:6]
        untouched = np.ones(original.shape[1], dtype=bool)
        untouched[qa] = False
        np.testing.assert_array_equal(plan['qpos'][:, untouched], original[:, untouched])
        outside = (plan['time'] <= start) | (plan['time'] >= end)
        np.testing.assert_array_equal(plan['qpos'][outside], original[outside])
        joint_margin = float(min(np.min(plan['qpos'][:, robot.qadr]-robot.model.jnt_range[robot.joints, 0]),
                                 np.min(robot.model.jnt_range[robot.joints, 1]-plan['qpos'][:, robot.qadr])))
        assert joint_margin >= -1e-7
        np.savez_compressed(args.output/'plan.npz', **plan)
        checks = []
        times = np.arange(start, end+1e-8, .005)
        data = mujoco.MjData(robot.model)
        sole = robot.model.geom('left_sole_surface').id
        rod = robot.model.geom('rung_01').id
        for case in cases:
            env.reset(options={'case': case})
            distances = []
            new_collisions = []
            for time in times:
                j = min(max(int(np.searchsorted(plan['time'], time, side='right')-1), 0), len(plan['time'])-2)
                u = (time-plan['time'][j])/(plan['time'][j+1]-plan['time'][j])
                data.qpos[:] = (1-u)*plan['qpos'][j]+u*plan['qpos'][j+1]
                mujoco.mj_kinematics(robot.model, data)
                distances.append(float(mujoco.mj_geomDistance(robot.model, data, sole, rod, 1., None)))
                mujoco.mj_collision(robot.model, data)
                for contact in data.contact:
                    a, b = int(contact.geom1), int(contact.geom2)
                    ba, bb = int(robot.model.geom_bodyid[a]), int(robot.model.geom_bodyid[b])
                    ina, inb = ba in robot.robot_bodies, bb in robot.robot_bodies
                    if not (ina or inb):
                        continue
                    expected = ina != inb and (ba if ina else bb) in robot.contact_groups
                    # Existing support contacts retain the original offline tolerance.
                    tolerance = .0007 if expected else .00001
                    if contact.dist < -tolerance:
                        new_collisions.append(dict(time_s=float(time), geoms=[robot.model.geom(a).name, robot.model.geom(b).name], distance_m=float(contact.dist)))
            checks.append(dict(case=case, minimum_sole_target_clearance_m=min(distances),
                               minimum_time_s=float(times[np.argmin(distances)]), collisions=new_collisions,
                               passed=min(distances) >= .003 and not new_collisions))
        result = dict(passed=all(r['passed'] for r in checks), cases=cases, checks=checks,
                      parent=str(args.parent.resolve()), parent_sha256=sha(args.parent),
                      plan_sha256=sha(args.output/'plan.npz'), source_sha256=sha(__file__),
                      start_s=start, end_s=end, peak_lift_m=amplitude, node_checks=records,
                      joint_margin_rad=joint_margin, max_joint_change_rad=float(np.max(abs(plan['qpos']-original))),
                      unchanged_outside_window=True, unchanged_other_joints_and_base=True,
                      scope='Fixed 5 mm lift chosen before evaluation; dense 5 ms offline geometry in the modified window only. Same phase clocks and endpoints. Inherited plan errors are historical, new IK errors are node_checks. No dynamic execution, torque feasibility or training claim. Previously observed failure is a development case.')
        (args.output/'manifest.json').write_text(json.dumps(result, indent=2))
        (args.output/'builder_source.py').write_bytes(Path(__file__).read_bytes())
        print(json.dumps({k:v for k,v in result.items() if k not in ('node_checks','checks')}))
        print(json.dumps([dict(case=r['case'], passed=r['passed'], clearance_m=r['minimum_sole_target_clearance_m'], collisions=len(r['collisions'])) for r in checks]))
    finally:
        env.close()


if __name__ == '__main__':
    main()
