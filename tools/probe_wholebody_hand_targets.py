"""Bounded whole-body scratch IK near the hand-only retarget failure."""
import argparse
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from training.wall_support_env import WallSupportEnv
from tools.probe_hand_reference_path import write, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan_path = Path('outputs/v2_swing_two_reference_20261029_v1/plan.npz')
    with np.load(plan_path, allow_pickle=False) as z:
        times, qs, phases = z['time'], z['qpos'], z['phase']
    requested = [159.96, 164.06, 164.86, 165.66, 170.06]
    cases = [{'name': f'x{i}_z{j}', 'offset_m': [float(x), 0., float(z)]}
             for i, x in enumerate(np.linspace(-.002, .002, 4))
             for j, z in enumerate(np.linspace(-.002, .002, 4))]
    protocol = dict(plan_sha256=sha(plan_path), source_sha256=sha(__file__),
        cases=cases, requested_times=requested, maximum_root_translation_norm_m=.003,
        maximum_joint_change_rad=.15, maximum_evaluations=400,
        position_tolerance_m=2e-5, orientation_tolerance_rad=1e-4,
        scope='Offline independent knots only. Both hook body targets displaced; both foot body positions and all four orientations preserved. Base quaternion fixed. No collision, dynamics, image identity, trajectory continuity or controller acceptance.')
    write(args.output / 'protocol.json', protocol)
    (args.output / 'probe_source.py').write_bytes(Path(__file__).read_bytes())
    env = WallSupportEnv('outputs/wall_support_bundle_pd15_v2')
    env.reset()
    m, d = env.model, mujoco.MjData(env.model)
    bodies = [m.body(n).id for n in ('hook', 'right_hook', 'left_ankle_roll_link', 'right_ankle_roll_link')]
    before = (env.data.qpos.copy(), env.data.qvel.copy(), float(env.data.time))
    results = []
    try:
        for case in cases:
            records, candidates = [], []
            for requested_time in requested:
                index = int(np.argmin(abs(times-requested_time)))
                original = qs[index].copy()
                d.qpos[:] = original
                mujoco.mj_kinematics(m, d)
                targets, orientations = d.xpos[bodies].copy(), d.xquat[bodies].copy()
                targets[:2] += np.asarray(case['offset_m'])
                lower = np.r_[np.full(3, -.003/np.sqrt(3)),
                    np.maximum(-.15, m.jnt_range[env.joints, 0]-original[env.qadr])]
                upper = np.r_[np.full(3, .003/np.sqrt(3)),
                    np.minimum(.15, m.jnt_range[env.joints, 1]-original[env.qadr])]
                def errors(x):
                    d.qpos[:] = original
                    d.qpos[:3] += x[:3]
                    d.qpos[env.qadr] += x[3:]
                    mujoco.mj_kinematics(m, d)
                    rotation = []
                    for body, orientation in zip(bodies, orientations):
                        difference, vector = np.zeros(4), np.zeros(3)
                        mujoco.mju_mulQuat(difference, d.xquat[body], orientation*[1, -1, -1, -1])
                        mujoco.mju_quat2Vel(vector, difference, 1.)
                        rotation.append(vector)
                    return d.xpos[bodies]-targets, np.asarray(rotation)
                def objective(x):
                    position, rotation = errors(x)
                    return np.r_[position.ravel(), .2*rotation.ravel(), 1e-5*x]
                fit = least_squares(objective, np.clip(np.zeros(len(lower)), lower+1e-10, upper-1e-10),
                    bounds=(lower, upper), max_nfev=400, ftol=1e-10, xtol=1e-10, gtol=1e-10)
                position, rotation = errors(fit.x)
                pe = float(np.linalg.norm(position, axis=1).max())
                re = float(np.linalg.norm(rotation, axis=1).max())
                root_norm = float(np.linalg.norm(fit.x[:3]))
                joint_max = float(abs(fit.x[3:]).max())
                passed = pe < 2e-5 and re < 1e-4 and root_norm <= .003+1e-12 and joint_max <= .15+1e-12
                records.append(dict(time_s=float(times[index]), phase=str(phases[index]),
                    position_error_m=pe, orientation_error_rad=re, root_delta_m=fit.x[:3].tolist(),
                    root_delta_norm_m=root_norm, maximum_joint_delta_rad=joint_max,
                    solver_status=fit.status, evaluations=fit.nfev, passed=bool(passed)))
                candidates.append(d.qpos.copy())
            artifact = args.output / f"{case['name']}_knots.npz"
            np.savez_compressed(artifact, qpos=np.asarray(candidates), requested_times=requested)
            row = dict(**case, records=records, passed=all(r['passed'] for r in records),
                       artifact_sha256=sha(artifact))
            write(args.output / f"{case['name']}.json", row)
            results.append(row)
            print(f"{case['name']}: {sum(r['passed'] for r in records)}/{len(records)} local poses", flush=True)
        assert np.array_equal(env.data.qpos, before[0]) and np.array_equal(env.data.qvel, before[1]) and env.data.time == before[2]
        assert sha(plan_path) == protocol['plan_sha256']
        write(args.output / 'summary.json', dict(declared_cases=16, poses_per_case=5,
            passed_cases=sum(r['passed'] for r in results), passed_poses=sum(t['passed'] for r in results for t in r['records']),
            cases=results, live_state_unchanged=True, protocol_sha256=sha(args.output/'protocol.json'), scope=protocol['scope']))
    finally:
        env.close()


if __name__ == '__main__':
    main()
