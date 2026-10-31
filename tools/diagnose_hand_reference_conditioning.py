"""Inspect local hand IK conditioning at the preserved failed reference pose."""
import argparse
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from training.wall_support_env import WallSupportEnv
from training.climb_cycle import LimbIK
from tools.probe_hand_reference_path import write, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    plan_path = Path('outputs/v2_swing_two_reference_20261029_v1/plan.npz')
    with np.load(plan_path, allow_pickle=False) as z:
        times, qs, phases = z['time'], z['qpos'], z['phase']
    env = WallSupportEnv('outputs/wall_support_bundle_pd15_v2')
    env.reset()
    ik = LimbIK(env)
    m, d = env.model, ik.d
    qa, va = env.qadr[ik.slices['lh']], env.vadr[ik.slices['lh']]
    joints, body = env.joints[ik.slices['lh']], ik.bodies['lh']
    requested_times = [138.66, 140.66, 159.96, 164.06, 164.86, 165.66, 170.06, 220.02]
    protocol = dict(plan_sha256=sha(plan_path), source_sha256=sha(__file__),
        ik_source_sha256=sha('training/climb_cycle.py'), hand='lh',
        requested_times=requested_times, failure_time_s=164.86,
        target_offset_m=[.002, 0., .002], orientation_weight_m_per_rad=.2,
        finite_difference_budgets=[80, 400, 2000],
        scope='Local kinematic conditioning and one-pose optimization only. No proof of global infeasibility; no dynamics or controller changes.')
    write(args.output / 'protocol.json', protocol)
    try:
        rows = []
        for requested in requested_times:
            index = int(np.argmin(abs(times-requested)))
            d.qpos[:] = qs[index]
            mujoco.mj_forward(m, d)
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jp, jr, body)
            jac = np.vstack((jp[:, va], .2*jr[:, va]))
            values = np.linalg.svd(jac, compute_uv=False)
            delta = np.linalg.lstsq(jac, np.array([.002, 0, .002, 0, 0, 0]), rcond=None)[0]
            rows.append(dict(time_s=float(times[index]), phase=str(phases[index]),
                scaled_singular_values=values.tolist(), linear_joint_delta_max_rad=float(abs(delta).max())))
        index = int(np.argmin(abs(times-164.86)))
        q = qs[index].copy()
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        target, orientation = d.xpos[body].copy() + [.002, 0, .002], d.xquat[body].copy()
        prior = q[qa].copy()
        def residual(x):
            d.qpos[:] = q
            d.qpos[qa] = x
            mujoco.mj_kinematics(m, d)
            difference, rotation = np.zeros(4), np.zeros(3)
            mujoco.mju_mulQuat(difference, d.xquat[body], orientation*[1, -1, -1, -1])
            mujoco.mju_quat2Vel(rotation, difference, 1.)
            return np.r_[d.xpos[body]-target, .2*rotation, 1e-5*(x-prior)]
        solves = []
        for budget in protocol['finite_difference_budgets']:
            fit = least_squares(residual, prior, bounds=(m.jnt_range[joints, 0], m.jnt_range[joints, 1]),
                max_nfev=budget, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            r = residual(fit.x)
            solves.append(dict(budget=budget, nfev=fit.nfev, status=fit.status, message=fit.message,
                position_error_m=float(np.linalg.norm(r[:3])), rotation_error_rad=float(np.linalg.norm(r[3:6])/.2),
                maximum_joint_delta_rad=float(abs(fit.x-prior).max()), active_mask=fit.active_mask.tolist()))
        write(args.output / 'summary.json', dict(protocol_sha256=sha(args.output/'protocol.json'),
            jacobian_samples=rows, finite_difference_solves=solves,
            local_fit_meets_original_tolerances=any(r['position_error_m']<2e-5 and r['rotation_error_rad']<1e-4 for r in solves),
            scope=protocol['scope']))
        (args.output / 'analysis_source.py').write_bytes(Path(__file__).read_bytes())
        print('Conditioning diagnosis saved; original feasibility thresholds retained.', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
