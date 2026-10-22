"""Frozen-state residual-to-torque/acceleration diagnostic; no simulation steps.

Reference offsets are steady biases, so their reference velocity is zero.
This is a local counterfactual at saved states, not endpoint optimization,
trajectory success, or evidence that an instantaneous response persists.
"""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from training.full_cycle_residual_env import FullCycleResidualEnv, ROOT
from training.contact_frame_controller import ContactFrameController


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.experiment/'manifest.json').read_text())
    audit = json.loads((args.experiment/'independent_training_audit.json').read_text())
    if not audit['passed'] or sha(args.experiment/'manifest.json') != audit['experiment_manifest_sha256']:
        raise ValueError('Audited matching experiment required')
    for relative, expected in manifest['sources'].items():
        if relative.replace('\\', '/').startswith('training/') and sha(ROOT/relative) != expected:
            raise ValueError('Frozen environment changed')
    args.output.mkdir(parents=True, exist_ok=False)
    cases = manifest['training_cases']
    env = FullCycleResidualEnv(plan=manifest['reference'], cases=cases)
    results = []
    for case in cases:
        env.reset(options={'case': case})
        e = env.env
        path = args.experiment/f'zero_{case}.npz'
        evidence = next(row for row in audit['trajectories'] if row['label'] == 'zero' and row['case'] == case)
        if sha(path) != evidence['trajectory_sha256']:
            raise ValueError('Zero trajectory changed')
        with np.load(path) as z:
            i = int(np.argmin(abs(z['time_s']-109.)))
            pose, velocity, t = z['qpos'][i].copy(), z['qvel'][i].copy(), float(z['time_s'][i])
        qref, vref, active, phase = env.reference(t)
        if phase != 'closed_hold' or not np.all(active):
            raise ValueError('Steady four-support closed-hold state required')
        e.data.qpos[:], e.data.qvel[:], e.data.time = pose, velocity, t
        mujoco.mj_forward(e.model, e.data)
        foot = e.model.body('right_ankle_roll_link').id
        jacobian = np.zeros((3, e.model.nv))
        mujoco.mj_jacBody(e.model, e.data, jacobian, None, foot)
        scratch = mujoco.MjData(e.model)
        scratch.qpos[:] = env.plan['qpos'][-1]
        mujoco.mj_kinematics(e.model, scratch)
        error = e.data.xpos[foot]-scratch.xpos[foot]
        toward = -error/max(float(np.linalg.norm(error)), 1e-12)

        def response(offset):
            # Only scratch diagnostic state is restored; no live episode exists.
            mujoco.mj_resetData(e.model, e.data)
            e.data.qpos[:], e.data.qvel[:], e.data.time = pose, velocity, t
            mujoco.mj_forward(e.model, e.data)
            controller = ContactFrameController(e, joint_kp=100., leg_kp=300., torque_weight=10000.)
            target = qref.copy()
            target[e.qadr[env.indices]] += offset
            torque, details = controller.command(target, vref, active)
            e.data.ctrl[e.motors] = torque
            mujoco.mj_forward(e.model, e.data)
            acceleration = e.data.qacc.copy()
            if not np.array_equal(e.data.qpos, pose) or e.data.time != t:
                raise AssertionError('Frozen diagnostic unexpectedly advanced physics')
            return torque, jacobian@acceleration, np.array(details['force_plan'])

        base_torque, base_acc, base_force = response(np.zeros(26))
        rows = []
        derivatives = []
        for j, joint in enumerate(env.indices):
            step = .003
            offset = np.zeros(26)
            offset[j] = step
            positive = response(offset)
            negative = response(-offset)
            derivative = (positive[1]-negative[1])/(2*step)
            derivatives.append(derivative)
            rows.append(dict(joint=e.model.joint(e.joints[joint]).name, steady_reference_bias_rad=step,
                positive_max_torque_change_nm=float(abs(positive[0]-base_torque).max()),
                negative_max_torque_change_nm=float(abs(negative[0]-base_torque).max()),
                positive_right_foot_acceleration_change_m_s2=(positive[1]-base_acc).tolist(),
                negative_right_foot_acceleration_change_m_s2=(negative[1]-base_acc).tolist(),
                positive_acceleration_toward_nominal_m_s2=float(toward@(positive[1]-base_acc)),
                negative_acceleration_toward_nominal_m_s2=float(toward@(negative[1]-base_acc)),
                positive_force_plan_change_norm_n=float(np.linalg.norm(positive[2]-base_force)),
                negative_force_plan_change_norm_n=float(np.linalg.norm(negative[2]-base_force))))
        results.append(dict(case=case, state_time_s=t, phase=phase, zero_trajectory_sha256=sha(path),
            right_foot_world_error_xyz_m=error.tolist(), foot_acceleration_jacobian_singular_values=np.linalg.svd(np.array(derivatives).T, compute_uv=False).tolist(),
            finite=all(np.isfinite(np.array(derivatives)).ravel()), responses=rows))
        print(json.dumps(dict(case=case, completed_joint_probes=len(rows))), flush=True)
    report = dict(results=results, parent_audit_sha256=sha(args.experiment/'independent_training_audit.json'),
        source_sha256=sha(__file__), controller_sha256=sha(ROOT/'training/contact_frame_controller.py'),
        interpretation='Finite steady-reference perturbations at fixed recorded states. Acceleration differences omit the common Jdot*qvel term, which cancels. No policy-step filtering transient, state rollout, endpoint success or hardware response is implied.')
    (args.output/'summary.json').write_text(json.dumps(report, indent=2))
    (args.output/'probe_source.py').write_bytes(Path(__file__).read_bytes())
    env.close()


if __name__ == '__main__':
    main()
