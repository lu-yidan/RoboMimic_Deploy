"""Offline hand-target IK probe; no controller, scene or adopted-plan edits."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from training.compiled_rung_geometry import CompiledPhysicalV2CycleEnv
from training.climb_cycle import LimbIK


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    parent_run = Path('outputs/hook_admission_T12_two_20261029_v1')
    manifest = read(parent_run / 'manifest.json')
    plan_path = Path(manifest['plan_path'])
    assert sha(plan_path) == manifest['plan_sha256']
    parent = read(Path(manifest['parent_experiment']) / 'manifest.json')
    assert sha(Path(manifest['parent_experiment']) / 'manifest.json') == manifest['parent_manifest_sha256']
    with np.load(plan_path, allow_pickle=False) as saved:
        plan = {k: saved[k] for k in saved.files}
    phases = ('rh_approach', 'rh_seat', 'lh_approach', 'lh_seat')
    indices = np.flatnonzero((plan['time'] > plan['cycle_ends_s'][0]) & np.isin(plan['phase'], phases))
    cases = [{'name': f'x{i}_z{j}', 'offset_m': [float(x), 0., float(z)]}
             for i, x in enumerate(np.linspace(-.002, .002, 4))
             for j, z in enumerate(np.linspace(-.002, .002, 4))]
    protocol = dict(plan_sha256=sha(plan_path), samples=indices.tolist(), cases=cases,
                    source_sha256=sha(__file__), ik_source_sha256=sha('training/climb_cycle.py'),
                    position_tolerance_m=.00002, orientation_tolerance_rad=.0001,
                    scope='Independent saved reference knots, second-cycle hand approach/seat, supplied 2 mm X/Z target offsets. Preserve hook orientation and all other DOFs. No image association, trajectory smoothing, collision acceptance, dynamics, policy action or promotion.')
    write(args.output / 'protocol.json', protocol)
    (args.output / 'probe_source.py').write_bytes(Path(__file__).read_bytes())
    base = CompiledPhysicalV2CycleEnv(plan=str(plan_path), cases={manifest['case']: manifest['offsets']},
              normal_bias_y_deg=parent['normal_bias_y_deg'], memory_enabled=parent['force_history_memory'])
    rows = []
    try:
        base.reset(options={'case': manifest['case']})
        e = base.env
        live_before = (e.data.qpos.copy(), e.data.qvel.copy(), float(e.data.time))
        ik = LimbIK(e)
        d = mujoco.MjData(e.model)
        for case in cases:
            records, candidates = [], []
            offset = np.asarray(case['offset_m'])
            for index in indices:
                original = plan['qpos'][index].copy()
                key = str(plan['phase'][index])[:2]
                group = 3 if key == 'rh' else 2
                assert not plan['active'][index, group], 'Probe restricted to unloaded hand knots'
                d.qpos[:] = original
                mujoco.mj_kinematics(e.model, d)
                body = ik.bodies[key]
                position, orientation = d.xpos[body].copy(), d.xquat[body].copy()
                other_bodies = [v for k, v in ik.bodies.items() if k != key]
                other_positions = d.xpos[other_bodies].copy()
                changed, _ = ik.pose(original, {key: position + offset}, orientations={key: orientation})
                d.qpos[:] = changed
                mujoco.mj_kinematics(e.model, d)
                difference = np.zeros(4)
                mujoco.mju_mulQuat(difference, d.xquat[body], orientation * [1, -1, -1, -1])
                rotation = np.zeros(3)
                mujoco.mju_quat2Vel(rotation, difference, 1.)
                qa = e.qadr[ik.slices[key]]
                joints = e.joints[ik.slices[key]]
                untouched = np.ones(e.model.nq, dtype=bool)
                untouched[qa] = False
                position_error = float(np.linalg.norm(d.xpos[body] - position - offset))
                rotation_error = float(np.linalg.norm(rotation))
                margin = float(np.minimum(changed[qa] - e.model.jnt_range[joints, 0],
                                          e.model.jnt_range[joints, 1] - changed[qa]).min())
                unchanged = bool(np.array_equal(original[untouched], changed[untouched]) and
                                 np.array_equal(other_positions, d.xpos[other_bodies]))
                passed = position_error < .00002 and rotation_error < .0001 and margin >= 0 and unchanged
                records.append(dict(index=int(index), time_s=float(plan['time'][index]), hand=key,
                     position_error_m=position_error, rotation_error_rad=rotation_error,
                     hand_joint_margin_rad=margin, other_dofs_and_effectors_exact=unchanged,
                     maximum_joint_delta_rad=float(abs(changed[qa] - original[qa]).max()),
                     hook_displacement_m=(d.xpos[body] - position).tolist(), passed=bool(passed)))
                candidates.append(changed)
            trajectory = args.output / f"{case['name']}_knots.npz"
            np.savez_compressed(trajectory, indices=indices, qpos=np.asarray(candidates))
            result = dict(**case, passed=all(r['passed'] for r in records), records=records,
                          knots_sha256=sha(trajectory))
            write(args.output / f"{case['name']}.json", result)
            rows.append({k: v for k, v in result.items() if k != 'records'})
            print(json.dumps(rows[-1]), flush=True)
        assert np.array_equal(e.data.qpos, live_before[0]) and np.array_equal(e.data.qvel, live_before[1]) and e.data.time == live_before[2]
        assert sha(plan_path) == protocol['plan_sha256'] and sha('training/climb_cycle.py') == protocol['ik_source_sha256']
        write(args.output / 'summary.json', dict(cases=rows, passed=sum(r['passed'] for r in rows),
              declared_cases=16, knots_per_case=len(indices), live_state_unchanged=True,
              protocol_sha256=sha(args.output / 'protocol.json'), scope=protocol['scope']))
    finally:
        base.close()


if __name__ == '__main__':
    main()
