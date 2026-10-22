"""Compare fixed-world and translated-rung endpoint errors without changing acceptance.

The alternative reference translates each nominal effector target with its
declared supporting rung. It is a diagnostic, not a replacement success test.
Joint closure, loads, collisions and dynamic stability remain separate checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from training.full_cycle_residual_env import FullCycleResidualEnv


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.experiment/'manifest.json').read_text())
    env = FullCycleResidualEnv(plan=manifest['reference'], cases=manifest['evaluation_cases'])
    names = ['hook', 'right_hook', 'left_ankle_roll_link', 'right_ankle_roll_link']
    ids = [env.env.model.body(name).id for name in names]
    data = mujoco.MjData(env.env.model)
    results = []
    for label in ('zero', 'after'):
        metadata = args.experiment/f'{label}.json'
        if not metadata.exists():
            continue
        for row in json.loads(metadata.read_text()):
            path = args.experiment/f"{label}_{row['case']}.npz"
            if sha(path) != row['trajectory_sha256']:
                raise ValueError('Trajectory hash changed')
            with np.load(path) as z:
                for cycle, end in enumerate(env.ends, start=1):
                    samples = np.flatnonzero((z['time_s'] > end-2.) & (z['time_s'] <= end+1e-7))
                    if len(samples) < 99:
                        results.append(dict(label=label, case=row['case'], cycle=cycle, complete_window=False))
                        continue
                    reference_index = int(np.argmin(abs(env.plan['time']-end)))
                    data.qpos[:] = env.plan['qpos'][reference_index]
                    mujoco.mj_kinematics(env.env.model, data)
                    desired = data.xpos[ids].copy()
                    rods = [f'rung_{cycle+4:02d}']*2+[f'rung_{cycle:02d}']*2
                    translations = np.array([manifest['evaluation_cases'][row['case']].get(rod, [0., 0., 0.]) for rod in rods])
                    vectors = []
                    for i in samples:
                        data.qpos[:] = z['qpos'][i]
                        mujoco.mj_kinematics(env.env.model, data)
                        vectors.append(data.xpos[ids]-desired)
                    vectors = np.array(vectors)
                    translated = vectors-translations
                    results.append(dict(label=label, case=row['case'], cycle=cycle, complete_window=True,
                        original_runtime_success=row['success'], original_reason=row['reason'],
                        nominal_world_peak_m=float(np.linalg.norm(vectors, axis=2).max()),
                        rung_translated_peak_m=float(np.linalg.norm(translated, axis=2).max()),
                        bodies=[dict(body=name, target_rung=rod, rung_translation_m=translation.tolist(),
                            mean_world_error_xyz_m=vectors[:, j].mean(axis=0).tolist(),
                            peak_world_error_m=float(np.linalg.norm(vectors[:, j], axis=1).max()),
                            peak_translated_error_m=float(np.linalg.norm(translated[:, j], axis=1).max()))
                            for j, (name, rod, translation) in enumerate(zip(names, rods, translations))]))
    result = dict(manifest_sha256=sha(args.experiment/'manifest.json'), reference_sha256=sha(manifest['reference']),
        source_sha256=sha(__file__), results=results,
        scope='Alternative coordinate-reference diagnostic only; original strict success labels remain unchanged. Translating targets does not establish feasible support, joint closure, strength or generalization.')
    args.output.write_text(json.dumps(result, indent=2))
    for row in results:
        if row['complete_window']:
            print(json.dumps({key: row[key] for key in ('label', 'case', 'nominal_world_peak_m', 'rung_translated_peak_m', 'original_runtime_success')}))
    env.close()


if __name__ == '__main__':
    main()
