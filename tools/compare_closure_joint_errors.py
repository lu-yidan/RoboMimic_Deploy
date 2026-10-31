"""Compare joint-wise closure error on audited saved full-cycle records."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.compare_audited_closure_candidates import load_run, sha


def main():
    p = argparse.ArgumentParser()
    for name in ('before', 'after', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    results, arrays, evidence = {}, {}, {}
    names = None
    for label, root in [('before', a.before), ('after', a.after)]:
        verified = load_run(root)
        m = json.loads((root/'manifest.json').read_text())
        evidence[label] = dict(experiment=str(root.resolve()), audit_sha256=verified['audit_sha256'])
        if names is None:
            names = list(m['evaluation_cases'])
        assert set(names) == set(m['evaluation_cases'])
        env = FullCycleResidualEnv(plan=m['reference'], cases=m['evaluation_cases'])
        rows = []
        try:
            for case in names:
                env.reset(options={'case': case})
                e = env.env
                with np.load(root/f'after_{case}.npz') as z:
                    ids = np.flatnonzero(z['time_s'] > env.horizon-2.)
                    assert len(ids) >= 99 and verified['rows'][case]['complete']
                    errors = z['qpos'][ids][:, e.qadr]-env.plan['qpos'][0, e.qadr]
                    times = z['time_s'][ids]
                    joints = []
                    for j, native in enumerate(e.joints):
                        index = int(np.argmax(abs(errors[:, j])))
                        joints.append(dict(joint=e.model.joint(int(native)).name,
                            peak_signed_mrad=float(errors[index, j]*1000),
                            peak_time_s=float(times[index]), final_signed_mrad=float(errors[-1, j]*1000),
                            rms_mrad=float(np.sqrt(np.mean(errors[:, j]**2))*1000)))
                    joints.sort(key=lambda r: abs(r['peak_signed_mrad']), reverse=True)
                    arrays[f'{label}_{case}_time_s'] = times
                    arrays[f'{label}_{case}_errors_mrad'] = errors*1000
                    rows.append(dict(case=case, success=verified['rows'][case]['success'], joints=joints,
                        trajectory_sha256=sha(root/f'after_{case}.npz')))
        finally:
            env.close()
        results[label] = rows
    a.output.mkdir(exist_ok=False)
    np.savez_compressed(a.output/'joint_error_data.npz', **arrays)
    output = dict(rows=results, evidence=evidence, source_sha256=sha(__file__),
        data_sha256=sha(a.output/'joint_error_data.npz'),
        scope='Joint closure errors over the unchanged last-two-second window, relative to original initial joint pose. '
              'Saved 50 Hz data; this localizes the limiting joint but does not establish the cause or justify changing thresholds.')
    (a.output/'summary.json').write_text(json.dumps(output, indent=2))
    print(json.dumps({k: [dict(case=r['case'], success=r['success'], limiting=r['joints'][0]) for r in rows]
                      for k, rows in results.items()}), flush=True)


if __name__ == '__main__':
    main()
