"""Check the steady-state-droop diagnosis across all zero-residual cases."""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(r'D:\G1Climbing\external\RoboMimic_Deploy')
sys.path.insert(0, str(REPO))

from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv  # noqa: E402

BASE = REPO / 'outputs/calibrated_geometry_training_baseline_20261027_v1'
ZERO = Path(r'D:\G1Climbing\.dsh-watch\zero_residual/ref_only/eval')
CASES = ['training_03', 'training_09', 'training_13', 'training_14', 'training_15']
manifest = json.loads((BASE / 'manifest.json').read_text())

print('%-13s %10s %10s %10s %14s' % ('case', 'max|dq|°', 'rms°', 'std°(末2s)', 'slope°/s'))
summary = []
for case in CASES:
    env = StrictEnv(plan=manifest['reference'], cases={case: manifest['evaluation_cases'][case]},
                    normal_bias_y_deg=manifest['normal_bias_y_deg'],
                    memory_enabled=manifest['force_history_memory'])
    model, qadr = env.env.model, np.asarray(env.env.qadr)
    with np.load(Path(manifest['reference'])) as reference:
        ref_t, ref_q = reference['time'], reference['qpos']
    with np.load(ZERO / f'after_{case}.npz') as trace:
        t, q = trace['time_s'], trace['qpos']
    ref_interp = np.stack([np.interp(t, ref_t, ref_q[:, k]) for k in qadr], axis=1)
    err = np.degrees(q[:, qadr] - ref_interp)
    per_sample_max = np.abs(err).max(axis=1)
    mask = t > 111.7
    steady = err[mask].mean(axis=0)
    names = [model.joint(int(j)).name for j in env.env.joints]
    worst = int(np.argmax(np.abs(steady)))
    slope = float(np.polyfit(t[mask], per_sample_max[mask], 1)[0])
    print('%-13s %10.3f %10.3f %10.4f %14.4f   worst=%s %+.3f°' % (
        case, per_sample_max[mask].max(), np.sqrt((err[mask] ** 2).mean()),
        per_sample_max[mask].std(), slope, names[worst], steady[worst]))
    summary.append(dict(case=case, std=float(per_sample_max[mask].std()), slope=slope,
                        worst_joint=names[worst], worst_offset_deg=float(steady[worst])))

path = Path(r'D:\G1Climbing\.dsh-watch\zero_residual\steady_state.json')
path.write_text(json.dumps(summary, indent=2))
print('\nall cases flat (std < 0.02 deg):', all(s['std'] < 0.02 for s in summary))
print('wrote', path)
