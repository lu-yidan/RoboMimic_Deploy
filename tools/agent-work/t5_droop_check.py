"""T5: does the controller's static droop still exist on the adopted reference?

No simulation: compares the recorded 8-cycle trajectory against the reference it was asked to track,
inside cycle 1's endpoint window, and reports the tracking error (max / RMS / steadiness) and the
worst joints -- the same measurement that gave +23.1 mrad on the old reference.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(r'D:\G1Climbing\external\RoboMimic_Deploy')
sys.path.insert(0, str(REPO))

RUN = REPO / 'outputs/v2_swing_all24_suite_20261029_v1/8_cycles'
CASES = ['interior_00', 'training_13', 'training_14', 'training_00']
OLD_DROOP_DEG = 1.87          # measured on the old reference (steady, 5 cases, std 0.005 deg)

# joint qpos addresses are constant across cases; recover them from the model once
from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv  # noqa: E402

manifest = json.loads((REPO / 'outputs/calibrated_geometry_training_baseline_20261027_v1/manifest.json').read_text())
env = StrictEnv(plan=str(Path(manifest['reference'])),
                cases={'x': {}}, normal_bias_y_deg=-3., memory_enabled=True)
qadr = np.asarray(env.env.qadr)
names = [env.env.model.joint(int(j)).name for j in env.env.joints]

rows = []
for case in CASES:
    directory = RUN / case
    with np.load(directory / 'trajectory.npz') as trace:
        t, q = trace['time_s'], trace['qpos']
    with np.load(directory / 'reference.npz') as ref:
        ref_t, ref_q, ends = ref['time'], ref['qpos'], ref['cycle_ends_s']
    if len(ends) != 8:
        print(case, 'unexpected cycle ends', len(ends))
    # cycle 1 endpoint window: last 2 s before ends[0]
    end = float(ends[0])
    mask = (t > end - 2) & (t <= end + 1e-7)
    ref_interp = np.stack([np.interp(t[mask], ref_t, ref_q[:, k]) for k in qadr], axis=1)
    error = np.degrees(q[mask][:, qadr] - ref_interp)
    per_sample = np.abs(error).max(axis=1)
    steady = error.mean(axis=0)
    worst = int(np.argmax(np.abs(steady)))
    rows.append(dict(case=case, cycle1_end_s=end, max_deg=float(per_sample.max()),
                     rms_deg=float(np.sqrt((error ** 2).mean())),
                     std_deg=float(per_sample.std()),
                     slope_deg_per_s=float(np.polyfit(t[mask], per_sample, 1)[0]),
                     worst_joint=names[worst], worst_offset_deg=float(steady[worst])))
    print('%-13s cycle1 end %.2f s  max %.3f deg  rms %.3f  std %.4f  worst %s %+.3f deg'
          % (case, end, rows[-1]['max_deg'], rows[-1]['rms_deg'], rows[-1]['std_deg'],
             rows[-1]['worst_joint'], rows[-1]['worst_offset_deg']))

mean_max = np.mean([r['max_deg'] for r in rows])
print('\nmean cycle-1 tracking error: %.3f deg (old reference: %.3f deg)' % (mean_max, OLD_DROOP_DEG))
print('all steady (std < 0.02 deg):', all(r['std_deg'] < 0.02 for r in rows))
Path(r'D:\G1Climbing\.dsh-watch\t5_droop_on_adopted_reference.json').write_text(json.dumps(
    dict(cases=rows, old_reference_droop_deg=OLD_DROOP_DEG), indent=2))
print('wrote .dsh-watch/t5_droop_on_adopted_reference.json')
