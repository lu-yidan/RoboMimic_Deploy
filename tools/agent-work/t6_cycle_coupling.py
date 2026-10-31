"""T6: is there a first-cycle / later-cycle coupling in the adopted 8-cycle validation?

Reads the adopted 24-case eight-cycle summary and reports:
  * per-cycle closure (mean over cases) for cycles 1..8,
  * the spread of cycle-1 closure vs the mean of cycles 2-8,
  * whether later cycles drift (linear slope over cycles 2-8),
  * per-case worst cycle and whether cycle 1 is the worst,
  * cumulative rise error growth (does the endpoint correction accumulate?).
"""
import json
from pathlib import Path

import numpy as np

RUN = Path(r'D:\G1Climbing\external\RoboMimic_Deploy\outputs\v2_swing_all24_eight_analysis_20261029_v1')
data = json.loads((RUN / 'summary.json').read_text(encoding='utf-8-sig'))
rows = data['rows']

matrix = np.array([[m['closure_mrad'] for m in r['cycle_metrics']] for r in rows])
cases = [r['case'] for r in rows]
rise = np.array([[m['cumulative_rise_mm'] for m in r['cycle_metrics']] for r in rows])

print('cases %d, cycles %d' % matrix.shape)
print('\nper-cycle closure across 24 cases (mrad):')
print('%-6s %8s %8s %8s %8s' % ('cycle', 'mean', 'median', 'max', 'n>=40'))
for k in range(matrix.shape[1]):
    col = matrix[:, k]
    print('%-6d %8.3f %8.3f %8.3f %8d' % (k + 1, col.mean(), np.median(col), col.max(), int((col >= 40).sum())))

first = matrix[:, 0]
later = matrix[:, 1:]
later_mean = later.mean(axis=1)
print('\ncycle 1 vs cycles 2-8 (per case):')
print('  mean cycle1 %.3f   mean cycles2-8 %.3f   difference %+.3f' % (first.mean(), later_mean.mean(), first.mean() - later_mean.mean()))
corr = np.corrcoef(first, later_mean)[0, 1]
print('  correlation(cycle1, mean cycles2-8) = %+.3f  (n=%d)' % (corr, len(first)))
print('  cases where cycle 1 is the worst cycle: %d/%d' % (int((matrix.argmax(axis=1) == 0).sum()), len(cases)))

slopes = [np.polyfit(np.arange(2, 9), later[i], 1)[0] for i in range(len(cases))]
print('\ndrift over cycles 2-8 (mrad per cycle):')
print('  mean %+.3f   median %+.3f   min %+.3f   max %+.3f' % (
    np.mean(slopes), np.median(slopes), min(slopes), max(slopes)))
print('  cases with slope > +0.2 (worsening): %d' % sum(1 for s in slopes if s > 0.2))
print('  cases with slope < -0.2 (improving): %d' % sum(1 for s in slopes if s < -0.2))

print('\ncumulative rise error (mm) at cycles 1/4/8:')
for k in (0, 3, 7):
    print('  cycle %d: mean %.3f  max %.3f' % (k + 1, np.abs(rise[:, k]).mean(), np.abs(rise[:, k]).max()))

print('\nworst five cases by cycle-1 closure:')
order = np.argsort(-first)[:5]
print('%-13s %9s %9s %9s' % ('case', 'cycle1', 'cy2-8 mean', 'worst cycle'))
for i in order:
    print('%-13s %9.3f %9.3f %9d' % (cases[i], first[i], later_mean[i], int(matrix[i].argmax()) + 1))

Path(r'D:\G1Climbing\.dsh-watch\t6_cycle_coupling.json').write_text(json.dumps(dict(
    per_cycle_mean=matrix.mean(axis=0).tolist(),
    per_cycle_max=matrix.max(axis=0).tolist(),
    correlation_cycle1_vs_later=float(corr),
    drift_mean_mrad_per_cycle=float(np.mean(slopes)),
    cases=np.array(cases).tolist(),
    closure=matrix.tolist()), indent=2))
print('\nwrote .dsh-watch/t6_cycle_coupling.json')
