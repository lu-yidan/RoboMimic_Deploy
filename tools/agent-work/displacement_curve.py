"""Decisive measurement: closure as a function of rung displacement.

The 24 'geometry cases' are not independent: each offsets a single rung (rung_01). This builds the actual
displacement -> closure relation from existing data, then reports where the 40 mrad line is crossed.
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(r'D:\G1Climbing\external\RoboMimic_Deploy')
WATCH = Path(r'D:\G1Climbing\.dsh-watch')

manifest = json.loads((REPO / 'outputs/calibrated_geometry_training_baseline_20261027_v1/manifest.json').read_text())
audit = json.loads((REPO / 'outputs/swing_lift_phase_sine_policy_20261029_v2/reference_chain_audit.json').read_text(encoding='utf-8-sig'))
frozen_1x = {r['case']: r['new_closure_mrad'] for r in audit['rows']}
frozen_125 = {r['case']: r['closure'] for r in json.loads((WATCH / 'authority_all24/result.json').read_text())}
trained = {r['case']: r['closure'] for r in json.loads((WATCH / 'retrained_all24/result.json').read_text())}

rows = []
for case, spec in manifest['evaluation_cases'].items():
    rung, offset = next(iter(spec.items()))
    rows.append(dict(case=case, rung=rung, x_mm=1000 * offset[0], z_mm=1000 * offset[2],
                     frozen_1x=frozen_1x.get(case), frozen_125=frozen_125.get(case),
                     trained=trained.get(case)))
rows.sort(key=lambda r: r['z_mm'])

print('%-13s %-8s %9s %9s %10s %10s %10s' % ('case', 'rung', 'x mm', 'z mm', 'frozen1x', 'frozen1.25', 'trained'))
for r in rows:
    print('%-13s %-8s %9.3f %9.3f %10s %10s %10s' % (
        r['case'], r['rung'], r['x_mm'], r['z_mm'],
        '%.3f' % r['frozen_1x'] if r['frozen_1x'] else '—',
        '%.3f' % r['frozen_125'] if r['frozen_125'] else '—',
        '%.3f' % r['trained'] if r['trained'] else '—'))

z = np.array([r['z_mm'] for r in rows])
for key, label in (('frozen_1x', 'frozen 1.0x'), ('frozen_125', 'frozen 1.25x'), ('trained', 'trained 1.25x')):
    y = np.array([r[key] for r in rows], dtype=float)
    ok = np.isfinite(y)
    slope, intercept = np.polyfit(z[ok], y[ok], 1)
    resid = y[ok] - (slope * z[ok] + intercept)
    print('\n%-14s slope %+.4f mrad/mm   intercept %.3f   residual std %.3f   corr %+.3f' % (
        label, slope, intercept, resid.std(), np.corrcoef(z[ok], y[ok])[0, 1]))
    if slope > 0:
        print('              40 mrad reached at z = %.1f mm' % ((40. - intercept) / slope))
    print('              closure at z=0 : %.3f   at z=15 : %.3f' % (intercept, slope * 15 + intercept))

levels = sorted(set(round(r['z_mm'], 1) for r in rows))
print('\ndistinct displacement levels:', levels)
for level in levels:
    members = [r for r in rows if abs(r['z_mm'] - level) < 0.1]
    values = [r['trained'] for r in members if r['trained']]
    print('  z=%.1f mm  n=%2d  trained mean %.3f  max %.3f' % (
        level, len(members), np.mean(values) if values else float('nan'),
        max(values) if values else float('nan')))
