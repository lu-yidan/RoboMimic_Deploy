"""Feasibility map of the frozen 24-case geometry split from the audited baseline run.

For every case: horizontal offset from the split centre, joint closure (mrad), root rise (mm),
acceptance margins. Everything is read from the already-audited baseline artefacts; no simulation.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(r'D:\G1Climbing\external\RoboMimic_Deploy\outputs\calibrated_geometry_training_baseline_20261027_v1')
PROTOCOL = Path(r'D:\G1Climbing\external\RoboMimic_Deploy\outputs\calibrated_geometry_curriculum_protocol_20261027_v1\protocol.json')
OUT = Path(r'D:\G1Climbing\.dsh-watch')

manifest = json.loads((BASE / 'manifest.json').read_text())
protocol = json.loads(PROTOCOL.read_text())
center_x = protocol['center']['rung_01'][0]
center_z = protocol['center']['rung_01'][2]
with np.load(manifest['reference']) as reference:
    initial_z = float(reference['qpos'][0, 2])
    end = float(reference['time'][-1])
audit = json.loads((BASE / 'independent_candidate_audit.json').read_text())
after = {row['case']: row for row in json.loads((BASE / 'after.json').read_text())}

rows = []
for row in audit['trajectories']:
    if row['label'] != 'after':
        continue
    case = row['case']
    if not case.startswith('training_'):
        continue          # the eight guard cases are a different, already-known geometry set
    geometry = protocol['training_cases'][case]['rung_01']
    offsets = (geometry[0] - center_x, geometry[2] - center_z)
    endpoints = row['independent_endpoints']
    closure = endpoints[0]['joint_closure_rad'] if endpoints else None
    with np.load(BASE / f'after_{case}.npz') as trace:
        mask = (trace['time_s'] > end - 2) & (trace['time_s'] <= end + 1e-7)
        rise = trace['qpos'][mask, 2] - initial_z
        complete = abs(float(trace['time_s'][-1]) - end) < 1e-6
    rows.append(dict(case=case, dx_mm=1000 * offsets[0], dz_mm=1000 * offsets[1],
                     success=bool(row['experiment_success']),
                     reason=after[case]['reason'], complete=complete,
                     closure_mrad=1000 * closure if closure is not None else None,
                     rise_mm=1000 * float(rise.max()) if len(rise) else None))

Figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
ordered = sorted(rows, key=lambda r: r['dx_mm'])
for axis, key, limit, title in ((axes[0], 'closure_mrad', 40., 'Joint closure (threshold 40 mrad)'),
                                (axes[1], 'rise_mm', 260., 'Max root rise (allowed 240-260 mm)')):
    xs = [r['dx_mm'] for r in ordered if r[key] is not None]
    ys = [r[key] for r in ordered if r[key] is not None]
    colors = ['#0f766e' if r['success'] else '#b91c1c' for r in ordered if r[key] is not None]
    axis.scatter(xs, ys, c=colors, s=48, zorder=3, edgecolor='white', linewidth=.6)
    for r in ordered:
        if r[key] is not None:
            axis.annotate(r['case'].replace('training_', 'T').replace('boundary_', 'B').replace('interior_', 'I'),
                          (r['dx_mm'], r[key]), textcoords='offset points', xytext=(4, 4), fontsize=7, color='#334155')
    axis.axhline(limit, color='#b91c1c', linestyle='--', linewidth=1.1)
    if key == 'rise_mm':
        axis.axhline(240, color='#b91c1c', linestyle='--', linewidth=1.1)
    axis.set_title(title, fontsize=10, loc='left')
    axis.set_xlabel('rung x offset from split centre (mm)')
    axis.grid(alpha=.25)
axes[0].set_ylabel('mrad')
axes[1].set_ylim(259.90, 260.15)
axes[1].set_ylabel('mm')
Figure.suptitle('16 new geometry cases (0.1 mm apart) under the audited baseline policy: the 40 mrad / 260 mm boundary runs through the middle of the split',
                fontsize=11, x=.01, ha='left')
Figure.tight_layout(rect=(0, 0, 1, .94))
figure_path = OUT / 'feasibility_map_baseline.png'
Figure.savefig(figure_path, dpi=170)

feasible = [r for r in rows if r['success']]
failed = [r for r in rows if not r['success']]
summary = dict(cases=len(rows), passed=len(feasible), failed=len(failed),
               x_offset_span_mm=[min(r['dx_mm'] for r in rows), max(r['dx_mm'] for r in rows)],
               z_offset_span_mm=[min(r['dz_mm'] for r in rows), max(r['dz_mm'] for r in rows)],
               closure_mrad_passed=[min(r['closure_mrad'] for r in feasible), max(r['closure_mrad'] for r in feasible)],
               closure_mrad_failed=[min(r['closure_mrad'] for r in failed if r['closure_mrad']) ,
                                    max(r['closure_mrad'] for r in failed if r['closure_mrad'])],
               feasible_x_window_mm=[min(r['dx_mm'] for r in feasible), max(r['dx_mm'] for r in feasible)],
               failed_dx_mm=sorted(round(r['dx_mm'], 3) for r in failed),
               rows=rows, figure=str(figure_path))
(OUT / 'feasibility_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))
lines = []
for r in ordered:
    lines.append('{:>11} dx={:+7.3f} mm dz={:+6.3f} mm  closure={:>8} rise={:>8}  {}  {}'.format(
        r['case'], r['dx_mm'], r['dz_mm'],
        '-' if r['closure_mrad'] is None else f"{r['closure_mrad']:8.3f}",
        '-' if r['rise_mm'] is None else f"{r['rise_mm']:8.3f}",
        'PASS' if r['success'] else 'fail', r['reason'] or ''))
print('\n'.join(lines))
(OUT / 'geometry_split_table.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


