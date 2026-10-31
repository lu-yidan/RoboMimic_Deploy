"""Trade-off curve for the fixed-calibration bias: closure quality vs swing-contact risk.

Reads the audited 5-point sweep (scale 0..1) and the wide-cap extension (1.25/1.50/1.75),
then writes a table + figure: closure distribution on the left axis, contact failures on the right.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WATCH = Path(r'D:\G1Climbing\.dsh-watch')
OUT = WATCH / 'bias_tradeoff'


def load(name):
    path = WATCH / name
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding='utf-8-sig'))


def summarise(arm):
    rows = [r for r in arm.get('rows', []) if r.get('closure_mrad') == r.get('closure_mrad')]
    contact = [r['case'] for r in arm.get('rows', []) if r.get('success') is False and
               r.get('closure_mrad') != r.get('closure_mrad')]
    closures = [r['closure_mrad'] for r in rows]
    return dict(scale=arm['scale'],
                mean=float(np.mean(closures)) if closures else None,
                max=float(np.max(closures)) if closures else None,
                passed=arm.get('successes'),
                contact=contact,
                n=len(arm.get('rows', [])))


def main():
    arms = [a for a in load('calibration_sweep/sweep.json') if 'rows' in a]
    arms += [a for a in load('calibration_sweep_wide/sweep_wide.json') if 'rows' in a]
    if not arms:
        raise SystemExit('no sweep data yet')
    arms.sort(key=lambda a: a['scale'])
    table = [summarise(a) for a in arms]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'tradeoff.json').write_text(json.dumps(table, indent=2))

    print('%-6s %8s %8s %7s  %s' % ('scale', 'meanCl', 'maxCl', 'passed', 'contact failures'))
    for row in table:
        print('%-6.2f %8s %8s %7s  %s' % (
            row['scale'],
            '—' if row['mean'] is None else '%.2f' % row['mean'],
            '—' if row['max'] is None else '%.2f' % row['max'],
            row['passed'], ','.join(row['contact']) or '—'))

    figure, axis = plt.subplots(figsize=(8.6, 4.8))
    scales = [r['scale'] for r in table]
    means = [r['mean'] for r in table]
    maxes = [r['max'] for r in table]
    axis.plot(scales, means, 'o-', color='#0f766e', label='mean closure (mrad)')
    axis.plot(scales, maxes, 's--', color='#0f766e', alpha=.55, label='max closure (mrad)')
    axis.axhline(40, color='#b91c1c', linestyle=':', linewidth=1.2, label='fitted 40 mrad line')
    axis.axhline(35, color='#d97706', linestyle=':', linewidth=1.0, label='35 mrad warning')
    axis.set_xlabel('fixed knee/ankle bias scale (1.0 = current audited calibration)')
    axis.set_ylabel('closure (mrad)')
    axis.grid(alpha=.25)
    right = axis.twinx()
    right.bar(scales, [len(r['contact']) for r in table], width=.05, alpha=.25,
              color='#b91c1c', label='swing-contact failures')
    right.set_ylabel('swing-contact failures (of 7 cases)')
    right.set_ylim(0, 7)
    axis.set_title('Calibration bias trade-off: closure improves monotonically,\n'
                   'but swing clearance runs out above 1.0', fontsize=10, loc='left')
    handles, labels = axis.get_legend_handles_labels()
    handles2, labels2 = right.get_legend_handles_labels()
    axis.legend(handles + handles2, labels + labels2, fontsize=8, loc='lower left')
    figure.tight_layout()
    figure.savefig(OUT / 'bias_tradeoff.png', dpi=170)
    print('\nwrote', OUT / 'bias_tradeoff.png')


if __name__ == '__main__':
    main()
