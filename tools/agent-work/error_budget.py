"""Error-budget waterfall for the endpoint closure metric (same five cases everywhere)."""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

WATCH = Path(r'D:\G1Climbing\.dsh-watch')
CASES = ['training_03', 'training_09', 'training_13', 'training_14', 'training_15']

zero = {r['case']: r['closure_mrad'] for r in json.loads((WATCH / 'zero_residual' / 'zero_result.json').read_text(encoding='utf-8-sig'))}
sweep = json.loads((WATCH / 'calibration_sweep' / 'sweep.json').read_text(encoding='utf-8-sig'))
b0 = {r['case']: r['closure_mrad'] for r in next(a for a in sweep if a['scale'] == 0.0)['rows']}
b1 = {r['case']: r['closure_mrad'] for r in next(a for a in sweep if a['scale'] == 1.0)['rows']}
mean = lambda d: statistics.mean(d[c] for c in CASES)

reference = 28.595
controller = mean(zero)
residual = mean(b0)
final = mean(b1)

steps = [('参考端点\n(运动学)', reference, '#334155'),
         ('控制器跟踪\n(+%.1f)' % (controller - reference), controller, '#b91c1c'),
         ('学到的残差\n(%.1f)' % (residual - controller), residual, '#0f766e'),
         ('校准偏置\n(%.1f)' % (final - residual), final, '#0f766e')]

figure, axis = plt.subplots(figsize=(9.2, 5.2))
running = 0
for index, (label, value, color) in enumerate(steps):
    if index == 0:
        axis.bar(index, value, color=color, width=.6)
        running = value
    else:
        bottom = min(running, value)
        axis.bar(index, abs(value - running), bottom=bottom, color=color, width=.6)
        axis.plot([index - 1 + .3, index - .3], [running, running], color='#94a3b8', linestyle='--', linewidth=1)
        running = value
    axis.text(index, max(value, running) + .8, '%.1f' % value, ha='center', fontsize=10,
              color='#111827', fontweight='bold')

axis.axhline(40, color='#dc2626', linestyle=':', linewidth=1.4)
axis.text(-0.42, 41.0, '40 mrad（拟合出来的门）', color='#dc2626', fontsize=9, ha='left')
axis.set_xticks(range(len(steps)))
axis.set_xticklabels([s[0] for s in steps], fontsize=9.5)
axis.set_ylabel('endpoint joint closure (mrad)')
axis.set_ylim(0, 58)
axis.grid(axis='y', alpha=.25)
axis.set_title('Where the 11 mrad comes from: controller tracking dominates the error budget',
               fontsize=10.5, loc='left')
figure.tight_layout()
target = WATCH / 'error_budget'
target.mkdir(exist_ok=True)
figure.savefig(target / 'closure_waterfall.png', dpi=180)
print('reference %.3f  controller %.3f  residual %.3f  final %.3f' % (reference, controller, residual, final))
print('wrote', target / 'closure_waterfall.png')
