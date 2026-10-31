"""Before/after comparison from the retrain's own summary (same reference, same cases)."""
import json
from pathlib import Path

run = Path(r'D:\G1Climbing\external\RoboMimic_Deploy\outputs\calibrated_geometry_ppo131072_20261029_wide125_v1')
data = json.loads((run / 'summary.json').read_text(encoding='utf-8-sig'))

before = {r['case']: r for r in data['before']}
after = {r['case']: r for r in data['after']}

print('%-13s %6s %6s  %10s %10s  %s' % ('case', 'before', 'after', 'ret_before', 'ret_after', 'reason(after)'))
changed = []
for case in sorted(before):
    b, a = before[case], after[case]
    flag = ''
    if b['success'] != a['success']:
        flag = '  <== flipped'
        changed.append(case)
    print('%-13s %6s %6s  %10.2f %10.2f  %s%s' % (
        case, b['success'], a['success'], b['return_sum'], a['return_sum'],
        '' if a['success'] else a['reason'], flag))

print('\nbefore: %d/24 pass   after: %d/24 pass' % (
    sum(1 for r in before.values() if r['success']),
    sum(1 for r in after.values() if r['success'])))
print('flipped:', changed if changed else 'none')
print('total return before %.1f  after %.1f  (delta %+.1f)' % (
    sum(r['return_sum'] for r in before.values()), sum(r['return_sum'] for r in after.values()),
    sum(r['return_sum'] for r in after.values()) - sum(r['return_sum'] for r in before.values())))
print('\ncheckpoint:', data.get('checkpoint_sha256', '?')[:16])
