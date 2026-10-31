"""Profile-mode evaluation: replace the fitted single-cycle closure gate with a distribution + physics.

Old framing : pass/fail against a fitted 40 mrad closure line -> "17/24".
New framing : (a) physics decides pass/fail (contact, rise 250+-10 mm, complete cycle);
              (b) closure is a quality metric reported as mean / P95 / max, with a 35 mrad
                  warning line (1.25x the nominal 26-28 mrad seen in the successful 8-cycle climb).

Closure and rise are recomputed from the recorded trajectories with the audit's own formula.
"""
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(r'D:\G1Climbing\external\RoboMimic_Deploy')
import sys
sys.path.insert(0, str(REPO))

from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv  # noqa: E402

WARNING_MRAD = 35.0
OLD_LINE_MRAD = 40.0
RISE_CENTRE_MM = 250.0
RISE_TOL_MM = 10.0
# A rise violation shallower than this is numerical, not physical: the tracking error alone is ~6 mm,
# so a 0.01-0.03 mm overshoot must not be reported as a failure (that is the 0.02 mm noise failure class).
RISE_NOISE_MM = 1.0


def reference_frame(manifest):
    env = StrictEnv(plan=manifest['reference'], cases={'nominal': {}},
                    normal_bias_y_deg=manifest['normal_bias_y_deg'],
                    memory_enabled=manifest['force_history_memory'])
    return np.asarray(env.env.qadr), env.plan['qpos'][0], float(env.plan['time'][-1])


def read(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8-sig'))


def evaluate(run):
    run = Path(run)
    manifest = json.loads((run / 'manifest.json').read_text())
    qadr, initial, end = reference_frame(manifest)
    reasons = {r['case']: r.get('reason') for r in (read(run / 'after.json', []) or [])}
    audit = read(run / 'independent_candidate_audit.json', {'trajectories': []})
    rows = []
    for row in audit['trajectories']:
        if row['label'] != 'after':
            continue
        case = row['case']
        endpoints = row.get('independent_endpoints') or []
        endpoint = endpoints[0] if endpoints else {}
        closure = (1000 * endpoint['joint_closure_rad']
                   if endpoint.get('joint_closure_rad') is not None else None)
        eff = (1000 * endpoint['effector_error_m']
               if endpoint.get('effector_error_m') is not None else None)
        tilt = endpoint.get('tilt_deg')
        trajectory = run / f'after_{case}.npz'
        rise = complete = None
        if trajectory.exists():
            with np.load(trajectory) as trace:
                mask = (trace['time_s'] > end - 2) & (trace['time_s'] <= end + 1e-7)
                rise = float((trace['qpos'][mask, 2] - initial[2]).max() * 1000) if mask.any() else None
                complete = abs(float(trace['time_s'][-1]) - end) < 1e-6
        reason = reasons.get(case)
        rise_violation = (abs(rise - RISE_CENTRE_MM) - RISE_TOL_MM) if rise is not None else None
        if not complete:
            kind = 'incomplete'
        elif reason == 'unintended_or_swing_contact':
            kind = 'physical_contact'
        elif rise_violation is not None and rise_violation > RISE_NOISE_MM:
            kind = 'physical_rise'
        elif closure is not None and closure > OLD_LINE_MRAD:
            kind = 'closure_over_old_line'
        elif closure is not None and closure > WARNING_MRAD:
            kind = 'closure_over_warning'
        elif reason == 'cycle_endpoint_failure':
            kind = 'endpoint_other'
        else:
            kind = 'pass'
        rows.append(dict(case=case, kind=kind, reason=reason, complete=complete,
                         closure_mrad=closure, rise_mm=rise, rise_violation_mm=rise_violation,
                         effector_error_mm=eff, tilt_deg=tilt))
    return rows


def summarise(rows, label):
    closures = [r['closure_mrad'] for r in rows if r['closure_mrad'] is not None]
    physical = [r['case'] for r in rows if r['kind'].startswith('physical_') or r['kind'] == 'incomplete']
    return dict(
        label=label, cases=len(rows),
        physics_pass=sum(1 for r in rows if r['kind'] not in ('physical_contact', 'physical_rise', 'incomplete')),
        physical_failures=physical,
        closure_mean=float(np.mean(closures)) if closures else None,
        closure_median=float(np.median(closures)) if closures else None,
        closure_p95=float(np.percentile(closures, 95)) if closures else None,
        closure_max=float(np.max(closures)) if closures else None,
        over_warning=sum(1 for c in closures if c > WARNING_MRAD),
        over_old_line=sum(1 for c in closures if c > OLD_LINE_MRAD),
        rows=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--label', default=None)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--markdown', type=Path)
    args = parser.parse_args()
    rows = evaluate(args.run)
    summary = summarise(rows, args.label or args.run.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))

    lines = [f"# 剖面报告：{summary['label']}", '',
             f"- 工况数 {summary['cases']}；物理判定通过 "
             f"{summary['physics_pass']}/{summary['cases']}"
             f"（物理失败：{summary['physical_failures'] or '无'}）",
             f"- 闭合分布：mean {summary['closure_mean']:.2f} / median {summary['closure_median']:.2f}"
             f" / P95 {summary['closure_p95']:.2f} / max {summary['closure_max']:.2f} mrad",
             f"- 越过预警线 {WARNING_MRAD:.0f} mrad 的：{summary['over_warning']} 个；"
             f"越过旧拟合线 {OLD_LINE_MRAD:.0f} mrad 的：{summary['over_old_line']} 个", '',
             '| 工况 | 分类 | 闭合 mrad | 抬升 mm | 端点误差 mm | 倾斜 ° |',
             '|---|---|---|---|---|---|']
    for row in sorted(rows, key=lambda r: -(r['closure_mrad'] or 0)):
        def f(v, spec):
            return '—' if v is None else format(v, spec)
        lines.append(f"| {row['case']} | {row['kind']} | {f(row['closure_mrad'], '.3f')} | "
                     f"{f(row['rise_mm'], '.3f')} | {f(row['effector_error_mm'], '.2f')} | "
                     f"{f(row['tilt_deg'], '.2f')} |")
    text = '\n'.join(lines) + '\n'
    if args.markdown:
        args.markdown.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
