"""Read this experiment's saved evidence; never infer process liveness."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def load(relative):
    path=ROOT/relative
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def grid(folder,declared):
    path=ROOT/folder
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((path/'results').glob('*.json'))]
    summary=load(folder+'/suite_summary.json')
    return dict(declared_runs=declared,saved_results=len(rows),audited_passes=sum(bool(r.get('passed')) for r in rows),
        statuses=dict(Counter(r['status'] for r in rows)),suite_finished=bool(summary and summary.get('finished')),
        report=folder+'/suite_summary.json',
        interpretation='文件状态，不代表进程仍在运行；审计异常不自动算作物理失败。')


def status():
    all24=load('outputs/v2_swing_all24_eight_analysis_20261029_v1/summary.json')
    reference=load('outputs/wholebody_hand_reference_candidate_20261029_v2/summary.json')
    pilot=load('outputs/wholebody_hand_dynamic_pilot_20261029_v1/reference_retarget_audit.json')
    comparison=load('outputs/wholebody_reference_dynamic_comparison_20261029_v1/summary.json')
    recovery=load('outputs/hook_rung06_audit_recovery_20261029_v1/summary.json')
    paired=grid('outputs/hook_rung06_paired_two_cycle_20261029_v1',32)
    timed=grid('outputs/hook_rung06_timed_two_cycle_20261029_v1',16)
    return dict(official_baseline='adopted swing-phase v2; frozen policy; no new training',
        all24_eight_cycle_report_exists=all24 is not None,paired_grid=paired,timed_grid=timed,
        three_arm_analysis_complete=(ROOT/'outputs/hook_rung06_three_arm_analysis_20261029_v1/summary.json').exists(),
        audit_only_recovery=None if recovery is None else {k:recovery[k] for k in ('attempted','passed')},
        wholebody_reference_kinematic_passed=bool(reference and reference['passed']),
        wholebody_dynamic_audit=None if pilot is None else bool(pilot['passed']),
        wholebody_comparison_complete=comparison is not None,
        interpretation='空值表示尚无相应完成报告，不能当作失败或成功；本命令不检查系统进程。')


def main():
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--json',action='store_true');a=p.parse_args()
    s=status()
    if a.json:
        print(json.dumps(s,ensure_ascii=False,indent=2));return
    print('正式基线：摆动阶段 v2，原冻结策略，未新增训练。')
    for label,key in [('原 v2 / 接触确认对照','paired_grid'),('固定延时对照','timed_grid')]:
        r=s[key]
        print(f"{label}：已有 {r['saved_results']}/{r['declared_runs']} 条结果，完整审计通过 {r['audited_passes']} 条；整组完成={r['suite_finished']}")
        print('  分类：'+json.dumps(r['statuses'],ensure_ascii=False))
    print(f"三方案汇总报告已生成：{s['three_arm_analysis_complete']}")
    print(f"只读补审：{s['audit_only_recovery']}")
    print(f"全身参考运动学候选通过：{s['wholebody_reference_kinematic_passed']}")
    print(f"全身参考动力学审计：{s['wholebody_dynamic_audit']}；比较报告已生成：{s['wholebody_comparison_complete']}")
    print(s['interpretation'])


if __name__=='__main__':main()
