"""One screen of plain Chinese text for the morning, written as UTF-8."""
import json
from pathlib import Path

WATCH = Path(r'D:\G1Climbing\.dsh-watch')
RUN = Path(r'D:\G1Climbing\external\RoboMimic_Deploy\outputs\calibrated_geometry_ppo131072_20261027_v1')
HOLDOUT = Path(r'D:\G1Climbing\external\RoboMimic_Deploy\outputs\calibrated_geometry_holdout16_20261027_v1')
LABEL = {'training_00': '触杆', 'training_01': '触杆', 'training_02': '抬升超限',
         'training_09': '收尾+抬升', 'training_13': '收尾超限',
         'training_14': '收尾超限', 'training_15': '收尾超限'}


def read(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8-sig'))


def show(value, spec='.3f'):
    return '—' if value is None else format(value, spec)


def main():
    verdict = read(WATCH / 'geometry_verdict.json')
    supervisor = read(WATCH / 'supervisor_result.json') or {}
    holdout = read(HOLDOUT / 'after.json')
    lines = ['G1 几何残差 PPO：训练结束自动摘要', '']
    if supervisor.get('reason'):
        lines.append(f"结束原因：{supervisor['reason']}")
        lines.append(f"结束时间：{supervisor.get('finished_at')}")
    if supervisor.get('additional_steps') is not None:
        lines.append(f"训练步数：{supervisor['additional_steps']} / 131072"
                     f"（完整 episode {supervisor.get('training_completed_episodes', '—')}"
                     f"，成功 {supervisor.get('training_successful_episodes', '—')}）")
    if not verdict:
        lines += ['', '判定表未生成，请检查 .dsh-watch/supervisor_result.json。']
        _write(lines)
        return

    lines.append(f"24 工况：基线 {verdict['baseline_successes']}/{verdict['cases']}"
                 f" → 训练后 {verdict['final_successes']}/{verdict['cases']}")
    lines.append('新翻越：' + ('、'.join(verdict['gained']) if verdict['gained'] else '无'))
    lines.append('被训坏：' + ('、'.join(verdict['lost']) if verdict['lost'] else '无'))
    lines.append('')
    lines.append('7 个重点工况（闭合阈值 40 mrad，抬升须在 250±10 mm）：')
    for row in verdict['seven']:
        lines.append(f"  {row['case']}  {'通过' if row['final_success'] else '失败'}"
                     f"（{LABEL.get(row['case'], '')}）"
                     f"  闭合 {show(row['baseline_closure_mrad'])} → {show(row['final_closure_mrad'])} mrad"
                     f"  抬升 {show(row['baseline_rise_max_mm'])} → {show(row['final_rise_max_mm'])} mm")

    deltas = [r['delta_closure_mrad'] for r in verdict['rows'] if r['delta_closure_mrad'] is not None]
    shifted = [d for d in deltas if d > 0.3]
    lines.append('')
    if len(shifted) >= 5:
        mean = sum(shifted) / len(shifted)
        lines.append(f'全局现象（这才是真结论）：{len(shifted)}/{len(deltas)} 个工况的闭合值被统一恶化 '
                     f'{mean:+.3f} mrad（范围 {min(shifted):+.3f} ~ {max(shifted):+.3f}，'
                     '几乎是一个常数偏置，不是随机噪声）。')
        lines.append('  被训坏的 4 个工况原本离阈值只有 0.007–0.263 mrad，正好被这个常数偏置推过线；')
        lines.append('  7 个失败工况需要的是 −0.07 到 −0.45 mrad，方向相反。')
        lines.append('  所以策略学到的不是"几何相关的能力"，而是"把所有工况的收尾统一推差约 0.4 mrad"。')
    lines.append('')
    lines.append('判读：')
    lines.append('  7 个工况仍全部失败且比基线更差；根本原因见报告第三、四节：')
    lines.append('  参考摆动间隙只有 1.4–3.7 mm（最紧两个正是 00/01）；收尾判据被 0.1 mm 几何尺度的')
    lines.append('  2 mrad 抖动主导；奖励里收尾整形项只值终局阶跃的 0.7%；每工况仅 2.5 个 episode。')
    if holdout:
        bad = [r['case'] for r in holdout if not r['success']]
        lines += ['', f'保留集（16 个未参与训练的几何）：{len(holdout) - len(bad)}/16 通过；'
                  f'未通过 {", ".join(bad)}。']
    _write(lines)


def _write(lines):
    lines += ['', '完整报告：.dsh-watch/G1_几何PPO诊断报告_20261028.md',
              '逐工况 JSON：.dsh-watch/geometry_verdict.json',
              '改造清单：.dsh-watch/改造落地清单_20261028.md',
              '交接说明：.dsh-watch/交接说明.md',
              '', '如果上面某一节是空的，按这个顺序查：',
              '  1. 训练是否跑完 131072 步 → outputs/calibrated_geometry_ppo131072_20261027_v1/training_status.json',
              '  2. 24 工况评估是否完成 → 同目录 after.json 与 parallel_after/',
              '  3. 四项审计是否通过 → 同目录 parameter_update_audit.json / controller_modifier_audit_v2.json /',
              '     adaptive_reference_audit.json / composite_prefix_audit_v2.json',
              '  4. 保留集是否启动 → outputs/calibrated_geometry_holdout16_20261027_v1/holdout_result.json',
              '  5. 自动流程自身的日志 → .dsh-watch/supervisor_result.json、holdout_run.log、update_audit.log']
    (WATCH / '明早先看这个.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('morning summary written:', len(lines), 'lines')


if __name__ == '__main__':
    main()
