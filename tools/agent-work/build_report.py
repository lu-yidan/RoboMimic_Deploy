"""Build the Chinese diagnostic report from the artefacts collected in .dsh-watch.

Reads whatever exists: geometry_verdict.json (post-training), feasibility_summary.json,
supervisor_result.json, training_history.jsonl. Writes G1_几何PPO诊断报告_20261028.md.
Runs before and after the training finishes, so every input is optional.
"""
import json
from pathlib import Path

WATCH = Path(r'D:\G1Climbing\.dsh-watch')
ROOT = Path(r'D:\G1Climbing\external\RoboMimic_Deploy')
RUN = Path(r'D:\G1Climbing\external\RoboMimic_Deploy\outputs\calibrated_geometry_ppo131072_20261027_v1')
BASE = Path(r'D:\G1Climbing\external\RoboMimic_Deploy\outputs\calibrated_geometry_training_baseline_20261027_v1')
SEVEN = ['training_00', 'training_01', 'training_02', 'training_09',
         'training_13', 'training_14', 'training_15']
WEIGHT = .1


def read(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except json.JSONDecodeError:
        return default


def quality_bonus(closure_rad, rise_m, ramp=1.0, weight=WEIGHT):
    """Verbatim copy of training/closure_quality_reward.py:quality_bonus."""
    joint = max(0., closure_rad - .032) / .008
    height = max(0., abs(rise_m - .25) - .006) / .004
    return float(.5 * weight * ramp * (1. / (1. + joint * joint) + 1. / (1. + height * height)))


def bonus_slope_per_mrad(closure_rad):
    step = 1e-6
    return ((quality_bonus(closure_rad + step, .260) - quality_bonus(closure_rad - step, .260))
            / (2 * step) / 1000)


def main():
    verdict = read(WATCH / 'geometry_verdict.json')
    feasibility = read(WATCH / 'feasibility_summary.json')
    supervisor = read(WATCH / 'supervisor_result.json')
    diagnosis = read(BASE / 'failure_endpoint_diagnosis.json', {}).get('rows', {})
    run_summary = read(RUN / 'summary.json', {})
    history = []
    history_path = WATCH / 'training_history.jsonl'
    if history_path.exists():
        for line in history_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line:
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    lines = ['# G1 几何残差 PPO：训练诊断与"可正式开展强化学习"改造方案', '',
             f'- 训练目录：`{RUN}`',
             f'- 生成时间：{__import__("datetime").datetime.now().isoformat(timespec="seconds")}',
             f'- 训练步数：{run_summary.get("additional_steps", "进行中")} / 131072'
             f'（parent {run_summary.get("parent_steps", 248832)}）',
             f'- 训练内完整 episode：{run_summary.get("completed_training_episodes", "—")}'
             f'，其中成功 {run_summary.get("successful_training_episodes", "—")}',
             '']

    # ---- 1. 结论 ----
    lines += ['## 一、结论摘要', '']
    if verdict:
        gained, lost = verdict.get('gained', []), verdict.get('lost', [])
        lines.append(f'- 24 工况全量评估：基线 {verdict["baseline_successes"]}/{verdict["cases"]}'
                     f' → 训练后 {verdict["final_successes"]}/{verdict["cases"]}。')
        lines.append(f'- 新翻越的工况：{gained if gained else "无"}；'
                     f'被训坏的工况：{lost if lost else "无"}。')
        seven_pass = [r['case'] for r in verdict['seven'] if r['final_success']]
        lines.append(f'- 7 个重点工况通过数：{len(seven_pass)}/7'
                     + (f'（{[c for c in seven_pass]}）' if seven_pass else '（仍全部失败）') + '。')
        delta = verdict['final_successes'] - verdict['baseline_successes']
        if delta < 0:
            lines.append(f'- **判读：净值下降 {abs(delta)} 个工况。** 131072 步非但没有翻越难点，'
                         '还把部分原本通过的工况推过了阈值——与第四节"整组压在判据线上、'
                         '训练只能拟合噪声"的结论一致。')
        elif delta == 0:
            lines.append('- **判读：与基线持平。** 131072 步、每工况约 2.5 个 episode，'
                         '没有换来任何一个工况的翻转；结合第四、五节，这不是"步数不够"，'
                         '而是可行域、判据判别宽度与奖励尺度三件事同时卡住。')
        else:
            lines.append(f'- **判读：净值增加 {delta} 个工况。** 说明在可行域内的部分工况确实还能被残差改进，'
                         '但需要核对 gain 是否集中在可行窗口内（第三节），且是否伴随 lost。')
    else:
        lines.append('- 训练尚未结束或评估尚未完成，本节在结束时自动补齐。')
    lines += ['- 结构性结论（与训练步数无关，见第三节）：训练集 44% 的样本落在固定校准/参考控制器的'
              '可行域之外；收尾判据在 0.1 mm 几何尺度上被 2 mrad 抖动主导，'
              '其中抬升判据存在 0.008–0.026 mm 量级的判定。', '']

    # ---- 2. 逐工况 ----
    lines += ['## 二、逐工况对比', '']
    if verdict:
        lines += ['| 工况 | 基线 | 训练后 | 闭合 mrad（基→后） | 抬升 mm（基→后） | 原因 |',
                  '|---|---|---|---|---|---|']
        for row in verdict['rows']:
            if row['case'] not in SEVEN:
                continue
            def f(v, spec='.3f'):
                return '—' if v is None else format(v, spec)
            lines.append('| {} | {} | {} | {} → {} | {} → {} | {} |'.format(
                row['case'],
                '✅' if row['baseline_success'] else '❌',
                '✅' if row['final_success'] else '❌',
                f(row['baseline_closure_mrad']), f(row['final_closure_mrad']),
                f(row['baseline_rise_max_mm']), f(row['final_rise_max_mm']),
                row['final_reason'] or row['baseline_reason'] or ''))
        lines += ['', '其余 17 个工况：', '', '| 工况 | 基线 | 训练后 |', '|---|---|---|']
        for row in verdict['rows']:
            if row['case'] in SEVEN:
                continue
            lines.append(f"| {row['case']} | {'✅' if row['baseline_success'] else '❌'} | "
                         f"{'✅' if row['final_success'] else '❌'} |")
    else:
        lines += ['| 工况 | 基线闭合 mrad | 基线抬升 mm | 基线判定 |', '|---|---|---|---|']
        for case in SEVEN:
            row = diagnosis.get(case, {})
            closure = row.get('closure_mrad')
            rise = row.get('rise_max_mm')
            lines.append('| {} | {} | {} | {} |'.format(
                case, '—' if closure is None else f'{closure:.3f}',
                '—' if rise is None else f'{rise:.3f}',
                '提前触杆' if closure is None else '收尾失败'))
    lines += ['', '注：闭合阈值 40 mrad，抬升必须落在 250±10 mm；判据还要求末段 2 s 内'
              '不少于 99 个采样点全部满足。', '']

    # ---- 3. 可行域 ----
    if feasibility:
        lines += ['## 三、可行域：这 7 个工况不是"学得不够"，而是"做不到"', '',
                  f'- 16 个新工况只沿 x 排布 {feasibility["x_offset_span_mm"][1] - feasibility["x_offset_span_mm"][0]:.3f} mm'
                  f'（间距约 0.1 mm），z 方向仅 {(feasibility["z_offset_span_mm"][1] - feasibility["z_offset_span_mm"][0]) * 1000:.1f} µm。',
                  f'- 基线策略的可解窗口：x ∈ [{feasibility["feasible_x_window_mm"][0]:.3f}, '
                  f'{feasibility["feasible_x_window_mm"][1]:.3f}] mm（相对切分中心）。',
                  f'- 窗口外的点恰好是那 7 个失败工况：{feasibility["failed_dx_mm"]} mm。',
                  '- 相邻 0.1 mm 的工况之间，闭合值可从 38.02 mrad 跳到 39.76 mrad、'
                  '从 39.99 掉到 38.33（跨度 2.4 mrad），而阈值就在这个跳变区内。',
                  f'- 抬升：全部 16 个点落在 259.93–260.03 mm，超限的 4 个只超 0.008–0.026 mm。', '']

    # ---- 3b. 参考间隙（仅碰撞体） ----
    clearance = read(WATCH / 'reference_clearance_collision.json')
    if clearance:
        rows = sorted(clearance['rows'], key=lambda r: r['clearance_at_77p57_mm'])
        values = [r['clearance_at_77p57_mm'] for r in rows]
        mins = [r['min_clearance_70_81_mm'] for r in rows]
        tightest = rows[:2]
        lines += ['### 摆动间隙只有 1.4–3.7 mm，最紧的两个正是 00/01', '',
                  '- 只统计**碰撞体**（左脚步的 `left_sole_surface`；脚上另有一个 `contype=0` 的纯视觉网格，'
                  '不参与物理，必须排除）后，冻结参考在 77.5675 s 的脚–杆间隙是 '
                  f'{min(values):+0.2f} ~ {max(values):+0.2f} mm，**全部为正**；'
                  f'70–81 s 区间的最小值 {min(mins):+0.2f} ~ {max(mins):+0.2f} mm。',
                  f'- 最紧的两个正是触杆失败的 {tightest[0]["case"]}（{tightest[0]["clearance_at_77p57_mm"]:+.2f} mm）'
                  f'和 {tightest[1]["case"]}（{tightest[1]["clearance_at_77p57_mm"]:+.2f} mm）；'
                  '间隙随 x 单调变化，最宽松的约 +3.70 mm。',
                  '- 参考本身**没有穿杆**（早期一版分析因为把视觉网格算进来而得出相反结论，已作废）。'
                  '真正的问题是余量只有 1.4–3.7 mm，而跟踪误差加上固定膝/踝校准偏置会把脚再推几毫米：'
                  '两个最紧的工况因此撞上，其余工况侥幸通过。',
                  '- 因此 00/01 的修法是**抬高摆动间隙（到 ≥10 mm）或减小校准偏置**，让余量大于控制误差；'
                  '否则残差策略只能学会"在 77.57 s 附近躲一下"，而这一下是毫米级掷硬币。', '']


    # ---- 4. 奖励梯度 ----
    sensitivity = read(WATCH / 'reward_episode_sensitivity.json')
    lines += ['## 四、为什么现有奖励给不出梯度', '',
              '奖励结构（`training/full_cycle_residual_env.py:265`）：', '',
              '- 每步：`0.02 - 2·跟踪误差 - 0.001·动作幅值 - 0.001·动作变化`，2173 步合计约 43；',
              '- 终局：成功 +20、失败 −20 —— 跨度 40，完全由"是否过线"这一个二值事件决定；',
              '- 另有每步累加的收尾质量整形（`closure_quality_reward`，权重 0.1），整集累计约 27。', '']
    if sensitivity:
        lines += ['用真实审计轨迹（`' + sensitivity['case'] + '`，' + str(sensitivity['steps']) + ' 步，'
                  '收尾段自 ' + format(sensitivity['closure_phase_start_s'], '.2f') + ' s 起）把整形项逐步累加，'
                  '得到"0.3 mrad 的进步值多少奖励"：', '',
                  '| 候选 | 整集整形总量 | 0.3 mrad 的价值 | 占终局阶跃(40) |',
                  '|---|---|---|---|']
        for row in sensitivity['rows']:
            lines.append('| {} | {:+.1f} | {:.2f} | {:.1f}% |'.format(
                row['name'], row['episode_sum'], row['delta_per_0p3mrad'], 100 * row['share_of_terminal']))
        lines.append('')
    lines += ['- 现状：0.3 mrad 只值 **0.29**，是终局阶跃的 **0.7%** —— 这就是"学不动"的量化原因。',
              '- 采样预算：24 工况 / 4 worker = 每 worker 6 工况，131072 步 / 24 / 2173 ≈ **2.5 个 episode/工况**。'
              '用 2.5 次采样去命中 0.2–0.5 mrad 的改判，等同于抽奖。', '']

    # ---- 4b. 两个改法 ----
    lines += ['### 4b 两个改法（都不动验收判据）', '',
              '| 改法 | 参数 | 0.3 mrad 的价值 | 整集回报尺度 | 风险 |',
              '|---|---|---|---|---|',
              '| A. 只调整形权重 | w 0.1 → 0.5 / 1.0 | 1.46 / 2.91（3.6% / 7.3%） | 由约 90 涨到约 200 / 360 | '
              '回报尺度变大，价值网络需重新适配，必要时同步降学习率或做奖励缩放 |',
              '| B. 终局有界裕量（推荐） | `r += 6·clip(m/1.5 mrad, −1, 1)` | 1.20（3.0%） | 基本不变（±6 的有界项） | '
              '要改奖励代码，但只加一项、量级明确 |', '',
              '其中 `m = min(40 − 闭合[mrad], (10 − |抬升 − 250|[mm]) · 10)`，与官方 `scaled_margin` 同口径。',
              'B 的好处是只在终局加一次、不随步数累积，量级固定为 ±6，既能给梯度又不淹掉 ±20 的成败信号。', '']
    if feasibility:
        def proposal_margin(closure_mrad, rise_mm):
            return min(40. - closure_mrad, (10. - abs(rise_mm - 250.)) * 10.)

        def proposal_bonus(closure_mrad, rise_mm):
            return 6. * max(-1., min(1., proposal_margin(closure_mrad, rise_mm) / 1.5))

        usable = [r for r in feasibility['rows']
                  if r['closure_mrad'] is not None and r['rise_mm'] is not None]
        usable.sort(key=lambda r: proposal_bonus(r['closure_mrad'], r['rise_mm']))
        lines += ['方案 B 已写成可审阅的模块（`.dsh-watch/proposed/terminal_margin_reward.py`，含自测），'
                  '在 16 个新工况的基线上取值如下 —— 失败拿负值、通过拿正值，量级都在 ±6 以内：', '',
                  '| 工况 | 闭合 mrad | 抬升 mm | 裕量 | 方案 B 终局奖励 | 基线判定 |',
                  '|---|---|---|---|---|---|']
        for row in usable:
            lines.append('| {} | {:.2f} | {:.3f} | {:+.3f} | {:+.2f} | {} |'.format(
                row['case'], row['closure_mrad'], row['rise_mm'],
                proposal_margin(row['closure_mrad'], row['rise_mm']),
                proposal_bonus(row['closure_mrad'], row['rise_mm']),
                '✅' if row['success'] else '❌'))
        lines += ['', '- 提前终止（触杆、关节越界）**不加**这一项：那时没有收尾点，±20 已经扣过分，'
                  '再扣一次裕量等于双重惩罚。模块用 `completed` 参数区分，自测已覆盖。', '']

    # ---- 4c. 训练内 rollout 证据 ----
    episodes_path = RUN / 'training_episodes.jsonl'
    if episodes_path.exists():
        episodes = [json.loads(line) for line in episodes_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        by_case = {}
        for row in episodes:
            entry = by_case.setdefault(row['case'], [0, 0])
            entry[0] += 1
            entry[1] += 1 if row['success'] else 0
        seven_tries = sum(by_case.get(c, [0, 0])[0] for c in SEVEN)
        seven_pass = sum(by_case.get(c, [0, 0])[1] for c in SEVEN)
        flipped = sorted(c for c, (n, ok) in by_case.items() if c not in SEVEN and ok < n)
        lines += ['### 4c 训练内部 rollout 的直接证据', '',
                  f'- 训练中共记录 {len(episodes)} 个完整 episode（成功 {sum(1 for e in episodes if e["success"])} 个）。',
                  f'- 7 个重点工况：**{seven_tries} 次尝试、{seven_pass} 次成功**。',
                  f'- 同时这 {len(flipped)} 个基线通过的工况在探索噪声（std ≈ 0.005）下出现失败：'
                  f'{", ".join(flipped) if flipped else "无"}。',
                  '- 把这两条放在一起：不是"7 个难点学不会"，而是**整个切分都压在判据线上**，'
                  '上下 0.3 mrad 就会改变判定，噪声本身就能翻转通过与否。', '']
        within = [r for r in (feasibility['rows'] if feasibility else [])
                  if r['closure_mrad'] is not None and abs(r['closure_mrad'] - 40.) <= 0.5]
        if within:
            lines += [f'- 量化：16 个新工况里有 **{len(within)} 个**的闭合值落在 40 ± 0.5 mrad 带内'
                      f'（占 {100 * len(within) / len(feasibility["rows"]):.0f}%），'
                      '也就是说整组切分的判别区间只有约 1 mrad 宽（相对阈值 2.5%）。', '']
        status = read(RUN / 'training_status.json', {})
        samples = status.get('case_samples') or {}
        if samples:
            values = sorted(samples.values())
            two = [samples.get(c, 0) for c in ('training_00', 'training_01')]
            lines += [f'- 预算还严重不均：按步数计，各工况拿到 {values[0]}–{values[-1]} 步；'
                      f'training_00/01 各只有 {two[0]} 步'
                      f'（≈{two[0] / 2173:.2f} 个 episode 等量），因为它们 366 步就触杆终止了——'
                      '既是结构不可行，又拿了最少的经验。',
                      '  采样器本身是"每工况一回合轮转"（`BalancedCaseReset`），'
                      '因此这个不均完全来自早终止，不是采样偏差。', '']

    # ---- 5. 改造方案 ----
    lines += ['## 五、改造方案（让它进入"可以正式做强化学习"的状态）', '',
              '四项前置条件，建议按此顺序落地；每项都标注了影响面，'
              '验收判据本身不建议改动。', '',
              '1. **判据分层（必须先做）**：验收继续用 40 mrad / 250±10 mm 的硬阈值，'
              '但训练目标改成连续裕量。首选 4b 的方案 B（终局有界裕量，模块已备好，不动权重）；'
              '若只想调整形权重，取 0.5–1.0（取 2.0 会把整集整形推到 500 以上、回报尺度失衡）。'
              '权重来自 protocol，`tools/prepare_geometry_training_gate.py:33` 目前把 `quality_bonus_weight` 硬编码为 0.1，'
              '加一个 `--quality-bonus-weight` 参数、新建一份 protocol 即可，不改公式、不动 `check_endpoint`。',
              '2. **工况分层**：先用固定校准跑一次 x 方向扫描，划出可解窗口（当前推断为 ±0.45 mm），'
              '训练集只放窗口内点；窗口外的点单列"边界探测集"，不计入成功率，也不进奖励。'
              '影响面：几何切分协议（新建一份，不改已冻结的 20261027_v1）。',
              '3. **预算**：每工况 ≥20–50 episode（现在 3.6）。要么把 24 工况降到 8–12 个，'
              '要么把步数提到 2–5×10^6；按工况难度分层采样，避免平均分配。',
              '4. **结构性修复**：training_00/01 在 77.5675 s 恒定触杆（峰值 37.8–38.0 N，'
              '与基线时刻完全一致），说明该时刻的接触由参考轨迹/固定校准决定，'
              '残差（±0.015 rad）动不了；必须先改参考抬脚高度或膝/踝校准，再谈训练。', '',
              '另外两条记账建议：验证集（16 个保留几何）只在最终模型冻结后评一次；'
              '每个工况报多 seed 的均值与区间，避免单次评估把噪声当成能力。', '']

    # ---- 6. 训练曲线 ----
    if history:
        lines += ['## 六、训练过程采样', '', '| 时间 | 步数 | 完成 episode | 已激活工况 |', '|---|---|---|---|']
        for row in history[::max(1, len(history) // 30)]:
            lines.append(f"| {row['at'][11:]} | {row['additional_steps']} | "
                         f"{row['completed_episodes']} | {row.get('active_cases', '—')} |")
        lines.append('')
        lines += [f"- 采样记录 {len(history)} 条，原始文件 `.dsh-watch/training_history.jsonl`。", '']

    # ---- 7. 保留集 ----
    holdout = read(ROOT / 'outputs/calibrated_geometry_holdout16_20261027_v1/holdout_result.json')
    holdout_rows = read(ROOT / 'outputs/calibrated_geometry_holdout16_20261027_v1/after.json')
    lines += ['## 七、保留集（16 个保留几何，泛化证据）', '']
    if holdout:
        lines += [f'- 通过 {holdout["successes"]}/{holdout["cases"]}。这是训练模型的**唯一一次**保留集评估，'
                  '没有用它挑选检查点（符合你们"最终模型冻结后再评估"的约定）。']
        if holdout_rows:
            bad = [r['case'] for r in holdout_rows if not r['success']]
            lines += [f'- 未通过的保留工况：{bad if bad else "无"}。']
    else:
        lines += ['- 尚未完成（训练结束并通过四项审计后由监视器自动启动；若训练中断则不会启动）。']
    lines += ['- 注意：16 个保留点与 16 个训练点在同一个 1.45 mm × 0.14 mm 的局部区域里交错，'
              '这一结果只能说明"该局部区域"的泛化，不能外推为整面墙的泛化能力。', '']

    # ---- 8. 产物 ----
    lines += ['## 八、产物与复现', '',
              '| 文件 | 内容 |', '|---|---|',
              '| `.dsh-watch/geometry_verdict.json` | 基线 vs 训练后的逐工况判定与闭合/抬升数值 |',
              '| `.dsh-watch/feasibility_summary.json` | 可行域分析原始数据 |',
              '| `.dsh-watch/feasibility_map_baseline.png` | 可行域图 |',
              '| `.dsh-watch/backup/<时间戳>/` | 训练前审计产物备份（含 sha256 清单，已校验） |',
              '| `.dsh-watch/training_history.jsonl` | 训练过程采样 |',
              '| `.dsh-watch/after_training_analysis.py` | 生成判定表的脚本（口径与官方诊断一致） |', '',
              '复现判定表：', '',
              '```powershell',
              r'D:\G1Climbing\external\RoboMimic_Deploy\.venv\Scripts\python.exe '
              r'D:\G1Climbing\.dsh-watch\after_training_analysis.py '
              r'--run <训练目录> --baseline <基线目录> --output .dsh-watch\geometry_verdict.json',
              '```', '']

    out = WATCH / 'G1_几何PPO诊断报告_20261028.md'
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('report written:', len(lines), 'lines', 'verdict=yes' if verdict else 'verdict=pending')


if __name__ == '__main__':
    main()
