"""Freeze this research increment and preserve pending job handles."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
out = root / 'outputs/research_progress_20261017'
report = root / 'docs/baselines/research_progress_20261017.md'
s = report.read_text(encoding='utf-8').replace('最近一次查询剩余 54%', '最近一次查询剩余 53%')
extra = '''
58. 三组历史单周期网格补做保存状态原生位置审计：原控制 3/17、真实接触控制 8/17、试探落脚 8/17，扩展通过数均未减少，见各目录 `extended_native_acceptance.json`。这不消除连续周期中已发现的越界。54 项回归测试再次全部通过，日志 `research_tests_20261017_v11.log`。
59. `early_contact_two_cycle_v1` 的独立时序审计通过：8 次控制模式切换、2 次缺接触门控均有日志解释，时钟连续。该审计不重建每个物理步的接触几何。第二次换脚左踝跟踪曲线见 `repeated_ankle_tracking_v1.png/.pdf`，已查看排版。
60. `contact_frame_per_cycle_eight_v2` 完成 885.76 s，旧动态检查通过，位置检查仍失败：最小余量 -0.000839822 rad，1521 个保存样本越界。附加切换审计中模式时序与连续时钟通过，但要求每次切入增益为 100 的检查失败，因接触控制器保留上一周期终点的 300 增益。因此严格时间语义修正没有解决位置越界或全部增益继承问题。
61. 隔离消融 `early_contact_reset_two_cycle_v1` 每次新周期将接触控制器腿部增益明确重置为 100，其余控制逻辑与提前接触版本相同。两周期全程、逐物理步位置、独立位置、独立切换时序均通过；独立保存状态最小位置余量 0.018772126 rad。第一周期不受该重置改动影响。此结果支持“提前接触候选的成功不依赖高增益继承”，但尚不足以将改善唯一归因于某个接触参数。
62. 已启动 `early_contact_reset_eight_cycle_v1` 连续八周期及 `early_contact_reset_grid_v1` 预定义 17 工况。原提前接触八周期仍在运行，所有运行的源码单独冻结，禁止运行中改动导入模块。完成前不报告长程或误差网格成功率。

'''
if '58. 三组历史' not in s:
    s = s.replace('## 在跑与下一步', extra + '## 在跑与下一步')
s = s.replace('- `contact_frame_per_cycle_eight_v2`：原切换阶段加严格周期时钟，完成后进行切换审计及新增关节位置审计。', '- `contact_frame_per_cycle_eight_v2` 已完成，新增位置及增益检查失败，保留失败证据。')
s = s.replace('- 比较提前切换及增益继承，明确改善来自何种控制变化；不得放宽位置、力矩或碰撞阈值。', '- `early_contact_reset_eight_cycle_v1` 与 `early_contact_reset_grid_v1` 正在运行；分别检验连续两米和预定义杆位误差。不得放宽位置、力矩或碰撞阈值。')
report.write_text(s, encoding='utf-8')
p = out / 'checkpoint.json'
d = json.loads(p.read_text())
d['last_observed_remaining_percent'] = 53
d['running'] = [dict(task='Early inherited-gain eight cycles', session_id=15851, out='outputs/early_contact_eight_cycle_v1'), dict(task='Early reset-gain eight cycles', session_id=56574, out='outputs/early_contact_reset_eight_cycle_v1'), dict(task='Early reset-gain 17-case grid', session_id=99473, out='outputs/early_contact_reset_grid_v1')]
d['latest_results'].update(tests=54, historical_native_grid='3/17,8/17,8/17 unchanged after saved-state position audits', strict_clock_eight='Legacy pass, position FAIL (-0.000839822rad); phase schedule pass, gain reset audit FAIL', early_reset_two='All checks pass including 2000Hz native positions and independent schedule; reset gain100 each cycle')
d['next'] = ['Read 15851/56574/99473 and existing outputs before restarting.', 'Do not edit imported modules while grid cases are queued; freeze source per run.', 'Audit completed early eight runs with audit_joint_limits.py and audit_early_contact_schedule.py; inspect all eight endpoints. Render only with accurate badge.', 'After17 grid completes compare same reference/geometry/core hashes and extended native acceptance, retain regressions.', 'Use existing report through item62. Do not rerun older progress updater scripts: they contain historical snapshots.', 'Original training/climb_cycle.py and policy contracts unchanged.', 'Check quota periodically; remaining<=20 stop new experiments and give full report.']
p.write_text(json.dumps(d, indent=2), encoding='utf-8')
old = json.loads((out / 'source_manifest_v3.json').read_text())
paths = set(old)
paths.update(['deploy_mujoco/run_early_contact_cycle.py', 'deploy_mujoco/run_early_contact_reset_cycle.py', 'training/cycle_clock.py'])
paths.update(str(p.relative_to(root)).replace('\\', '/') for p in (root / 'tools').glob('*early_contact*'))
paths.update(['tools/audit_joint_limits.py', 'tools/audit_suite_native_positions.py', 'tools/run_rung_error_suite.py', 'tools/render_climb_cycle.py', 'tools/plot_repeated_ankle_tracking.py', 'tools/checkpoint_early_contact_v4.py', 'tools/audit_climb_cycle.py', 'tools/create_gain_reset_ablation.py'])
snapshot = out / 'source_snapshot_v4'
snapshot.mkdir(exist_ok=False)
hashes = {}
for rel in sorted(paths):
    source = root / rel
    target = snapshot / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    hashes[rel] = hashlib.sha256(target.read_bytes()).hexdigest()
(out / 'source_manifest_v4.json').write_text(json.dumps(hashes, indent=2))
print(f'Frozen {len(hashes)} files')
