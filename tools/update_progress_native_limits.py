"""Record the revised acceptance contract without rewriting historical evidence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
report = ROOT / 'docs/baselines/research_progress_20261017.md'
s = report.read_text(encoding='utf-8')
s = s.replace('最近一次查询剩余 60%', '最近一次查询剩余 54%')
notice = '''
## 最新验收更正

历史条目中的“严格通过”指当时的 12 项运行检查和 12 项独立审计，不包含新增的原生关节位置范围检查。旧结果保留，但不能据此宣称全部模型约束通过。原控制与每周期接触控制均完成连续八周期、885.76 s、约两米爬升，八个周期端点通过旧检查；新增位置检查发现左踝俯仰轻微越界，因此扩展验收均失败。此前连续两周期的旧结果也存在同类越界。

提前启用真实接触控制的新候选已完成两周期，221.44 s、爬升 0.498483 m，全部旧检查及每个 2000 Hz 物理步的位置检查通过，最小原生关节余量 0.01875587 rad。独立 50 Hz 保存状态检查通过。两米扩展尚在运行，不能提前宣称长程成功。该控制使用 MuJoCo 接触真值，尚非训练策略或真实感知方案。

'''
if '## 最新验收更正' not in s:
    s = s.replace('## 已完成的增量', notice + '## 已完成的增量')
extra = '''
52. `verified_memory_segments_v1` 完成九例，8/9，与未加求解复核的记忆控制相同。X=-10 mm/+3 度在 54.130..54.160 s 恢复七次错误不可行报告，但 54.165 s 原约束经独立 LP 确认为不可行，仍停止。`solver_verification_audit.json` 对八次分类全部通过；未扩大力矩界限，不能称为新增任务成功。
53. 两米旧控制记录：原控制爬升 1.999562 m，最小位置余量 -0.000355101 rad，128/44289 个保存样本越界；每周期接触控制爬升 1.998658 m，最小余量 -0.000839822 rad，1518 个样本越界。最差均为左踝俯仰、158.28 s。`tools/audit_joint_limits.py` 保留独立附加结论，未篡改旧审计。新实验套件将此检查纳入通过条件。保存采样检查不等于采样间保证。
54. 周期时钟已统一为与参考一致的严格半开区间，新增控制切换日志与边界测试。`contact_frame_per_cycle_eight_v2` 正在长程回归；旧运行保留。54 项测试通过，见 `research_tests_20261017_v10.log`。
55. 左踝越界时参考值约 0.380715 rad，原生上界 0.5236 rad，实际达到 0.524440 rad；参考已有约 0.143 rad 余量，不能归因于参考贴近限位。新候选从每周期 step_preparation 阶段使用实际接触控制；若计划承载肢体缺少杆接触，则临时回到原电机控制建立接触。`early_contact_two_cycle_v1` 完成两周期并通过新增逐物理步检查，最大身体跟踪误差 5.010 mm，无非预期、摆脚或自碰撞。两次接触门控均发生于右脚开始加载，持续一个 5 ms 控制周期。
56. 提前切换候选第二周期进入准备阶段的控制增益记录为 300，源于上一周期终点状态；随后按原阶段增益逻辑变化。该行为已记录，不能把候选解释为仅改变接触点而其他控制时序完全相同。需独立消融确认增益继承的影响，不在现有运行中修改源码。
57. 原控制两米完整 8 倍速录像已生成，双视角预览已检查，明确显示“诊断录像：关节位置限位未通过”。视频位于 `clearance_eight_cycle_baseline_v1/nominal/cycle_recorded_8x.mp4`，不可作为全部验收通过的演示。

'''
if '52. `verified_memory_segments_v1`' not in s:
    s = s.replace('## 在跑与下一步', extra + '## 在跑与下一步')
start = s.index('## 在跑与下一步')
end = s.index('## 失败实验保留', start)
s = s[:start] + '''## 在跑与下一步

- `early_contact_eight_cycle_v1`：提前接触控制连续八周期，逐物理步检查原生位置范围；完成后复核八个端点、独立状态及完整录像。
- `contact_frame_per_cycle_eight_v2`：原切换阶段加严格周期时钟，完成后进行切换审计及新增关节位置审计。
- 比较提前切换及增益继承，明确改善来自何种控制变化；不得放宽位置、力矩或碰撞阈值。
- 学习基线尚无性能提升；接触控制仍依赖仿真真值。先固定完整动力学验收，再扩展误差工况及感知约束。
- 剩余额度达到 20% 时停止新实验并汇总。

''' + s[end:]
report.write_text(s, encoding='utf-8')
p = ROOT / 'outputs/research_progress_20261017/checkpoint.json'
d = json.loads(p.read_text())
d['last_observed_remaining_percent'] = 54
d['running'] = [dict(task='Eight cycles early contact with per-step position audit', session_id=15851, out='outputs/early_contact_eight_cycle_v1'), dict(task='Eight cycles strict clock original transfer phase', session_id=40886, out='outputs/contact_frame_per_cycle_eight_v2')]
d['latest_results'].update(tests=54, verified_memory='8/9; seven numerical recoveries then true infeasibility at54.165s', two_cycles='OLD legacy passes fail new position audit. Early candidate passes all checks plus 2000Hz position bounds; rise0.498483m', eight_cycles='Old baseline and hybrid complete2m/885.76s, legacy pass, native position audit FAIL', early_contact='Two cycles pass; minimum position margin0.01875587rad. Eight cycles running')
d['next'] = ['Read existing running jobs before restarting.', 'Audit v2 strict switch events and native positions after finish.', 'Audit early eight-cycle per-step and independent checks, render truthful video.', 'Investigate inherited gain300 at second preparation; preserve live source and run isolated ablation.', 'Keep original training/climb_cycle.py unchanged; no torque/position bound relaxation.', 'Check quota; remaining<=20 stop new experiments and report.']
p.write_text(json.dumps(d, indent=2), encoding='utf-8')
