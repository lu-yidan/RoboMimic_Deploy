"""Record completed long-horizon, robustness and prefix studies."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
out = root / 'outputs/research_progress_20261017'
report = root / 'docs/baselines/research_progress_20261017.md'
summary = json.loads((root/'outputs/early_contact_reset_eight_cycle_v1/nominal/summary.json').read_text())
continuity = json.loads((root/'outputs/contact_load_transfer_eight_prefix_v1/continuity_comparison.json').read_text())
maximum = max(r['qpos_max_difference'] for r in continuity['results'])
s = report.read_text(encoding='utf-8')
s = s.replace('显式增益重置版本仍独立验证中。', '显式增益重置版本也完成八周期并通过相同附加审计。')
extra = f'''
73. `early_contact_reset_eight_cycle_v1` 完成连续八周期，实际爬升 {summary['final']['base_rise_m']:.9f} m，逐物理步关节最小余量 {summary['minimum_native_joint_margin_rad']:.9f} rad。原运行、独立审计及八个端点通过；附加位置、切换及全过程保存杆位检查通过。354599 条承载接触均在预期杆上。说明此前两米成功不依赖上一周期高增益继承。
74. `early_contact_reset_grid_v1` 完成全部 17 工况：整体通过 10/17、全程完成 15/17。`early_contact_grid_comparison_v1` 核对共同参考、几何/BVH、控制核心源码及增益参数后，确认较此前接触控制 8/17 新增手杆 X=-5 mm、Z=-5 mm 两例，无原成功工况退化；原固定点控制为 3/17。全部新工况含运行时位置检查及独立保存状态位置审计。确定性网格不是统计成功概率。
75. `contact_load_transfer_eight_prefix_v1` 八条零残差片段全部成功，独立审计 8/8 通过；与各自连续轨迹的最大 qpos 差为 {maximum:.3g}。同时单独保留 Z=+10 mm 的失败前缀。57 项回归测试全部通过，见 `research_tests_20261017_v13.log`。
76. `early_contact_bias_segments_v1` 九个新前缀法向偏差片段完成，8/9 成功、9/9 完整性审计通过。唯一失败为脚杆 Z=+5 mm、法向偏差 -3 度，在 53.700 s 失去左钩实际接触后控制器拒绝继续分配。它与旧前缀的 6/9 使用不同动态初态，不能仅归因于法向控制改进。下一步以这组有明确失败的偏差片段开展残差训练，保留零动作、训练前、训练后完整对照，尚未宣称学习改善。

'''
if '73. `early_contact_reset_eight' not in s:
    s = s.replace('## 在跑与下一步', extra+'## 在跑与下一步')
start=s.index('## 在跑与下一步');end=s.index('## 失败实验保留',start)
s=s[:start]+'''## 在跑与下一步

- 两种提前接触控制的连续两米验证、17 工况网格、八个合格训练前缀及九个法向偏差片段均已完成。
- 下一步针对新前缀法向偏差任务做残差训练；零动作基线为 8/9，保留全部失败、独立审计与训练前后对照。准确接触拓扑及力反馈仍是限制。
- 全程杆位误差中仍有低杆闭合、高杆摆脚碰撞及手杆分配失败，不将片段学习或整体时长完成等同于解决这些问题。
- 剩余额度达到 20% 时停止新实验并汇总。

'''+s[end:]
report.write_text(s,encoding='utf-8')
p=out/'checkpoint.json';d=json.loads(p.read_text())
d['running']=[]
d['latest_results'].update(tests=57,early_reset_eight='PASS all runtime/independent/endpoints/native/schedule/contactidentity audits',early_grid='10/17 pass,15complete; prior8/17,base3/17; addedhandX-5/Z-5,no lost passes',contact_training_baseline='8/8 valid prefixes zero residual and audits pass; ninth highZ+10 prefix invalid',new_prefix_normal_bias='8/9 success,9/9 integrity. OnlyZ+5/normal-3 fails53.7s absent left-hook contact. PPO not started')
d['next']=['All prior jobs complete; do not restart. Report through76; sourcesv6+v7.', 'Implement isolated biased-contact residual training on new prefix suite, preserving76obs/6RFactions/0.015rad bounds and perphysics native checks. Reuse BiasedContactFrameController, keep exact topology/force limitation explicit.', 'Evaluate zero/before/after all3prefixes x angles0,-3,+3; predeclare training distribution and heldout angles if used. FailureZ+5/-3 at53.7 is known, do not cherry-pick only successful episodes. Train finite budget with checkpoint reload and full audit.', 'Original training/climb_cycle.py and old policy contracts unchanged.', 'Full-cycle grid still has7failures; no claim segmenttraining fixesfullcycle or real perception.', 'Quota latest48%remaining. Continue until<=20 then stop new experiments and give fullreport.']
p.write_text(json.dumps(d,indent=2),encoding='utf-8')
snapshot=out/'source_snapshot_v7';snapshot.mkdir(exist_ok=False)
paths=set(json.loads((out/'source_manifest_v6.json').read_text()))|{'tools/checkpoint_completed_grid_v7.py'}
hashes={}
for rel in sorted(paths):
    target=snapshot/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/rel,target)
    hashes[rel]=hashlib.sha256(target.read_bytes()).hexdigest()
(out/'source_manifest_v7.json').write_text(json.dumps(hashes,indent=2))
print('Completed research records saved')
