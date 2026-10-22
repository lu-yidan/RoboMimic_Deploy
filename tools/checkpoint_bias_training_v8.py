"""Preserve active PPO, classical ablation and high-clearance experiments."""
import hashlib
import json
from pathlib import Path
import shutil

root=Path(__file__).resolve().parents[1];out=root/'outputs/research_progress_20261017'
report=root/'docs/baselines/research_progress_20261017.md'
train=root/'outputs/biased_load_transfer_ppo_8192_v1'
counts={label:sum(r['success'] for r in json.loads((train/(label+'.json')).read_text())) for label in ['zero','before']}
s=report.read_text(encoding='utf-8').replace('最近一次查询剩余 48%','最近一次查询剩余 43%')
extra=f'''
77. 启动 `biased_load_transfer_ppo_8192_v1`，8192 步 PPO，保留 76/6 观测动作契约、0.015 rad 右腿残差上限及原生约束。训练回合按预声明分布：已知 Z+5/-3 度难例占 50%，其余八个训练场景共享 50%；这不是转移样本的 50%。评估为三种杆位乘 0/-3/+3/-1.5/+1.5 度，共 15 工况；±1.5 度未参与训练。零动作 {counts['zero']}/15、训练前 {counts['before']}/15，已知失败在训练前仍未解决。训练尚未完成，不能报告学习收益。保存了训练前 checkpoint，准备训练后重载及动作复核。
78. 偏差环境新增测试：偏差标签未加入 actor 观测，重置不会改变真实几何及初始状态，随机种子可复现。全部 59 项回归测试通过，日志 `research_tests_20261017_v14.log`。独立训练审计入口 `audit_biased_load_transfer.py` 将检查 45 条评估记录、原生位置、来源、动作范围及 checkpoint 动作重放；汇总工具保留新增成功和丢失成功。
79. `early_contact_memory_comparison_v1` 在相同 15 场景上测试“法向区间锥加短时记忆”，通过 11/15；三个 +3 度分配失败，X-10/+1.5 度终点支撑确认失败。15 条完整性审计与记忆事件边界检查通过，不等于任务成功。该方法额外依赖已知不确定性方向及准确几何间隙，不能作为同信息量的学习公平对照。
80. 启动 `early_contact_memory_only_v1` 消融，关闭角度区间收紧，仅保留 <=20 ms 且几何间隙 <=0.1 mm 的接触记忆。已测难例 Z+5/-3 度恢复成功，使用四次缓存；完整 15 工况仍在运行。必须等全结果检查退化；不得将区间锥与记忆贡献混为一谈。
81. 高杆 Z+10 mm 摆脚诊断：2000 Hz 累计峰值在 42.08 s 的右摆脚窗口已达 34.97 N，而 50 Hz 接触快照没有记录该次接触；左摆脚 74.90 s 保存快照记录到 21.88 N。不能只用保存接触快照判断无碰撞。原高杆还存在原生位置及闭合失败，不是仅碰撞失败。
82. `early_swing14_lift26_plan_v1` 将规划摆动净空从 8 提高至 14 mm、右脚卸载抬升从 18 提高至 26 mm、接近抬升从 15 提高至 25 mm，其余原参数保持，密集几何审计通过。新周期 113.66 s，最小参考关节余量 0.01798844 rad。`early_swing14_lift26_dynamic_v1` 正常和 Z+5 mm 已完整通过，Z+10 mm 正在运行。它改变了相位时序，不能直接套用旧 49..61.56 s 训练窗口。

'''
if '77. 启动' not in s:s=s.replace('## 在跑与下一步',extra+'## 在跑与下一步')
start=s.index('## 在跑与下一步');end=s.index('## 失败实验保留',start)
s=s[:start]+'''## 在跑与下一步

- 偏差残差 PPO 8192 步正在运行，完成后审计全部 45 条评估记录及 checkpoint；报告新增成功和退化，不只看奖励。
- 纯接触记忆消融及高净空高杆动力学仍在运行；保留经典对照的额外信息假设，检查全部原生约束。
- 两种两米基线、17 工况主网格、八个合格训练前缀均已完成。学习片段尚不能替代完整攀爬策略。
- 剩余额度达到 20% 时停止新实验并汇总。

'''+s[end:]
report.write_text(s,encoding='utf-8')
p=out/'checkpoint.json';d=json.loads(p.read_text());d['last_observed_remaining_percent']=43
d['running']=[dict(task='PPO8192 with45evaluations',session_id=24097,out='outputs/biased_load_transfer_ppo_8192_v1'),dict(task='Memory only15cases',session_id=26707,out='outputs/early_contact_memory_only_v1'),dict(task='Highclearance3cases',session_id=68068,out='outputs/early_swing14_lift26_dynamic_v1')]
d['latest_results'].update(tests=59,biased_ppo=f'Zero{counts["zero"]}/15 andbefore{counts["before"]}/15; learning underway; afterunknown',classical_memory_with_cone='11/15;15audits andmemorybounds pass; addsinfo assumptions',memory_only_partial='HardZ+5/-3 rescued with4memoryuses; whole15pending',high_clearance='Planpassed113.66s; nominal/Z+5fullpass;Z+10running; new phase timing not oldtrainingwindow')
d['next']=['Inspect24097,26707,68068 before restarting; no imported-module edits while workers run.', 'After PPO finishes: tools/audit_biased_load_transfer.py --run outputs/biased_load_transfer_ppo_8192_v1, then summarize_biased_training.py --run same --memory outputs/early_contact_memory_comparison_v1. No claim until45audits/reload/actionreplaypass. Record lostsuccesses.', 'After memoryonly finishes audit_contact_memory_events.py; compare15sameprefixes with rawzero and cone+memory. Extra exactgap assumption persists. Current summarizer --memory description assumescone, so adapt separate output label for memoryonly before using.', 'After highclearance3 finish run audit_suite_native_positions.py, contactidentityaudit oncompletecases, compareold3cases/phasechanges. IfZ+10passes testbroader grid before replacingnominal.', 'Do not use new113.66s plan with hard-coded49..61.56s trainingwindow.', 'Original policy contracts unchanged; sourcesv8/reportthrough82.', 'Quota43%remaining; stop newexperiments at<=20 and report.']
p.write_text(json.dumps(d,indent=2),encoding='utf-8')
paths=set(json.loads((out/'source_manifest_v7.json').read_text()))
paths.update(['training/biased_load_transfer_residual_env.py','tools/train_biased_load_transfer.py','tools/audit_biased_load_transfer.py','tools/summarize_biased_training.py','tools/run_early_memory_suite.py','tools/run_early_memory_only_suite.py','tools/audit_contact_memory_events.py','tools/diagnose_swing_contacts.py','tests/test_biased_load_transfer_env.py','tools/checkpoint_bias_training_v8.py'])
snapshot=out/'source_snapshot_v8';snapshot.mkdir(exist_ok=False);hashes={}
for rel in sorted(paths):
    target=snapshot/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/rel,target);hashes[rel]=hashlib.sha256(target.read_bytes()).hexdigest()
(out/'source_manifest_v8.json').write_text(json.dumps(hashes,indent=2))
print('Active experiments and evidence saved')
