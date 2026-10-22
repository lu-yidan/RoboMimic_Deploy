"""Archive audited PPO null result and continuous memory recovery evidence."""
import hashlib
import json
from pathlib import Path
import shutil
root=Path(__file__).resolve().parents[1];out=root/'outputs/research_progress_20261017'
report=root/'docs/baselines/research_progress_20261017.md';s=report.read_text(encoding='utf-8').replace('最近一次查询剩余 43%','最近一次查询剩余 38%')
extra='''
83. `early_contact_memory_only_v1` 15/15 片段成功，15 条完整性审计及记忆时间/距离/阶段边界检查通过。该结果优于同组“区间锥加记忆”的 11/15，也优于原偏差控制的 14/15。它依赖准确几何间隙和此前理想力测量，不能称为真实传感器验证。不要把摩擦锥收紧的退化归因于接触记忆。
84. 高净空三例全部完成 113.66 s，正常与 Z+5 mm 通过；Z+10 mm 摆脚峰值降为 0 N，逐物理步位置最小余量 0.00159850 rad，但终点关节误差 0.07318775 rad 仍超过 0.04 rad，整体仍失败。未降低闭合标准。
85. `biased_load_transfer_ppo_8192_v1` 完成 8192 步、训练耗时 227.13 s。零动作、训练前、训练后均为 14/15：训练角度 8/9、评估专用 ±1.5 度 6/6；没有新增成功或退化。45 条独立审计全部通过；checkpoint 重载动作完全一致，独立逐记录与批量重放最大差分别约 4.19e-9、2.98e-8。难例仍失去左钩，不能宣称这轮 PPO 改善控制。
86. `residual_authority_probe_v1` 预声明零动作及六个右腿关节各 ±0.25/±0.5 的 25 个常量动作，在同一已知难例测试，16/25 完成且独立审计通过，25/25 审计通过。残差仍经原滤波和 0.015 rad 缩放。它证明已测试动作范围中存在简单解，不是学习、泛化或 16/25 统计概率证据。未将搜索结果冒充训练后策略。
87. `window_memory_fullcycles_v1` 六条从零时刻连续运行的对照，只在 49..61.56 s 加 -3 度法向偏差，窗口内不允许丢接触后退回原控制来掩盖失败。无记忆为 2/3，Z+5 mm 在 53.700 s 停止；有记忆 3/3 完成并通过原运行、独立、位置及保存杆位归属检查。成对审计核对参考、几何、源码及除记忆开关外配置完全相同；49 s 前 qpos 完全一致。正常与 X-10 两例全程 qpos 也完全一致。难例使用四次记忆，最大年龄 10 ms、最大几何间隙 12.847 微米。偏差不是全周期持续存在，不能外推为全过程感知鲁棒性。
88. 上述完整周期的汇总脚本误将 `run_climb_cycle` 的空返回值当作结果，导致六条物理运行及其独立审计完成后汇总失败。已修正为读取保存的 `summary.json`，通过 `recover_window_memory_results.py` 恢复汇总；原错误脚本、错误日志和恢复说明保留，未重跑或修改物理轨迹。
89. 从独立审计通过的 8192 步 checkpoint 启动 `biased_load_transfer_ppo_32768_v1`，追加 24576 步，保持同一训练分布及评估场景。零动作及训练前结果分别复用父实验零动作和训练后记录，逐文件来源与 SHA256 写入 manifest；策略/优化器续接，环境与随机数重新初始化，不声称逐位不间断续训。最终仍需新训练后评估与独立审计，目前没有新增学习结论。

'''
if '83. `early_contact_memory_only' not in s:s=s.replace('## 在跑与下一步',extra+'## 在跑与下一步')
start=s.index('## 在跑与下一步');end=s.index('## 失败实验保留',start)
s=s[:start]+'''## 在跑与下一步

- 累计 32768 步残差 PPO 正在续训，完成后独立审计全部复用与新评估记录、checkpoint 及动作重放。
- 纯接触记忆的 15 场景和限定窗口完整周期验证已完成；仍需保留准确几何间隙、理想力及有限偏差窗口的限制。
- 高净空候选解决高杆摆脚碰撞与位置越界，尚未解决周期闭合，不替换已接受的两米基线。
- 剩余额度达到 20% 时停止新实验并汇总。

'''+s[end:]
report.write_text(s,encoding='utf-8')
p=out/'checkpoint.json';d=json.loads(p.read_text());d['last_observed_remaining_percent']=38
d['running']=[dict(task='PPO continuation to32768',session_id=99626,out='outputs/biased_load_transfer_ppo_32768_v1'),dict(task='Regression60 tests',session_id=48199,out='outputs/research_tests_20261017_v15.log')]
d['latest_results'].update(ppo8192='Zero/before/after14/15;45audits pass; no improvement; parentcheckpoint audited',memory_only='15/15,15audits andmemorybounds pass',constant_actions='16/25success,25/25audits on one known hardcase; not learned or generalization',window_memory='Raw2/3 vs memory3/3 fullcycles; matchedsource/geometry/refs/config; prefixidentical, unaffectedfulltrajectoriesidentical; only49..61.56bias; max10ms/12.847um',high_clearance='2/3;Z+10 collision0 andnativepass butclosure0.07318775rad fails',tests='59 plusnewwindowtest passed; fullv15 pending inspection')
d['next']=['Read99626; do not restart. Parent8192complete; all other experiment jobs finished. v15 regressioncheckpending.', 'When32768summary exists run audit_biased_load_transfer.py, add separate verification of manifest.reused_evaluations against parent files and hashes; summarizepaired results. Summarizer memory description currentlyassumescone, so do not labelmemoryonlywiththattext.', 'Validate recorded warmstart/parent provenance, no learningimprovementclaimuntilaftereval.', 'Reports through89; sourcev9. Fullwindowcollector bug recovered from saved files, no physics rerun.', 'Future useful step: validate a predeclared minimum-norm successful constant action across15cases, keeping selectiononknownhardcase explicit; notyetdone. Or investigate highrodclosure with same hard thresholds; no fakepass.', 'No source changes to modules loadedby activecontinuation; all oldpolicy contracts preserved.', 'Quota38remaining; stop newexperiments when<=20 and delivercompleteprogressreport.']
p.write_text(json.dumps(d,indent=2),encoding='utf-8')
paths=set(json.loads((out/'source_manifest_v8.json').read_text()))
paths.update(['training/window_contact_controller.py','deploy_mujoco/run_window_memory_cycle.py','tools/run_window_memory_experiment.py','tools/recover_window_memory_results.py','tools/audit_window_memory_pairs.py','tools/probe_residual_authority.py','tools/audit_constant_residual.py','tools/audit_residual_authority_suite.py','tools/continue_biased_load_transfer.py','tests/test_window_contact_controller.py','tools/checkpoint_memory_and_ppo_v9.py'])
snapshot=out/'source_snapshot_v9';snapshot.mkdir(exist_ok=False);hashes={}
for rel in sorted(paths):
    target=snapshot/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/rel,target);hashes[rel]=hashlib.sha256(target.read_bytes()).hexdigest()
(out/'source_manifest_v9.json').write_text(json.dumps(hashes,indent=2));print('Audited evidence and live continuation saved')
