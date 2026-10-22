"""Archive no-gap persistence results, bounded dropout behavior and metadata errata."""
import hashlib
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'outputs/research_progress_20261017';report=ROOT/'docs/baselines/research_progress_20261017.md'
s=report.read_text(encoding='utf-8');assert '## 110.' not in s
s=s.replace('最近一次查询剩余 27%；','最近一次查询剩余 24%；')
s+='''

## 110. 去掉当前间隙查询：30项局部测试全部通过

`outputs/known_force_history_v1/study_audit.json`原三种几何×五种角度15/15通过；`outputs/untrained_force_history_v1/study_audit.json`另五种几何×三种角度15/15通过。全部轨迹、源码/前缀、缓存年龄/计划支撑审计通过。新几何15项相对原几何记忆的qpos逐元素相同，缓存使用时刻也相同。

因此这些工况中不需要当前精确钩杆间隙门限：过去≥10N承重确认与≤20ms短时接触保持已足以复现成功。但控制仍使用当前有效接触的精确点/拓扑和理想力反馈，缓存也使用身体位姿传播；不能称为完成真实感知部署。

## 111. 无间隙查询的三个完整周期通过

`outputs/window_force_history_fullcycles_v1/paired_audit.json`nominal、脚杆Z+5mm、脚杆X−10mm均完成110.72秒。运行、独立轨迹、原生关节范围、接触杆身份、缓存边界和配对审计均通过；全部qpos/qvel与先前几何记忆完整周期逐元素相同。Z+5mm使用4次缓存，另外两项未使用缓存。

窗口仍仅49至61.56秒，法向−3°。这里消除的是一项当前几何真值查询，不代表任意脱钩都能恢复。

## 112. 接触定位检测中断8项实验：短中断恢复，长中断到期终止

`outputs/visibility_dropout_study_v1/study_audit.json`在名义工况53.65秒开始仅屏蔽控制器的左钩接触定位信息，物理接触和理想力反馈保持不变。无保持与有保持各测试0、10、20、30ms，全部原始数据审计通过，预先声明的有限时长行为全部符合。

0ms两者通过；10/20ms无保持立即分配失败，有保持恢复并完成至61.56秒；30ms有保持在53.670秒因缓存过期终止诊断，无保持仍在53.65秒失败。失败仍记录为任务失败，没有把“符合预期终止”计入攀爬成功。此实验不是物理脱离，也没有证明真实机器人安全停机。

## 113. 局部PPO元数据显式更正

检查发现`run_window_policy_cycle.py`从固定控制器继承了`trained_policy=False`，与实际使用已训练局部残差不符。后续输出改为`trained_policy=True, learned_full_cycle=False`。原三个策略实验的summary、轨迹和冻结源码均保留，分别新增`metadata_erratum.json`记录原摘要哈希、正确含义及策略/动作审计证据。

录像工具识别并验证`policy_provenance.json`和策略动作审计，明确标注“仅49–61.56s使用训练残差”，不再将此类录像写成尚非学习策略。策略Z+5mm预览已重新生成并查看，未重跑物理实验。旧高杆录像不使用学习策略，原标签正确且仍保留。

## 114. 双周期接触历史实验启动，结果尚未完成

`outputs/periodic_force_history_two_cycles_v1`会话79191：使用已验证双周期参考，连续221.44秒，每个周期局部49至61.56秒都施加−3°法向偏差。测试名义场景，以及rung_01/rung_02均升高5mm场景。每周期重置控制增益，物理状态不重置。新周期窗口起止边界检查已通过。

这与前面只在单周期窗口启用不同，需独立检查第二周期的切换、缓存时限、原生限位和最终接触。当前不预报通过。最近账户剩余24%，继续20%停止新增实验和整理报告的条件。
'''
report.write_text(s,encoding='utf-8')
c=json.loads((out/'checkpoint.json').read_text());c['last_observed_remaining_percent']=24;c['running']=[dict(task='Periodic forcehistory two cycles nominal andtwohighfootrods',session_id=79191,out='outputs/periodic_force_history_two_cycles_v1')]
c['latest_results'].update(force_history_local='30/30;known15 andnew15;newqpos exactmatch geometricmemory;all audits PASS',force_history_fullcycles='3/3;fullqpos/qvel exactmatch geometricmemory;allruntime/native/independent/identity/paired audits PASS',visibility_dropout='8/8integrity andexpectedbehavior. Rawonly0ms passes; persistence0/10/20pass,30expires at53.670. Failuresnotcountedasclimbsuccess.',policy_metadata='Futuretrained_policy flag correctedtrue withlearned_full_cyclefalse; original3summaries preserved+errata; renderer verifiedcorrectpreview')
c['next']=['Collect79191 first;2continuous221.44s cases nominal andrung01+02Z+5. Explicit cyclelocalbias49..61.56 eachcycle. No sourceedits while running.','After complete run audit_cycle_contact_identity.py, audit_early_contact_schedule.py percase. Runtime anddriverindependent/native audits exist; checkeveryendpoint andmemoryevents cycle-localage.','Reportthrough114/sourcev15; lastquota24. Stopnewexperiments at<=20 andpreparefinalconsolidatedreport.','All otherjobs completed/audited. Keepforcehistoryscope limited; dropout30msdiagnosticstop isNOTsafehold orphysicalseparation recovery.','PPOcycletrained_policyfalsehistoricalmetadata corrected via3metadata_erratum files; oldsource/summary untouched. Newrunner andgenerator corrected. Currentrenderer sourcechanged; oldvideos remainvalidfrozen.']
(out/'checkpoint.json').write_text(json.dumps(c,indent=2))
paths=set(json.loads((out/'source_manifest_v14.json').read_text()))
paths.update(['training/window_force_history_controller.py','deploy_mujoco/run_window_force_history_cycle.py','tools/run_window_force_history_experiment.py','training/visibility_dropout_controller.py','tools/run_visibility_dropout_study.py','tools/audit_visibility_dropout_study.py','tools/audit_force_history_fullcycles.py','training/periodic_force_history_controller.py','deploy_mujoco/run_periodic_force_history_cycle.py','tools/run_periodic_force_history_experiment.py','tools/checkpoint_force_history_v15.py','tools/render_climb_cycle.py'])
snap=out/'source_snapshot_v15';snap.mkdir(exist_ok=False);hashes={}
for rel in sorted(paths):
    target=snap/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target);hashes[rel]=hashlib.sha256(target.read_bytes()).hexdigest()
(out/'source_manifest_v15.json').write_text(json.dumps(hashes,indent=2));print('Report114/sourcev15saved')
