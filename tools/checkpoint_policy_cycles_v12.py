"""Record continuous policy checks and exact regression diagnosis."""
import hashlib
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'outputs/research_progress_20261017'
report=ROOT/'docs/baselines/research_progress_20261017.md'
addition='''

## 99. 冻结局部PPO完整周期3/3，额外动作审计通过

`outputs/window_policy_fullcycles_v1`三个−3°窗口工况均完成110.72秒完整周期，运行验收、独立轨迹/原生关节限位、抓持杆身份、参考重建、50Hz策略动作复算、前次动作通道和200Hz滤波审计全部通过。49秒以前与零残差对照逐状态一致。原生位置最小余量分别0.01851778、0.01376541、0.01771922rad。参考位置和速度重建误差均0，检查点批量动作复算误差在2e−6阈值以内。

此结果只说明局部策略能接入既有完整动作；主体轨迹、接触控制和窗口外运动仍为既有方案。不能称为PPO学会完整攀爬，也不抵消其+3°场景中的退步。

## 100. PPO新增失败精确重放：一个右脚速度边界样本

`outputs/ppo32768_terminal_diagnostic_v2`重放零残差与PPO的杆后移10mm/+3°工况，qpos/qvel均与原始评估逐元素相同。PPO的支撑历史保留52个200Hz样本，从61.305至61.560秒，共0.255秒；最早样本右脚速度0.020214417782m/s，超过0.020m/s阈值，导致终端支撑未确认。右脚最低承重172.414N，两个钩爪承重充足且身份正确，左脚已卸载至0N。

该历史实现保留一个跨过0.25秒边界的旧样本。仅查看最后0.25秒会遗漏它，因此v1诊断不足，v2按运行时实际history重建。原训练和验收逻辑保持不变，仍记为失败。此结果是速度裕量不足的精确诊断，不是掉落或抓错杆，也不作为放宽阈值的依据。

## 101. 新几何比较已启动，结果待收集

`outputs/untrained_geometry_comparison_v1`固定32,768步策略和已选小偏置，在脚杆X−5/+5/+10mm、Z−5/−10mm五种几何上，分别测试0/−3/+3°法向偏差；零残差、固定偏置、PPO共45项。五种几何均不在本PPO训练分布中，但先前经典控制研究已见过它们，故不是盲测，也不称为预注册实验。检查点和动作在评估前冻结，全部场景均保留。

当前运行会话13050。实验完成后应检查每组独立审计，并对三种方法做相同前缀/源码哈希和逐场景得失比较；不要只报总通过数量。
'''
assert '## 99.' not in report.read_text(encoding='utf-8')
with report.open('a',encoding='utf-8') as f:f.write(addition)
c=json.loads((out/'checkpoint.json').read_text());c['running']=[dict(task='45 frozen-method evaluations on5untrained geometries',session_id=13050,out='outputs/untrained_geometry_comparison_v1')]
c['latest_results'].update(policy_fullcycles='3/3 fullcycles;runtime/native/independent/identity/action/reference/prefix audits PASS',ppo32768_failure='Exact replay qpos/qvel diff0;history52samples0.255s,oldestRFspeed0.0202144>0.02,only1invalid requiredsupport;force andidentity fine. Keepfail unchanged.')
c['next']=['Collect13050;9groups5cases each, zero/constant/PPO at0,-3,+3deg. No imports/source edits untilalljobscomplete.','Compare45results with sameprefix/corehashes and assertnewcasesdisjointparenttraining. Rootprotocolpredeclared. Allgroupaudits mustpass.','Reportthrough101/sourcev12. Remaining33lastobserved; pollquota andstopnewexperiments at20.','Potential next highrod endpoint geometric feasibility diagnosis; notyetimplemented. Highrodclosure remains unresolved.']
(out/'checkpoint.json').write_text(json.dumps(c,indent=2))
paths=set(json.loads((out/'source_manifest_v11.json').read_text()))
paths.update(['tools/diagnose_policy_terminal_support.py','tools/diagnose_policy_terminal_support_v2.py','tools/evaluate_untrained_geometries.py','tools/audit_untrained_policy.py','tools/checkpoint_policy_cycles_v12.py'])
snap=out/'source_snapshot_v12';snap.mkdir(exist_ok=False);hashes={}
for rel in sorted(paths):
    target=snap/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target);hashes[rel]=hashlib.sha256(target.read_bytes()).hexdigest()
(out/'source_manifest_v12.json').write_text(json.dumps(hashes,indent=2));print('Report101/sourcev12saved')
