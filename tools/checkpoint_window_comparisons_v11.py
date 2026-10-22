"""Append audited continuation and continuous constant-baseline evidence."""
import hashlib
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'outputs/research_progress_20261017'
report=ROOT/'docs/baselines/research_progress_20261017.md'
text='''

## 95. 32,768步续训完成：恢复一项、退步一项，总体仍14/15

`outputs/biased_load_transfer_ppo_32768_v1`已完成全部训练和15项训练后评估。父模型8,192步，新增24,576步。策略和优化器延续，但模拟器与随机数重新初始化，不是无缝仿真续跑。

零残差、父模型、续训后均14/15。续训恢复`row01_z_+5mm / -3°`，但新增`row01_x_-10mm / +3°`失败，原因`unconfirmed_loaded_tripod`，61.56秒到达而末端承重未确认。不能以新增成功替代总体比较。

独立45轨迹审计、父模型哈希、30份复用评估文件、训练步数与分布一致性审计全部通过。`paired_comparison.json`与`outputs/residual_methods_comparison_v2/summary.json`保存逐项得失。固定偏置与纯接触记忆仍15/15；区间锥加记忆11/15。全部为有限确定性工况，不是成功率概率估计。

## 96. 固定小偏置完整周期：3/3通过

`outputs/window_constant_fullcycles_v1`在nominal、杆高+5mm、杆后移10mm三个工况中，从0秒连续运行至110.72秒，全部通过。法向−3°仅作用49至61.56秒。右髋俯仰动作−0.25，比例0.015rad，稳态偏置−0.00375rad；200Hz以0.05系数滤波，61.56秒后平滑衰减至零。

原生关节位置最小余量分别0.01888396、0.01390928、0.01832010rad。运行验收、独立审计、杆身份审计、参考残差重建和配对前缀审计全部通过。49秒之前与对应零残差对照轨迹一致。固定偏置的作用不是人工拉力，没有状态替换。此结果说明现阶段记忆控制尚不具备相对于简单固定偏置的独有恢复优势。

新增参考审计最初因NumPy布尔值不能直接JSON序列化而退出；只修复审计序列化后重读原始数据，未重跑仿真。原始错误日志保留，修复后日志为`outputs/window_constant_fullcycles_v1_audit_fixed.log`。

复现（输出目录必须新建）：
```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
.venv\\Scripts\\python.exe tools/run_window_constant_experiment.py --out outputs/window_constant_fullcycles_reproduce --workers 2
# 对三个constant__子目录分别运行audit_cycle_contact_identity.py，然后：
.venv\\Scripts\\python.exe tools/audit_window_constant.py --suite outputs/window_constant_fullcycles_reproduce
```

## 97. 冻结PPO的完整周期接入验证正在运行

`outputs/window_policy_fullcycles_v1`使用已审计32,768步检查点，仅49至61.56秒提供局部残差，其余阶段仍由既有轨迹和控制器执行。没有重新训练整周期策略。观测接口与原训练环境在三个初始工况逐元素一致，50Hz策略、200Hz滤波、前次动作通道、平滑退出及输入参考不被修改均经过检查。

截至本节写入，三个完整周期及额外策略审计尚未全部完成，不预报通过。独立文件保存检查点来源、每次观测/动作和每个200Hz参考偏置。

## 98. 当前剩余任务与额度

最近查询账户剩余33%，停止新增实验阈值仍20%。下一步收齐完整周期策略结果和参考/动作审计；随后诊断PPO新退步的末端承重条件，并优先比较简单基线。高杆+10mm的姿态闭合仍未解决。2米连续攀爬通过的是既有控制方案，不能宣称上述局部PPO已学会2米攀爬。
'''
assert '## 95.' not in report.read_text(encoding='utf-8')
with report.open('a',encoding='utf-8') as f:f.write(text)
checkpoint=json.loads((out/'checkpoint.json').read_text())
checkpoint['last_observed_remaining_percent']=33
checkpoint['running']=[dict(task='Frozen PPO continuous full cycles',session_id=54800,out='outputs/window_policy_fullcycles_v1')]
checkpoint['latest_results'].update(ppo32768='14/15: rescues Z+5/-3, loses X-10/+3;45audits and continuation audit PASS',constant_fullcycles='3/3 full110.72s;runtime/native/independent/contactidentity/reference/prefix audits PASS',policy_fullcycles='running54800; adapter contract checks PASS')
checkpoint['next']=['Collect54800; run contact identity audits on3policy__cases then audit_window_policy.py --suite outputs/window_policy_fullcycles_v1. Audit script generated before run; no imported source edits while running.','PPO32768 complete and independently audited; no aggregate improvement over zero14/15. Constant full3/3 with proven matchedprefix49s.','Diagnose X-10/+3 terminal loadedtripod regression; forcechannels currently ideal.','Reportthrough98/sourcev11; quota33 lastseen, stopnewexperiments at20. Highrodclosure unresolved.']
(out/'checkpoint.json').write_text(json.dumps(checkpoint,indent=2))
paths=set(json.loads((out/'source_manifest_v10.json').read_text()))
paths.update(['tools/create_window_constant_experiment.py','deploy_mujoco/run_window_constant_cycle.py','tools/run_window_constant_experiment.py','tools/audit_window_constant.py','training/window_residual_policy.py','tools/create_window_policy_experiment.py','deploy_mujoco/run_window_policy_cycle.py','tools/run_window_policy_experiment.py','tools/create_window_policy_audit.py','tools/audit_window_policy.py','tools/check_window_policy_contract.py','tools/checkpoint_window_comparisons_v11.py'])
snapshot=out/'source_snapshot_v11';snapshot.mkdir(exist_ok=False);hashes={}
for rel in sorted(paths):
    target=snapshot/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target);hashes[rel]=hashlib.sha256(target.read_bytes()).hexdigest()
(out/'source_manifest_v11.json').write_text(json.dumps(hashes,indent=2))
print('Report through98 and sourcev11 saved')
