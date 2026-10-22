"""Checkpoint completed generalization and matched highrod closure experiment."""
import hashlib
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'outputs/research_progress_20261017';report=ROOT/'docs/baselines/research_progress_20261017.md'
s=report.read_text(encoding='utf-8');assert '## 102.' not in s
s=s.replace('五种未参与本PPO训练的几何，三种法向偏差、三种方法，共45项比较正在收尾。不能将部分结果写成总结果。','五种未参与本PPO训练的几何，三种法向偏差、三种方法，共45项比较完成；零残差12/15，固定偏置13/15，PPO13/15，全部独立审计通过。')
s=s.replace('先前去碰撞方案仍闭合失败。现已找到满足原0.04rad闭合边界的离线姿态，独立适配参考正在完整动力学验证；未验收前不算解决。','先前去碰撞方案闭合失败；已知几何的终点姿态适配现已通过完整113.66秒及附加审计。仅已知+10mm单工况，尚无误差鲁棒性或多周期保证。')
s+='''

## 102. 五种未训练几何的45项比较完成

`outputs/untrained_geometry_comparison_v1/paired_summary.json`已核实全部场景、独立审计、检查点/前缀及共享源码哈希。各方法15项：零残差12通过，固定偏置13通过，PPO13通过。按0/−3/+3°拆分分别是零残差5/2/5，固定偏置5/3/5，PPO5/4/4。

固定偏置仅恢复Z−5mm/−3°且无新增失败。PPO恢复X+5、X+10、Z−5mm的−3°工况，但使原本成功的Z−10mm/−3°和+3°失败。PPO相对于简单偏置没有总体通过数量优势，行为差异值得后续分析，不能宣称普遍鲁棒提升。

## 103. 高杆+10mm已知几何终点适配：完整周期通过

`outputs/highrod_endpoint_pose_probe_v1`在独立运动学数据上进行三个初值的受约束求解：钩爪目标不变，双脚目标升高10mm，机身平移范数限制0.03m，所有关节同时满足原生限位及相对名义终点±0.039rad。三个候选均满足目标且无非预期穿透。选择最大关节偏移最小的候选2（0.028595rad），机身参考变化约X+3.416mm、Z+8.319mm。

`outputs/highrod_endpoint_adapt_plan_v1`仅在原计划99.66至105.66秒姿态闭合阶段平滑加入终点偏移，之前参考完全不变。名义场景密集位置/碰撞检查通过；继承的IK误差不作为新混合轨迹的IK证明。使用真实已知+10mm几何离线生成，不是感知策略。

`outputs/highrod_endpoint_adapt_dynamic_v1/row01_z_+10mm`完成113.66秒，全部运行验收、独立轨迹审计、原生位置、杆身份及前后配对审计通过。摆脚/自碰撞/非预期支撑峰值均0N，最大机身误差5.652mm，2000Hz最小原生关节余量0.00159850rad。最后两秒最大关节闭合误差0.0347360rad（右腕俯仰），小于原0.04rad标准；此前为左踝约0.07333rad。力矩比峰值1.0，故不声称额外硬件力矩裕量。

`endpoint_pair_audit.json`确认原场景、控制器、时序、支撑调度和验收配置相同；99.66秒以前物理qpos逐元素一致。仅闭合参考变化使该工况从失败转为通过。尚未验证误差估计偏差、通用参考族或跨周期连续性，不将其自动加入既有17工况同参考通过数量。

复现：
```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
.venv\\Scripts\\python.exe tools/run_early_contact_reset_suite.py --out outputs/highrod_endpoint_reproduce --workers 1 --plan outputs/highrod_endpoint_adapt_plan_v1/plan.npz --contact-frame-from-transfer --reset-contact-frame-each-cycle --cases row01_z_+10mm
.venv\\Scripts\\python.exe tools/audit_joint_limits.py --run outputs/highrod_endpoint_reproduce/row01_z_+10mm
.venv\\Scripts\\python.exe tools/audit_cycle_contact_identity.py --run outputs/highrod_endpoint_reproduce/row01_z_+10mm
```

## 104. 后续优先级

当前实验全部结束，最近账户剩余31%，继续按20%阈值推进。优先检查终点补偿幅度变化是否仍通过，明确其裕量；再分析PPO新几何失败，不先追加无诊断训练。2米重复攀爬、局部学习、几何适配三类证据必须分别标注，真实感知与钩爪硬件承载仍待验证。
'''
report.write_text(s,encoding='utf-8')
c=json.loads((out/'checkpoint.json').read_text());c['running']=[];c['last_observed_remaining_percent']=31
c['latest_results'].update(untrained_geometry='45/45integrity; zero12/15,constant13/15,PPO13/15; PPOadds3loses2; fixedadds1loses0',highrod_endpoint_adaptation='PASS113.66s,closurepeak0.034736rad,nativeperstepmargin0.0015985rad; allaudits including exactprefixbefore99.66. Privilegedknown+10mm only.')
c['next']=['All jobscomplete:13050newgeometry,30178highrod. Results audited. Reportthrough104/sourcev13.','Next meaningfulexperiment: endpointcorrection gain0.75 and1.25 on same highrod+10mm scene vs1.0passed. Labelcorrectiongain sensitivity, NOTequivalent measuredheight error (nonlinearIK). Generate newplanfiles; preserve parentplans/runners andthresholds.','Aftergainrange, consider geometry-conditioned endpointplanning and validation fornominal/+5/+10 withseparate refs; never silentlymerge into same-reference17grid.','PPOuntrainedgeom13/15 equalsconstant13/15 butdifferentfailures. DiagnoseZ-10 +/-3 failures beforemoretraining.','Readquota periodically; remaining31lastseen, stopnewexperiments at<=20 andreport. No commits/push ornewagents.']
(out/'checkpoint.json').write_text(json.dumps(c,indent=2))
paths=set(json.loads((out/'source_manifest_v12.json').read_text()))
paths.update(['tools/probe_highrod_endpoint_pose.py','tools/create_highrod_endpoint_plan.py','tools/audit_highrod_endpoint_pair.py','tools/summarize_untrained_geometries.py','tools/checkpoint_highrod_success_v13.py'])
snap=out/'source_snapshot_v13';snap.mkdir(exist_ok=False);hashes={}
for rel in sorted(paths):
    target=snap/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target);hashes[rel]=hashlib.sha256(target.read_bytes()).hexdigest()
(out/'source_manifest_v13.json').write_text(json.dumps(hashes,indent=2));print('Report104/sourcev13saved')
