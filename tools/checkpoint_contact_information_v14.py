"""Archive sensitivity, diagnosis and stricter classical baseline evidence."""
import hashlib
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'outputs/research_progress_20261017';report=ROOT/'docs/baselines/research_progress_20261017.md'
s=report.read_text(encoding='utf-8');assert '## 105.' not in s
s=s.replace('最近一次查询剩余 31%；','最近一次查询剩余 27%；')
s=s.replace('零残差12/15，固定偏置13/15，PPO13/15，全部独立审计通过。','零残差12/15，固定偏置13/15，PPO13/15，纯接触记忆15/15，全部独立审计通过；记忆仍有精确间隙信息假设。')
s+='''

## 105. 高杆终点补偿幅度敏感性完成

`outputs/highrod_gain_sensitivity_v1/paired_summary.json`汇总四个离散幅度：0、0.75、1、1.25。最大末端关节误差依次0.0733314、0.0263496、0.0347360、0.0417886rad；只有0.75和1通过原0.04rad阈值。1.25仍完成全程，但右腕俯仰及左膝超出闭合误差边界，保留失败。四个工况99.66秒前的物理轨迹逐元素一致，原生关节限位均通过。

补偿幅度不是等价杆高估计误差，四个点也不能证明连续区间鲁棒性。图`outputs/research_progress_20261017/highrod_gain_sensitivity_v1.png/.pdf`已视觉检查。成功高杆双视角录像`outputs/highrod_endpoint_adapt_dynamic_v1/row01_z_+10mm/cycle_recorded_4x.mp4`已完成全文件解码检查，预览已查看。

## 106. PPO低杆退步：接触丢失与原生力矩约束分别成立

`outputs/untrained_policy_failure_diagnostic_v1`精确重放Z−10mm两种失败，qpos/qvel相对原记录差异均0。−3°在53.740秒丢失左钩实测接触，最小钩杆间隙约0.936微米；+3°在57.735秒分配失败，各支撑仍有接触。

+3°失败状态的独立线性规划验证原约束确实不可行；仅离线移除腕俯仰/偏航边界后可行，约束残差约6.1e−14。没有把放宽边界应用到仿真，也不将其称为恢复。由此不能将该失败归为求解器误报或直接通过重试修复。

## 107. 最小最大关节偏移候选通过，但不优于已测0.75幅度

`outputs/highrod_endpoint_minimax_v1`在保持双脚与钩爪目标姿态相同的条件下最小化最大关节偏移，得到0.026333rad（原离线候选0.028595rad），目标姿态残差约3.3e−16，无非预期穿透。局部优化结果不代表全局最优。

`outputs/highrod_endpoint_minimax_dynamic_v1/row01_z_+10mm`完整113.66秒通过运行、独立、原生位置、杆身份及配对审计。实际最终两秒最大误差0.0329279rad，优于幅度1的0.0347360，但劣于幅度0.75的0.0263496。离线姿态目标较好不自动意味着实际动力学误差最小；不替换已测最优结果或改变阈值。

## 108. 纯接触记忆新增几何15/15通过

`outputs/untrained_geometry_memory_v1/paired_summary.json`五种新几何×0/±3°全部15通过，轨迹、源/前缀配对与缓存年龄/间隙/支撑调度审计通过。对照零残差12、固定偏置13、PPO13，记忆新增3项成功且无退步。但记忆利用精确几何间隙≤0.1mm，信息条件不同；过去≥10N载荷门限由冻结控制器执行，未独立重建200Hz力历史。

## 109. 去除当前几何间隙查询的对照已启动

独立`training/force_history_contact_controller.py`继承同一力分配，仅覆盖缓存点恢复：过去承重确认、20ms时限和计划支撑条件保留，不查询当前钩杆间隙；不额外施加力。针对禁止间隙查询、缓存过期、负年龄、缺少历史的检查已通过。精确接触点/拓扑和理想力反馈仍与原控制相同，不能称为真实传感部署。

`outputs/untrained_force_history_v1`（会话51502）测试五种新几何的15工况；`outputs/known_force_history_v1`（会话3230）测试原三种几何×五种角度的15工况。两组尚未完成，不能预报通过。完成后运行`audit_force_history_study.py`；新几何还应与几何记忆比较完整保存轨迹和缓存使用时刻。

当前最后查询剩余27%，继续20%停止条件。原训练模型与实验源文件保持兼容，未提交或推送。
'''
report.write_text(s,encoding='utf-8')
c=json.loads((out/'checkpoint.json').read_text());c['last_observed_remaining_percent']=27;c['running']=[dict(task='Forcehistory newgeometry15',session_id=51502,out='outputs/untrained_force_history_v1'),dict(task='Forcehistory knowngeometry15',session_id=3230,out='outputs/known_force_history_v1')]
c['latest_results'].update(highrod_gain='0,.75,1,1.25 closurepeaks .073331,.026350,.034736,.041789;fail,pass,pass,fail;pairedprefixidentical',highrod_minimax='PASS113.66s all audits;actualclosurepeak.0329279,not betterthan gain.75',untrained_memory='15/15all auditsaudit;extra exactgap assumption',ppo_lowrod_diagnosis='Exactreplay: -3 missingLHgap.936um; +3trueLPinfeasible, removingwristbounds feasible offlineonly',video_highrod='4xvideo rendered,previewviewed,full decodePASS')
c['next']=['Collect51502 and3230; do notedit imported force_history/contact_memory/env modules while running.','Then run audit_force_history_study.py --run outputs/untrained_force_history_v1 --compare-memory outputs/untrained_geometry_memory_v1; and --run outputs/known_force_history_v1. Allgroupaudits mustpass.','If no-gap persistence matchesmemory, reducesoneprivilegedinformation requirement only; stillidealcontact/forces. Need longer/false-contact-loss or fullcycles beforegeneralclaims.','Highrodgain75% besttestedclosure. Minimaxnotbetter; retainallfailures. Need geometric-conditioned generalization, not arbitraryextra tuning.','Reportthrough109/sourcev14. Remaining27lastseen; stopnewexperiments at<=20 andreport. No push/commit/newagents.']
(out/'checkpoint.json').write_text(json.dumps(c,indent=2))
paths=set(json.loads((out/'source_manifest_v13.json').read_text()))
paths.update(['tools/run_highrod_gain_sensitivity.py','tools/diagnose_untrained_policy_failures.py','tools/evaluate_untrained_memory.py','tools/compare_untrained_memory.py','tools/refine_endpoint_minimax.py','tools/create_minimax_endpoint_plan.py','tools/audit_minimax_endpoint_pair.py','tools/audit_highrod_gain_pairs.py','tools/plot_highrod_gain_sensitivity.py','training/force_history_contact_controller.py','tools/check_force_history_contract.py','tools/evaluate_force_history.py','tools/evaluate_known_force_history.py','tools/audit_force_history_study.py','tools/checkpoint_contact_information_v14.py'])
snap=out/'source_snapshot_v14';snap.mkdir(exist_ok=False);hashes={}
for rel in sorted(paths):
    target=snap/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target);hashes[rel]=hashlib.sha256(target.read_bytes()).hexdigest()
(out/'source_manifest_v14.json').write_text(json.dumps(hashes,indent=2));print('Report109/sourcev14saved')
