"""Consolidate existing evidence after the user-requested quota stop; no experiments."""
import hashlib
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'outputs/research_progress_20261017';report=ROOT/'docs/baselines/research_progress_20261017.md'
def read(rel):return json.loads((ROOT/rel).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    multi='outputs/multirung_force_history_two_cycles_v1/persistence__rows01_02_z_+5mm'
    m=read(multi+'/summary.json');ma=read(multi+'/independent_audit.json')
    extra={name:read(multi+'/'+name)['passed'] for name in ['joint_limit_audit.json','contact_identity_audit.json','early_schedule_audit.json']}
    multi_pass=bool(m['passed'] and ma['passed'] and all(extra.values()))
    sources={}
    folders=['outputs/early_contact_eight_cycle_v1/nominal','outputs/early_contact_reset_eight_cycle_v1/nominal','outputs/highrod_endpoint_adapt_dynamic_v1/row01_z_+10mm','outputs/periodic_force_history_two_cycles_v1/persistence__nominal',multi]
    for folder in folders:
        for name in ['summary.json','independent_audit.json','joint_limit_audit.json','contact_identity_audit.json']:
            p=ROOT/folder/name;sources[str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
        for name in ['early_schedule_audit.json','endpoint_pair_audit.json']:
            p=ROOT/folder/name
            if p.exists():sources[str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
    for rel in ['outputs/early_contact_reset_grid_v1/results.json','outputs/biased_load_transfer_ppo_32768_v1/continuation_audit.json','outputs/residual_methods_comparison_v2/summary.json','outputs/untrained_geometry_memory_v1/paired_summary.json','outputs/known_force_history_v1/study_audit.json','outputs/untrained_force_history_v1/study_audit.json','outputs/window_force_history_fullcycles_v1/paired_audit.json','outputs/visibility_dropout_study_v1/study_audit.json','outputs/highrod_gain_sensitivity_v1/paired_summary.json','outputs/highrod_endpoint_minimax_dynamic_v1/row01_z_+10mm/endpoint_pair_audit.json']:
        sources[rel]=sha(ROOT/rel)
    assert read('outputs/early_contact_eight_cycle_v1/nominal/summary.json')['passed']
    assert read('outputs/biased_load_transfer_ppo_32768_v1/continuation_audit.json')['passed']
    assert read('outputs/known_force_history_v1/study_audit.json')['successes']==15
    assert read('outputs/untrained_force_history_v1/study_audit.json')['successes']==15
    assert read('outputs/window_force_history_fullcycles_v1/paired_audit.json')['successes']==3
    assert read('outputs/visibility_dropout_study_v1/study_audit.json')['expected_bounded_behavior']
    assert sum(r['passed'] for r in read('outputs/early_contact_reset_grid_v1/results.json'))==10
    failures=[k for k,v in m['checks'].items() if not v]
    multi_text=('通过全部运行及附加审计' if multi_pass else '未通过完整验收；失败项：'+', '.join(failures))
    final=f'''
## 本轮最终汇总：达到剩余额度20%后的收尾报告

已从账户工具确认已用80%、剩余20%，随后停止新增实验，只收齐已启动结果并整理证据。本轮研究执行与报告交付结束，不代表整个研究或论文已完成。

### 主要成果

1. **真实动力学完成约两米连续攀爬。** 提前启用接触控制的方案完成8周期、885.76秒、爬升1.998482米；每个物理步的原生位置约束、运行验收、独立审计、全部周期端点和杆身份均通过。显式每周期增益重置版本也通过。旧版本虽然完成两米，但新增关节位置审计失败，已明确保留为失败，不能混用旧录像宣称全部通过。
2. **同参考17工况通过数由3提升到10。** `early_contact_reset_grid_v1`为10/17严格通过、15/17走完整段；其余失败未放宽阈值。成功的高杆专用参考另列，未混入同参考分母。
3. **高杆+10mm从闭合失败变为完整通过。** 先消除摆脚碰撞，再在99.66至105.66秒适配终点姿态；控制器、场景和此前物理轨迹相同。113.66秒完整通过，末端最大关节误差从0.07333降至0.03474rad，原阈值仍0.04rad。补偿幅度0.75时进一步降至0.02635rad；幅度1.25时0.04179rad，仍判失败。仅四个离散幅度，不能宣称连续容差范围。最小原生位置余量仅约0.00160rad，力矩比达到1.0，尚无硬件额外裕量保证。
4. **训练已实际执行，但PPO尚未胜过经典基线。** 局部残差PPO累计32,768步，模型、优化器续训来源和45份评估记录审计通过；原15工况仍14通过，恢复1项但新增1项退步。局部策略已接入3个完整周期，但只有49至61.56秒使用学习残差，不能称为学会完整两米攀爬。
5. **接触历史方案减少一项几何真值依赖。** 不查询当前钩杆间隙，仅保留此前承重确认和20ms接触历史，原15项与另15项几何测试全部通过；新增15项轨迹与带几何间隙门限的记忆版本完全相同。3个完整周期也通过，并与原记忆方案qpos/qvel完全相同。
6. **跨周期验证。** 名义场景在每周期局部窗口均施加−3°法向偏差，连续221.44秒、两周期通过，最小原生关节余量0.0187818rad。两根脚杆同时升高5mm的追加实验运行至{m['simulation_time_s']:.2f}秒，{multi_text}。原接口不支持rung_02导致的初始化失败已保留；使用独立多杆接口后重跑，未改旧训练/场景接口。

卸载保护另有独立原型：要求其余支撑持续250ms承重≥10N且速度≤0.02m/s；准确模型短时预测曾使Z+5mm工况保持0.5秒，但X−10mm仍失败。不能宣称通用撤退或保护保证，也不把停止诊断等同于安全保持。各完整周期是否启用联锁，以对应manifest配置为准。

### 公平对照与失败结论

| 方法 | 原三种几何×五角度 | 另五种几何×三角度 | 说明 |
|---|---:|---:|---|
| 零残差 | 14/15 | 12/15 | 相同低层约束 |
| 固定小偏置 | 15/15 | 13/15 | 右髋俯仰稳态−0.00375rad；依据已知失败选择 |
| PPO 32,768步 | 14/15 | 13/15 | 一个种子；有新增成功，也有退步 |
| 几何门限接触记忆 | 15/15 | 15/15 | 额外精确间隙门限 |
| 无当前间隙查询的接触历史 | 15/15 | 15/15 | 去掉当前间隙门限，仍依赖理想接触/力信息 |

这是有限确定性工况计数，不是统计成功率或盲测。另五种几何未参与这次PPO训练，但曾用于经典控制研究。原生力矩、位置、摩擦和验收阈值保持不变。

PPO低杆退步已有精确重放证据：−3°下左钩约0.94微米间隙导致实测接触消失；+3°下原约束线性规划确实不可行，离线移除腕部力矩边界才可行，未把放宽约束用于仿真。另一项训练内退步来自支撑确认历史中一个速度略超0.02m/s的样本，仍保留失败。

检测中断实验共8项：10/20ms接触定位中断可由接触历史恢复；30ms在记忆过期时终止诊断。物理接触和理想力反馈仍然存在，因此既不是实际脱钩恢复，也不是实机安全停机验证。

### 仍需解决

- 真实感知、噪声与延迟；当前仍有准确接触点/拓扑及理想力反馈。已去掉间隙查询不等于去掉所有真值依赖。
- 真正失去物理支撑后的减载、保持或撤退；不能把停止仿真当成安全控制。
- PPO奖励/约束设计、多种子训练与独立评估；当前没有普遍优于简单方法的证据。
- 将专用终点适配推广为有明确定义的几何条件规划，验证连续多杆、多周期和估计误差；不依靠单点调参宣称鲁棒性。
- 钩爪硬件强度、摩擦、关节力矩速度和热约束；没有真机连接或硬件验证。论文贡献尚未确认。

### 复现与证据入口

工作目录`D:\\G1Climbing\\external\\RoboMimic_Deploy`，使用本仓库`.venv`，输出目录必须未存在。以下命令是复现说明，本轮额度停止后未自动执行：

```powershell
Set-Location D:\\G1Climbing\\external\\RoboMimic_Deploy
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
# 无当前间隙查询：三个完整周期工况
.venv\\Scripts\\python.exe tools/run_window_force_history_experiment.py --out outputs/reproduce_force_history --workers 2
# 高杆专用终点适配：已审计幅度1.0参考
.venv\\Scripts\\python.exe tools/run_early_contact_reset_suite.py --out outputs/reproduce_highrod --workers 1 --plan outputs/highrod_endpoint_adapt_plan_v1/plan.npz --contact-frame-from-transfer --reset-contact-frame-each-cycle --cases row01_z_+10mm
# 双周期多杆场景
.venv\\Scripts\\python.exe tools/run_multirung_force_history_experiment.py --out outputs/reproduce_multirung --workers 1
```

关键文件：

- 两米录像：`outputs/early_contact_eight_cycle_v1/nominal/cycle_recorded_8x.mp4`。
- 高杆录像：`outputs/highrod_endpoint_adapt_dynamic_v1/row01_z_+10mm/cycle_recorded_4x.mp4`，已完整解码检查。
- 高杆敏感性图：`outputs/research_progress_20261017/highrod_gain_sensitivity_v1.png/.pdf`。
- 本轮最终证据哈希：`outputs/research_progress_20261017/final_evidence.json`。
- 控制源码、实验参数、轨迹、逐场景失败和审计保留在各输出目录；最终源码快照为`source_snapshot_v16`。

历史PPO完整周期摘要中的`trained_policy=False`是继承的元数据错误：原摘要与冻结源码保留，三个目录均附`metadata_erratum.json`。正确含义是已训练局部残差、非整周期学习；新入口和录像标签已纠正。

'''
    s=report.read_text(encoding='utf-8');assert '## 本轮最终汇总：' not in s
    s=s.replace('最近一次查询剩余 24%；工作仍在进行。','已查询剩余20%；停止新增实验，已启动实验收尾完成。')
    s=s.replace('## 当前结论速览（以已审计结果为准）',final+'## 阶段性结论速览（以下保留研究过程记录）',1)
    s+=f'\n\n## 115. 收尾记录\n\n账户80%已用触发停止新增实验。名义双周期通过；多杆双周期：{multi_text}。原初始化拒绝和新的独立多杆支持均保留，几何检查确认单杆结果兼容、无效多项请求不部分修改模型、qpos/qvel和关节索引保持、静态碰撞BVH重新编译且拓扑一致。没有提交、推送或真机操作。\n'
    report.write_text(s,encoding='utf-8')
    evidence=dict(quota_observed_used_percent=80,quota_remaining_percent_at_stop=20,new_experiments_stopped=True,in_flight_collected=True,multirung_passed=multi_pass,multirung_checks=m['checks'],sources=sources,report_sha256=sha(report),scope='Completion of requested quota-bounded research round and report, not completion of the entire research project.')
    (out/'final_evidence.json').write_text(json.dumps(evidence,indent=2))
    c=read('outputs/research_progress_20261017/checkpoint.json');c['status']='requested_quota_stop_report_complete';c['last_observed_remaining_percent']=20;c['running']=[];c['next']=['No automatic newexperiments: userquota stop reached. Finalreport andevidence complete. Resume research onlyonnew userinstruction.','Nextresearch priorities listedinreport; do not claimfulllearnedclimbing,realperception,hardwarevalidation ornovelty.']
    c['latest_results'].update(nominal_periodic_forcehistory='Two uninterrupted cycles221.44s PASS all runtime/independent/native/identity/schedule;eachcyclebiaswindow',multirung_periodic_forcehistory=dict(passed=multi_pass,time_s=m['simulation_time_s'],checks=m['checks']))
    (out/'checkpoint.json').write_text(json.dumps(c,indent=2))
    paths=set(read('outputs/research_progress_20261017/source_manifest_v15.json'))
    paths.update(['training/repeated_rung_errors.py','deploy_mujoco/run_multirung_force_history_cycle.py','tools/audit_multirung_cycle.py','tools/run_multirung_force_history_experiment.py','tools/check_repeated_rung_errors.py','tools/finalize_research_round_20261017.py'])
    snap=out/'source_snapshot_v16';snap.mkdir(exist_ok=False);hashes={}
    for rel in sorted(paths):
        dest=snap/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest);hashes[rel]=sha(dest)
    (out/'source_manifest_v16.json').write_text(json.dumps(hashes,indent=2));print(json.dumps(dict(report=str(report),multirung_passed=multi_pass,evidence_sources=len(sources)),indent=2))

if __name__=='__main__':main()
