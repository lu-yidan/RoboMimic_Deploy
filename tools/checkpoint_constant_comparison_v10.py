"""Record the stronger simple baseline and preserve the ongoing continuation."""
import hashlib
import json
from pathlib import Path
import shutil
root=Path(__file__).resolve().parents[1];out=root/'outputs/research_progress_20261017';report=root/'docs/baselines/research_progress_20261017.md'
s=report.read_text(encoding='utf-8')
extra='''
90. `constant_residual_selection_v1.json` 记录固定选择规则：在已审计成功动作中按最小平方范数、最低关节序号、动作字典序选择，得到右髋俯仰动作 -0.25，经原 0.015 rad 缩放及滤波，对应稳态偏置 -0.00375 rad。选择仅使用已知 Z+5/-3 度难例，不是学习所得。
91. `constant_residual_comparison_v1` 在相同三种杆位、五种法向偏差的 15 场景全部成功，15 条独立审计通过。`residual_methods_comparison_v1` 核对同一前缀及核心环境源码后：零动作 14/15、PPO8192 14/15、固定小偏置 15/15、纯记忆 15/15、区间锥加记忆 11/15。固定偏置与 PPO 使用相同控制器信息；记忆额外使用准确几何间隙。当前证据不足以宣称学习提升，也不足以把这些片段恢复视为接触记忆独有优势。
92. `memory_window_recovery_v1.png/.pdf` 已生成并查看，展示限定偏差窗口完整周期的实测左钩/左脚竖向力及四次缓存事件。力曲线为 50 Hz 保存值，不能用它证明每个物理步的接触存在性。原控制停止于 53.7 s，记忆控制继续完成 110.72 s；实际物理力未被人工补上。
93. 高净空候选 Z+10 mm 的闭合误差主要来自左踝俯仰约 -0.07333 rad、右踝俯仰约 -0.04867 rad，其余关节峰值低于 0.04 rad。末端身体相对参考位置偏差约 [-3.787,1.982,3.498] mm。`endpoint_joint_diagnostic.json` 只是定位，不是修复或可行性证明。
94. 累计 32768 步续训仍在执行，不能将目标步数当已完成步数。已准备 `audit_continued_training.py`，完成后核对父 checkpoint、30 条复用轨迹及其来源哈希、步数记账、训练分布、评估角度，以及新的 45 条轨迹/策略动作审计。60 项回归测试通过，日志 `research_tests_20261017_v15.log`。

'''
if '90. `constant_residual_selection' not in s:s=s.replace('## 在跑与下一步',extra+'## 在跑与下一步')
report.write_text(s,encoding='utf-8')
p=out/'checkpoint.json';d=json.loads(p.read_text());d['running']=[dict(task='PPO continuation to32768',session_id=99626,out='outputs/biased_load_transfer_ppo_32768_v1')]
d['latest_results'].update(tests=60,constant_comparison='15/15 and15audits; selectedright hip pitch action-0.25 =>-0.00375rad; sameinformationasPPO; matchedprefix/core comparison complete',memory_recovery_figure='png/pdf generated and visuallychecked',highrod_endpoint='Main errors leftanklepitch-0.07333 andrightanklepitch-0.04867; no fixyet')
d['next']=['Inspect99626 progress/afterevaluations; target32768 notyetcomplete. All otherjobscomplete.', 'After summary run audit_continued_training.py --run outputs/biased_load_transfer_ppo_32768_v1; then compare_residual_methods.py --ppo outputs/biased_load_transfer_ppo_8192_v1 outputs/biased_load_transfer_ppo_32768_v1 --out outputs/residual_methods_comparison_v2.', 'Possible next meaningful experiment: same selectedconstant action in full continuous cycle, onlyloadwindow and filteredrampout, comparewith raw2/3 andmemory3/3. Notyetimplemented. Avoidclaimmemoryuniqueeffectbecausefixedoffsetalready15/15segments.', 'Fixedaction selectionrule andsourceSHA in outputs/constant_residual_selection_v1.json; selectedjoint0_-0.25. Nolearningclaim.', 'Highrodclosurediagnostic indicatesanklepitchdominates, but no endpointadaptation demonstrated. Keep thresholds unchanged.', 'Report through94/sourcev10. Do not rerunhistoricalupdater scripts. Originalpolicycontractsunchanged.', 'Lastquota38%; checkperiodically, stop newexperiments at<=20 and report.']
p.write_text(json.dumps(d,indent=2),encoding='utf-8')
paths=set(json.loads((out/'source_manifest_v9.json').read_text()))
paths.update(['tools/create_constant_residual_comparison.py','tools/run_constant_residual_comparison.py','tools/audit_continued_training.py','tools/plot_window_memory_recovery.py','tools/diagnose_endpoint_joint_errors.py','tools/compare_residual_methods.py','tools/checkpoint_constant_comparison_v10.py'])
snapshot=out/'source_snapshot_v10';snapshot.mkdir(exist_ok=False);hashes={}
for rel in sorted(paths):
    target=snapshot/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/rel,target);hashes[rel]=hashlib.sha256(target.read_bytes()).hexdigest()
(out/'source_manifest_v10.json').write_text(json.dumps(hashes,indent=2));print('Simplebaseline and ongoingtraining evidence saved')
