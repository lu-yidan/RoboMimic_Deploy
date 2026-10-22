"""Archive the contact-training interface and its measured prefix validation."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
out = root / 'outputs/research_progress_20261017'
report = root / 'docs/baselines/research_progress_20261017.md'
s = report.read_text(encoding='utf-8').replace('最近一次查询剩余 53%', '最近一次查询剩余 51%')
extra = '''
63. 新增独立 `contact_foot_residual_env.py`、`contact_load_transfer_residual_env.py` 及训练入口 `train_contact_load_transfer.py`，保留原 76 维观测、6 维右腿动作及 0.015 rad 残差上限，底层改为准确接触几何控制。每个 0.5 ms 物理步检查原生关节位置，违规立即终止。原训练环境和 checkpoint 契约未修改。底层使用真实接触点、法向和拓扑，不能因策略观测没有杆坐标就称为无特权感知。
64. `contact_load_transfer_prefix_smoke_v1` 从新连续轨迹 49 s 状态恢复，正常与脚杆 X=-10 mm 两条零残差片段均完成到 61.56 s，满足其余三支撑确认及左脚卸载。每物理步位置检查通过，最小余量约 0.01931 rad；独立来源、几何、力矩、观测及保存位置审计 2/2 通过。尚未用该环境训练新 PPO，不宣称学习性能改善。
65. 前缀连续性量化：两个初始 qpos/qvel 与完整轨迹逐项相同，各 629 个保存状态比较，qpos 最大差分别 6.01e-11、6.59e-11，qvel 最大差 2.28e-10、6.09e-10。保存于 `continuity_comparison.json`。这只是已测两例、零动作的对照；不能推广为任意扰动下复位与连续动力学严格相同。
66. 新增训练环境测试验证 76/6 契约、零动作推进以及物理步内关节越界终止。全部 56 项回归通过，日志 `research_tests_20261017_v12.log`。网格仍在运行：截至本次记录，正常与四个脚杆 X 偏差均通过；Z=-10 mm 完成全时长、位置及碰撞检查通过，但终点关节误差 0.078812 rad 超过原 0.04 rad 标准，保留失败。不能以部分案例计算最终整体通过率。

'''
if '63. 新增独立' not in s:
    s = s.replace('## 在跑与下一步', extra + '## 在跑与下一步')
report.write_text(s, encoding='utf-8')
p = out / 'checkpoint.json'
d = json.loads(p.read_text())
d['last_observed_remaining_percent'] = 51
d['latest_results'].update(tests=56, contact_training_interface='Prepared independent76obs/6act environment with exact-contact low-level controller and2000Hz native position stop; not yet trained', contact_training_prefix='nominal andX-10 zero residual both pass; independent2/2; initial states exact, maxqpos difference6.59e-11 vs uninterrupted records', early_grid_partial='first6 returned: nominal andfourfootX pass, footZ-10 completes but closure0.078812rad fails. Grid still running')
d['next'].insert(3, 'When all nine foot-prefix cases are available from early_contact_reset_grid_v1, evaluate zero residual using tools/check_contact_load_transfer.py. Audit with audit_contact_load_transfer.py and compare_contact_prefix_continuity.py. Only then choose meaningful PPO experiment; do not claim new training already occurred.')
p.write_text(json.dumps(d, indent=2), encoding='utf-8')
paths = set(json.loads((out / 'source_manifest_v4.json').read_text()))
paths.update(['training/contact_foot_residual_env.py', 'training/contact_load_transfer_residual_env.py', 'tools/train_contact_load_transfer.py', 'tools/check_contact_load_transfer.py', 'tools/audit_contact_load_transfer.py', 'tools/compare_contact_prefix_continuity.py', 'tests/test_contact_load_transfer_env.py', 'tools/checkpoint_contact_training_v5.py'])
snapshot = out / 'source_snapshot_v5'
snapshot.mkdir(exist_ok=False)
hashes = {}
for rel in sorted(paths):
    target = snapshot / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / rel, target)
    hashes[rel] = hashlib.sha256(target.read_bytes()).hexdigest()
(out / 'source_manifest_v5.json').write_text(json.dumps(hashes, indent=2))
print(f'Frozen {len(hashes)} files')
