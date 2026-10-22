"""Record the first extended-acceptance two-metre result and prefix exclusions."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
out = root / 'outputs/research_progress_20261017'
report = root / 'docs/baselines/research_progress_20261017.md'
s = report.read_text(encoding='utf-8').replace('最近一次查询剩余 51%', '最近一次查询剩余 48%')
s = s.replace('两米扩展尚在运行，不能提前宣称长程成功。', '最新两米扩展 `early_contact_eight_cycle_v1` 已完成：885.76 s、八周期、爬升 1.998482 m，运行时逐物理步位置、独立审计、八个端点、切换时序及保存接触杆位归属全部通过。显式增益重置版本仍独立验证中。')
extra = '''
67. 新增 `audit_cycle_contact_identity.py` 检查全过程已保存的承载接触归属，而非只看终点。提前接触的两种两周期版本分别检查 89545、89568 条大于 1 N 的接触记录，均无错杆。范围为 50 Hz 保存日志，不声称采样间归属或传感器可靠性。
68. `early_contact_eight_cycle_v1` 完成八周期、885.75999998 s，爬升 1.998482233 m。13 项运行检查（含 2000 Hz 原生位置）和 12 项独立检查全部通过，八个端点通过。最小逐物理步关节余量 0.018755871 rad，最大身体跟踪误差 5.047 mm，无非预期、摆脚或自碰撞。附加切换审计记录 32 次模式切换与 8 次门控；355686 条已保存承载接触的杆位检查全部通过。这是电机控制、准确接触几何条件下的连续动力学成功，尚非学习或真实感知验证。
69. 上述两米完整八倍速双视角录像 `early_contact_eight_cycle_v1/nominal/cycle_recorded_8x.mp4` 已生成，预览已查看，验收标记与审计一致；FFmpeg 完整解码 1109 帧、110.90 s 成功，日志 `early_contact_eight_video_decode.log`。旧两米失败录像保留，不能混用。
70. 九个候选 49 s 训练前缀审计完成，八个合格，脚杆 Z=+10 mm 因此前摆脚碰撞被排除，结果在 `contact_training_prefix_audit_v1.json`。新训练环境现在检查起点前累计碰撞、跟踪及保存位置，并明确拒绝不合格前缀；新增拒绝测试通过。不能将排除后的成功率宣称为九工况成功率。训练入口允许显式列出评估集合，完整记录训练与评估工况。
71. `contact_load_transfer_eight_prefix_v1` 的八个合格前缀均在零残差下完成 49..61.56 s 载荷迁移；第九个是失败前缀，而非已解决。该任务的成功率指标已饱和，未盲目增加 PPO 步数。启动 `early_contact_bias_segments_v1`：正常、脚杆 Z+5、脚杆 X-10 三组新前缀，各测 0/-3/+3 度法向偏差，用于确认有实际失败的后续学习任务。准确接触存在性、身份及力反馈仍保留。
72. 新网格中 Z=-5 mm 运行检查通过但独立末端误差 10.762 mm，超过 10 mm 标准，整体失败；Z=-10 mm 终点关节闭合仍失败；Z=+10 mm 摆脚碰撞峰值约 40.89 N。保持原阈值，未将完成时长或运行检查单独当作整体成功。

'''
if '67. 新增' not in s:
    s = s.replace('## 在跑与下一步', extra + '## 在跑与下一步')
s = s.replace('- `early_contact_eight_cycle_v1`：提前接触控制连续八周期，逐物理步检查原生位置范围；完成后复核八个端点、独立状态及完整录像。', '- `early_contact_eight_cycle_v1` 已通过扩展验收并生成完整录像；等待显式增益重置八周期的独立结果。')
report.write_text(s, encoding='utf-8')
p = out / 'checkpoint.json'
d = json.loads(p.read_text())
d['last_observed_remaining_percent'] = 48
d['running'] = [dict(task='Reset-gain eight cycles', session_id=56574, out='outputs/early_contact_reset_eight_cycle_v1'), dict(task='Reset-gain17case grid', session_id=99473, out='outputs/early_contact_reset_grid_v1'), dict(task='New-prefix nine normal-bias cases', session_id=50378, out='outputs/early_contact_bias_segments_v1'), dict(task='Eight-prefix audits and full regression', session_id=7101, out='outputs/research_tests_20261017_v13.log')]
d['latest_results'].update(early_eight='PASS885.76s/8cycles/rise1.998482233m; native per-step margin0.018755871rad; endpoint/schedule/contactidentity audits PASS; video decoded1109frames', training_prefixes='8eligible,footZ+10 excluded for pre49s swing collision; all8zero-residual segments successful; no new PPO training', latest_tests='New prefix rejection test passes; full57-test regression running; inspectv13')
d['next'] = ['Read current handles and logs, do not restart live jobs.', 'When reset eight completes audit_joint_limits.py, audit_early_contact_schedule.py, audit_cycle_contact_identity.py and all8endpoints; render accurate video.', 'When17grid completes audit_suite_native_positions.py then compare_early_contact_grid.py --new outputs/early_contact_reset_grid_v1 --out outputs/early_contact_grid_comparison_v1. Inspect failures and regressions; same core/reference/geometry hashes required.', 'Read eight-prefix independent_audit.json and continuity_comparison.json; fulltests v13 pending. Zero8/8 is conditional on eight valid prefixes; report excludedZ+10.', 'Read new normal-bias nine cases50378; choose meaningful residual learning task only if baseline has failures. No new PPO trained yet.', 'Do not edit modules imported by queued bias/grid workers. Source snapshots preserve executed versions.', 'Keep original policy contracts untouched. Do not rerun historical checkpoint updater scripts.', 'Check quota; remaining<=20 stop new experiments and produce fullreport.']
p.write_text(json.dumps(d, indent=2), encoding='utf-8')
paths = set(json.loads((out / 'source_manifest_v5.json').read_text()))
paths.update(['tools/audit_training_prefixes.py', 'tools/audit_cycle_contact_identity.py', 'tools/compare_early_contact_grid.py', 'tools/run_early_contact_bias_suite.py', 'tools/checkpoint_two_meter_v6.py'])
snapshot = out / 'source_snapshot_v6'; snapshot.mkdir(exist_ok=False)
hashes = {}
for rel in sorted(paths):
    target = snapshot / rel; target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / rel, target)
    hashes[rel] = hashlib.sha256(target.read_bytes()).hexdigest()
(out / 'source_manifest_v6.json').write_text(json.dumps(hashes, indent=2))
print(f'Frozen {len(hashes)} files')
