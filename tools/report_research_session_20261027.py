"""Build a dated progress report from saved evidence, preserving pending results."""
from datetime import datetime
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'outputs'
    dest = out / 'research_progress_20261027_v1'
    dest.mkdir(exist_ok=True)
    evidence = {}

    def read(path):
        if not path.exists():
            return None
        data = path.read_bytes()
        evidence[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    average = out / 'calibrated_landing_ppo32768_20261027_v1'
    eight = out / 'calibrated_landing_ppo32768_eight_handoff_20261027_v1'
    geometry = out / 'calibrated_geometry_training_baseline_20261027_v1'
    comparison = read(out / 'calibrated_reward_comparison_20261027_v1/comparison.json')
    summary = read(eight / 'summary.json')
    physical = read(eight / 'independent_audit.json')
    action = read(eight / 'policy_transfer_audit.json')
    render = read(eight / 'render_manifest.json')
    baseline = read(geometry / 'baseline_result.json')
    diagnosis = read(geometry / 'failure_endpoint_diagnosis.json')
    gate_dir = out / 'calibrated_geometry_training_gate_20261027_v1'
    gate = read(gate_dir / 'summary.json')
    gate_started = (gate_dir / 'protocol.json').exists()
    training_dir = out / 'calibrated_geometry_ppo131072_20261027_v1'
    training_started = (training_dir / 'manifest.json').exists()
    training_status = read(training_dir / 'training_status.json')
    rows = []
    # Each worker saves a one-case result before the parent combines all results.
    for path in sorted((geometry / 'parallel_after').glob('training_*/result.json')):
        data = read(path)
        rows.append(data)
    if baseline:
        rows = [r for r in read(geometry / 'summary.json')['after'] if r['case'].startswith('training_')]
    failed = [{k: r.get(k) for k in ('case', 'reason', 'time_s')} for r in rows if not r['success']]
    state = dict(updated_at=datetime.now().astimezone().isoformat(),
        user_budget_instruction='Continue until quota is exhausted; supersedes earlier 20% stop.',
        average_candidate=str(average),
        eight_cycle=dict(completed=summary is not None, physical_audit_passed=bool(physical and physical['passed']),
            policy_audit_passed=bool(action and action['passed']),
            final_rise_m=summary.get('final_rise_m') if summary else None,
            duration_s=summary.get('time_s') if summary else None,
            video_decode_passed=bool(render and render['decode_passed'])),
        matched_reward_comparison=({k: {'successes': v['successes'], 'minimum_scaled_margin': v['minimum_scaled_margin']}
            for k, v in comparison['arms'].items()} if comparison else None),
        geometry_baseline=dict(completed=bool(baseline and baseline['completed']),
            new_cases_completed=len(rows), new_cases_passed=sum(r['success'] for r in rows),
            failures=failed, result=baseline),
        geometry_validation_evaluated=False, geometry_131072_step_training_started=training_started,
        geometry_training=dict(path=str(training_dir), status=training_status,
            log=str(out / 'calibrated_geometry_ppo131072_20261027_v1.log')),
        geometry_training_gate=dict(started=gate_started, completed=gate is not None,
            passed=bool(gate and gate['passed']), path=str(gate_dir)),
        no_automatic_default_promotion=True,
        evidence_sha256=evidence)
    lines = ['G1 攀爬研究进度（2026-10-27）', '', '一、已完成',
        '平均奖励 PPO 续训 32,768 步，已知工况 36/36 通过；局部邻域仍为 3/8。',
        '这里的 PPO 是固定参考动作、校准和控制器上的残差策略，不能表述为完全自主攀爬。']
    if comparison:
        for k, v in state['matched_reward_comparison'].items():
            lines.append(f"{k}：共同开发工况 {v['successes']}/16；最小归一化收尾裕量 {v['minimum_scaled_margin']:.6f}。")
        lines += ['相同初始参数、优化器、预算和工况的单种子对照。瓶颈奖励未改善最小裕量，不替换平均奖励模型。',
            '瓶颈奖励尚未完成其他已知工况及局部邻域回归；不能套用平均奖励模型的 36/36。']
    if summary:
        lines.append(f"新平均奖励 PPO 八周期：{summary['time_s']:.2f} 秒，爬升 {summary['final_rise_m']:.9f} 米。")
        lines.append(f"物理审计通过：{state['eight_cycle']['physical_audit_passed']}；策略审计通过：{state['eight_cycle']['policy_audit_passed']}；录像解码通过：{state['eight_cycle']['video_decode_passed']}。")
    lines += ['跨周期保持物理状态连续；第一周期为 boundary_07，随后为标称布局。不是八周期专门训练，也不是感知验证。',
        '私有协作仓库已补依赖和相对路径；干净克隆、新虚拟环境完成单周期复现，11组数组完全一致。',
        '单线程复现也通过，但负载不同，未作性能提升结论，未改默认线程设置。', '', '二、新几何基线',
        f"16 个新训练几何已完成 {len(rows)} 个，其中 {sum(r['success'] for r in rows)} 个通过；8 个旧保护工况复用已审计轨迹。",
        f"整组基线及审计完成：{bool(baseline and baseline['completed'])}。",
        f"代表性训练环境重放：已启动={gate_started}，已完成={gate is not None}，通过={bool(gate and gate['passed'])}。",
        '新样本位于 boundary_07 附近，水平半宽 0.75 mm、竖直半宽 0.075 mm；范围很窄，不能称广泛随机布局泛化。',
        'training_00/01 在约77.57秒左脚摆动后段再次触杆而失败；已离杆后再接触，不是初始未抬起。',
        '当前固定校准使膝/踝正向动作上限为0.5；这是动作接口上限，不是电机力矩饱和，也不证明增加动作必然有效。',
        f'独立保留的16个验证几何未运行；131,072步新几何训练已启动={training_started}。', '', '三、下一步与复现命令',
        '以下命令记录依赖顺序；输出目录存在时不要重复启动。先完成基线和重放门槛，通过后才训练。',
        r'.\.venv\Scripts\python.exe -m tools.prepare_geometry_training_gate --baseline outputs/calibrated_geometry_training_baseline_20261027_v1 --geometry-protocol outputs/calibrated_geometry_curriculum_protocol_20261027_v1 --output outputs/calibrated_geometry_training_gate_20261027_v1',
        r'.\.venv\Scripts\python.exe -m tools.train_calibrated_landing --protocol outputs/calibrated_geometry_training_gate_20261027_v1 --mode average --output outputs/calibrated_geometry_ppo131072_20261027_v1',
        '门槛只重放4个代表工况，不等同于24工况全量重放。训练后必须补参数审计、训练分布回归和保留验证集评估。',
        '最终模型冻结后再评估保留验证集；不使用验证结果选择中间检查点。',
        r'训练结束后参数审计：.\.venv\Scripts\python.exe -m tools.audit_geometry_training_update outputs/calibrated_geometry_ppo131072_20261027_v1',
        '已增加 tools/evaluate_geometry_training_regression.py，支持24工况训练模型回归旧8/12/16工况集；目前只通过导入和命令行检查，尚未实跑。',
        '已增加 tools/evaluate_geometry_holdout.py，只允许完成131,072步训练和参数审计后的最终模型运行；目前仅通过导入和命令行检查，未评估保留集。',
        r'保留集评估命令（训练和审计完成后）：.\.venv\Scripts\python.exe -m tools.evaluate_geometry_holdout --candidate outputs/calibrated_geometry_ppo131072_20261027_v1 --geometry-protocol outputs/calibrated_geometry_curriculum_protocol_20261027_v1 --output outputs/calibrated_geometry_holdout16_20261027_v1',
        '本报告是生成时快照；运行中的基线结果以日志和 baseline_result.json 为准。']
    if diagnosis:
        lines += ['', '四、失败终点量化（阈值保持不变）',
            '关节闭合阈值40 mrad；根部抬升上限260 mm。提前终止的触杆工况没有收尾指标。']
        for case, row in diagnosis['rows'].items():
            if row['complete']:
                lines.append(f"{case}：关节 {row['closure_mrad']:.6f} mrad；抬升 {row['rise_max_mm']:.6f} mm。")
    (dest / 'report.txt').write_text('\n'.join(lines), encoding='utf-8-sig')
    (dest / 'summary.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    (dest / 'source.py').write_bytes(Path(__file__).read_bytes())
    (out / 'research_session_status_20261027.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    print(json.dumps(state, ensure_ascii=True))


if __name__ == '__main__':
    main()
