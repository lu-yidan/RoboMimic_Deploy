"""Consolidate completed audited calibration, PPO training and transfer evidence."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from tools.compare_audited_closure_candidates import load_run, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--neighborhood', type=Path)
    p.add_argument('--remaining-percent', type=int, required=True)
    a = p.parse_args()
    evidence = {}

    def read(path):
        path = Path(path)
        value = json.loads(path.read_text())
        evidence[str(path)] = sha(path)
        return value

    training = Path('outputs/calibrated_landing_ppo32768_20261027_v1')
    suite_dirs = {'old16': training,
        'original8': Path('outputs/calibrated_landing_ppo32768_original8_20261027_v1'),
        'grid12': Path('outputs/calibrated_landing_ppo32768_grid12_20261027_v1')}
    suites = {}
    geometries = []
    for name, directory in suite_dirs.items():
        run = load_run(directory)
        for f in ('independent_candidate_audit.json', 'composite_prefix_audit_v2.json', 'controller_modifier_audit_v2.json', 'adaptive_reference_audit.json'):
            assert read(directory/f)['passed']
        read(directory/'manifest.json'); read(directory/'summary.json')
        assert run['checkpoint_sha256'] == sha(training/'policy.zip')
        suites[name] = dict(directory=str(directory.resolve()), rows=run['rows'],
            successes=sum(row['success'] for row in run['rows'].values()), cases=len(run['rows']))
        geometries.extend(json.dumps(g, sort_keys=True) for g in run['cases'].values())
    assert len(geometries) == len(set(geometries)) == 36
    update = read(training/'parameter_update_audit.json')
    assert update['passed']
    trained = read(training/'summary.json')
    compare = read('outputs/calibrated_landing_ppo32768_compare16_20261027_v1/comparison.json')
    fixed_report = read('outputs/boundary_repair_report_20261027_v1/summary.json')
    local_fixed = read('outputs/boundary_combined_compare_neighborhood_20261027_v1/comparison.json')
    local_trained = None
    if a.neighborhood:
        run = load_run(a.neighborhood)
        for f in ('independent_candidate_audit.json', 'composite_prefix_audit_v2.json', 'controller_modifier_audit_v2.json', 'adaptive_reference_audit.json'):
            assert read(a.neighborhood/f)['passed']
        read(a.neighborhood/'summary.json'); read(a.neighborhood/'manifest.json')
        assert run['checkpoint_sha256'] == sha(training/'policy.zip')
        local_trained = dict(rows=run['rows'], successes=sum(r['success'] for r in run['rows'].values()), cases=len(run['rows']),
            scope='Previously evaluated local development cases. Not a new blind generalization test; not used for this training.')
    transfer = Path('outputs/boundary_combined_eight_handoff_20261027_v1')
    multi = read(transfer/'summary.json')
    physical = read(transfer/'independent_audit.json')
    action = read(transfer/'policy_transfer_audit.json')
    render = read(transfer/'render_manifest.json')
    manifest = read(transfer/'manifest.json')
    assert multi['passed'] and physical['passed'] and action['passed'] and render['decode_passed']
    assert sha(transfer/'trajectory.npz') == multi['trajectory_sha256'] == physical['trajectory_sha256'] == action['trajectory_sha256']
    assert sha(Path(manifest['parent_experiment'])/'policy.zip') != sha(training/'policy.zip')
    with np.load(transfer/'trajectory.npz') as z:
        root_rise = float(z['qpos'][-1, 2]-z['qpos'][0, 2])
    total = sum(s['successes'] for s in suites.values())
    selected = [r for r in compare['rows'] if r['case'] in ('boundary_05', 'boundary_06', 'boundary_07')]
    result = dict(checked_at=datetime.now().astimezone().isoformat(), remaining_percent_at_check=a.remaining_percent,
        fixed_calibration_known_successes=fixed_report['after_successes'], trained_known_successes=total,
        known_cases=36, suites=suites, trained_checkpoint_sha256=sha(training/'policy.zip'),
        additional_training_steps=trained['additional_steps'], training_wall_seconds=trained['training_wall_seconds'],
        selected_endpoints=selected, local_fixed=local_fixed, local_trained=local_trained,
        frozen_policy_transfer=dict(cycles=len(multi['cycle_results']), time_s=multi['time_s'], root_rise_m=root_rise,
            physical_passed=physical['passed'], action_passed=action['passed'], new_ppo_policy_used=False,
            video=str((transfer/'calibrated_transfer_8_cycles_16x.mp4').resolve())),
        automatic_promotion=False, evidence_sha256=evidence, source_sha256=sha(__file__),
        scope='Known-case repair and limited single-seed PPO follow-up; separate frozen-policy 2m transfer. Scripted reference, exact geometry and ideal force feedback remain. No hardware or autonomous vision claim.')
    lines = ['G1 钩爪攀爬：三个失败工况修复、PPO 续训与连续爬升报告',
        f'核验时间：{result["checked_at"]}；最近检查剩余额度 {a.remaining_percent}%。', '',
        '一、原有三个失败工况已修复',
        '先诊断末段左踝动作饱和，再联合固定膝踝校准与参考量校准。原 PPO 权重不变，36 个已知工况从 33/36 提升到 36/36。',
        '保持关节误差 40 mrad、单周期根部抬升 240–260 mm、原生关节及力矩约束，没有放宽验收。', '',
        '二、校准后追加真实 PPO 训练',
        f'新增 {trained["additional_steps"]:,} 步，训练耗时 {trained["training_wall_seconds"]:.2f} 秒；12 个完整训练回合成功。参数审计确认 actor 和 critic 均更新。',
        f'独立完整周期核验：十六工况 {suites["old16"]["successes"]}/16；原八工况 {suites["original8"]["successes"]}/8；十二网格 {suites["grid12"]["successes"]}/12；共 {total}/36。',
        '训练固定原校准参数，仅使用原八个训练工况。邻近八工况没有加入训练。',
        '以下对比为固定校准起点与续训后，不能写成未校准原策略与新策略的对比：']
    for r in selected:
        lines.append(f'{r["case"]}：关节误差 {r["before_closure_mrad"]:.6f} / {r["after_closure_mrad"]:.6f} mrad；最大抬升 {r["before_rise_max_mm"]:.6f} / {r["after_rise_max_mm"]:.6f} mm。')
    lines += ['', '三、连续约 2 米记录',
        f'冻结校准策略完成 {multi["time_s"]:.2f} 秒八周期连续仿真，净抬升 {root_rise:.6f} m，独立物理端点与动作重算审计均通过。',
        '周期间没有物理状态和滤波历史重置；首周期为 boundary_07，后续为名义几何。仍保留原阶段/增益重置与脚本参考。',
        '这条记录使用原冻结 PPO，不是新 32,768 步模型的八周期结果。录像为真实保存状态的 16 倍速重放，已检查解码。', '',
        '四、必须保留的局限',
        '最弱工况的余量仍很小；单随机种子、有限步数训练不构成收敛证明。',
        '冻结校准在预声明的局部八点扰动上仅 3/8，原策略 0/8。已知 36/36 不代表稳健泛化。']
    if local_trained:
        lines.append(f'新 PPO 在同一局部开发集上 {local_trained["successes"]}/8；这次是重复比较，不能再称新盲测。')
    lines += ['最高杆固定校准后，末两秒根部加速度分量 RMS 约增加 7.5%；不能声称所有动力学指标改善。',
        '当前仍依赖脚本参考、精确几何和理想力反馈。未完成真实视觉闭环、随机短杆、硬件实验或新 PPO 跨周期训练。', '',
        '五、下一步',
        '优先扩大稳健余量：保持验收门槛，用预先划分的训练/验证几何分布处理局部失败，再做多随机种子训练。',
        '新 PPO 如通过单周期回归，再单独核验连续两周期与八周期；不能复用旧策略的连续成功结论。',
        '没有自动替换默认策略、推送 GitHub 或连接真机。', '',
        '视频：'+result['frozen_policy_transfer']['video'],
        '训练检查点：'+str((training/'policy.zip').resolve()),
        '全部机器可读结果及哈希：本目录 summary.json。']
    a.output.mkdir(exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    (a.output/'report.txt').write_text('\n'.join(lines), encoding='utf-8-sig')
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(known_successes=total, local_trained=local_trained and local_trained['successes'], output=str(a.output))))


if __name__ == '__main__':
    main()
