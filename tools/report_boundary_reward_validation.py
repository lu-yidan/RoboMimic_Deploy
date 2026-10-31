"""Collect completed reward training, regression and telemetry evidence."""
import argparse
import json
import sys
from pathlib import Path
from tools.compare_audited_closure_candidates import load_run, sha


def read(path):
    return json.loads(path.read_text())


def number(value):
    return '未完成，无末段指标' if value is None else f'{value:.6f}'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    state = read(root/'status.json')
    assert state['completed'] and all(s['exit_code'] == 0 for s in state['steps'])
    matched = read(root/'matched_comparison'/'summary.json')
    assert matched['passed'] and all(matched['matched_checks'].values())
    result = dict(completed=True, arms={}, evidence_hashes={}, automatic_promotion=False,
                  scope='One seed, equal budget, known development cases. Geometry counts are deduplicated; no independent generalization claim.')
    lines = ['收尾奖励匹配训练与验证结果', '两组同起点、同配置，各新增 65,536 个 50 Hz 采样步；原验收门槛不变。',
             '学习窗口为 70.2–113.66 秒的落脚与收尾；之前的动作由冻结策略执行。',
             '最终验证均从零运行完整 113.66 秒周期。', '']
    protocol = read(Path(state['protocol'])/'protocol.json')
    parent_paths = {'development16': Path(protocol['parent'])}
    for suite in ['original8', 'grid12']:
        parent_paths[suite] = Path(read(root/('average_'+suite)/'manifest.json')['parent_experiment'])
    parent_runs = {suite: load_run(path) for suite, path in parent_paths.items()}
    assert len({run['checkpoint_sha256'] for run in parent_runs.values()}) == 1
    result['parent_suites'] = {suite: dict(rows=run['rows'], successes=sum(r['success'] for r in run['rows'].values()),
        total=len(run['rows']), audit_sha256=run['audit_sha256']) for suite, run in parent_runs.items()}
    lines += ['共同起点：' + ', '.join(f"{suite}: {data['successes']}/{data['total']}" for suite, data in result['parent_suites'].items()), '']
    for mode in ['average', 'bottleneck']:
        arm = dict(training=matched['outcomes'][mode], suites={})
        unique = {}
        checkpoints = set()
        for suite, directory in [('development16', root/mode), ('original8', root/(mode+'_original8')), ('grid12', root/(mode+'_grid12'))]:
            run = load_run(directory)
            checkpoints.add(run['checkpoint_sha256'])
            for case, geometry in run['cases'].items():
                key = json.dumps({k: [float(x) for x in v] for k, v in geometry.items() if any(v)}, sort_keys=True)
                row = run['rows'][case]
                if key in unique:
                    assert unique[key]['success'] == row['success']
                    for metric in ['closure_mrad', 'rise_max_mm']:
                        first, second = unique[key][metric], row[metric]
                        assert first == second or (first is not None and second is not None and abs(first-second) < 1e-7)
                else:
                    unique[key] = row
            for name in ['composite_prefix_audit_v2.json', 'controller_modifier_audit_v2.json', 'adaptive_reference_audit.json']:
                audit_path = directory/name
                assert read(audit_path)['passed']
                result['evidence_hashes'][str(audit_path.resolve())] = sha(audit_path)
            arm['suites'][suite] = dict(successes=sum(r['success'] for r in run['rows'].values()),
                total=len(run['rows']), failed=[c for c, r in run['rows'].items() if not r['success']],
                rows=run['rows'], audit_sha256=run['audit_sha256'])
            if suite != 'development16':
                arm['suites'][suite]['regression'] = read(directory/'regression.json')
        assert len(checkpoints) == 1
        arm['unique_geometry_coverage'] = dict(total=len(unique), successes=sum(r['success'] for r in unique.values()))
        telemetry = root/(mode+'_telemetry')/'telemetry_audit.json'
        assert read(telemetry)['passed']
        arm['highest_rod_telemetry'] = read(telemetry)['rows'][0]
        result['evidence_hashes'][str(telemetry.resolve())] = sha(telemetry)
        diagnosis_path = root/(mode+'_endpoints')/'summary.json'
        arm['endpoint_diagnosis'] = read(diagnosis_path)
        assert arm['endpoint_diagnosis']['audit_sha256'] == arm['suites']['development16']['audit_sha256']
        result['evidence_hashes'][str(diagnosis_path.resolve())] = sha(diagnosis_path)
        result['arms'][mode] = arm
        label = '平均质量奖励' if mode == 'average' else '瓶颈质量奖励'
        counts = ', '.join(f"{suite}: {data['successes']}/{data['total']}" for suite, data in arm['suites'].items())
        highest = arm['suites']['original8']['rows']['foot_z_plus15']
        lines += [label + '：' + counts,
            f"去重后已知几何：{arm['unique_geometry_coverage']['successes']}/{len(unique)}。",
            f"最高杆：{number(highest['closure_mrad'])} mrad，末段最大根部抬升 {number(highest['rise_max_mm'])} mm，通过={highest['success']}。",
            f"最高杆 2000 Hz 遥测：{arm['highest_rod_telemetry']['physics_samples']} 个样本，末两秒根部线加速度三分量 RMS={arm['highest_rod_telemetry']['final_root_acceleration_rms_m_s2']:.9f} m/s²。",
            '末两秒支撑竖直力最小值 (N)：'+', '.join(f'{key}={value:.3f}' for key, value in arm['highest_rod_telemetry']['final_support_vertical_minimum_n'].items()),
            '十六工况失败项：' + str(arm['suites']['development16']['failed']), '']
        names = dict(joint_closure_rad='关节收尾误差', root_rise_m='根部抬升高度',
                     effector_error_m='末端位置误差', tilt_deg='机身倾角',
                     linear_speed_m_s='线速度', angular_speed_rad_s='角速度', joint_speed_rad_s='关节速度')
        for row in arm['endpoint_diagnosis']['rows']:
            if row['success']:
                continue
            failed = [names.get(key, key) for key, slack in row.get('signed_slack', {}).items() if slack < 0]
            lines.append(row['case']+' 未满足：'+('、'.join(failed) if failed else '需查看原轨迹失败原因'))
        lines.append('')
    result['bottleneck_gained_vs_average'] = matched['bottleneck_gained_vs_average']
    result['bottleneck_lost_vs_average'] = matched['bottleneck_lost_vs_average']
    result['all_suite_changes'] = {}
    for suite in ['development16', 'original8', 'grid12']:
        average = result['arms']['average']['suites'][suite]['rows']
        bottleneck = result['arms']['bottleneck']['suites'][suite]['rows']
        assert set(average) == set(bottleneck)
        result['all_suite_changes'][suite] = dict(
            gained=[case for case in average if bottleneck[case]['success'] and not average[case]['success']],
            lost=[case for case in average if average[case]['success'] and not bottleneck[case]['success']])
    added = any(row['gained'] for row in result['all_suite_changes'].values())
    lost = any(row['lost'] for row in result['all_suite_changes'].values())
    result['conclusion'] = ('瓶颈奖励出现新增失败，需要保留回归风险，不能直接推广。' if lost else
                            '瓶颈奖励在本轮增加了通过工况，仍需更多随机种子验证。' if added else
                            '瓶颈奖励未增加通过工况；本轮不支持将其作为更好的默认策略。')
    result['physics_comparison'] = read(root/'physics_comparison'/'summary.json')
    lines += ['瓶颈奖励相对平均奖励（十六工况）：',
        '新增通过：' + str(result['bottleneck_gained_vs_average']),
        '新增失败：' + str(result['bottleneck_lost_vs_average']), '',
        '结论：'+result['conclusion'], '',
        '以上是单种子、已知开发工况比较；去重数量仍不是独立随机测试样本数。',
        '未自动替换默认模型。接触力与根部加速度只覆盖最高杆一个确定性工况。']
    for path in [root/'status.json', root/'matched_comparison'/'summary.json', root/'physics_comparison'/'summary.json']:
        result['evidence_hashes'][str(path.resolve())] = sha(path)
    result['source_sha256'] = sha(__file__)
    (root/'final_report.json').write_text(json.dumps(result, indent=2))
    (root/'final_report.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8-sig')
    (root/'final_report_source.py').write_bytes(Path(__file__).read_bytes())
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
