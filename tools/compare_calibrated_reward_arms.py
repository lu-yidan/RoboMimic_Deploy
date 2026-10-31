"""Audit the matched initialization/budget and report both calibrated reward arms."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from tools.compare_audited_closure_candidates import load_run, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--average', type=Path, required=True)
    p.add_argument('--bottleneck', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    runs = {}
    evidence = {}
    manifests = {}
    protocols = {}
    for mode, directory in [('average', a.average), ('bottleneck', a.bottleneck)]:
        run = load_run(directory)
        m = json.loads((directory/'manifest.json').read_text())
        s = json.loads((directory/'summary.json').read_text())
        protocol = json.loads((directory/'matched_protocol.json').read_text())
        assert m['boundary_reward_mode'] == mode and s['additional_steps'] == 32768
        for name in ('parameter_update_audit.json', 'independent_candidate_audit.json',
                     'composite_prefix_audit_v2.json', 'controller_modifier_audit_v2.json', 'adaptive_reference_audit.json'):
            report = json.loads((directory/name).read_text())
            assert report['passed']
            evidence[str(directory/name)] = sha(directory/name)
        for name in ('manifest.json', 'summary.json', 'matched_protocol.json', 'policy.zip'):
            evidence[str(directory/name)] = sha(directory/name)
        runs[mode] = dict(directory=str(directory.resolve()), rows=run['rows'],
            successes=sum(r['success'] for r in run['rows'].values()),
            minimum_scaled_margin=min(r['scaled_margin'] for r in run['rows'].values() if r['scaled_margin'] is not None),
            training_episodes=s['completed_training_episodes'], training_successes=s['successful_training_episodes'],
            training_wall_seconds=s['training_wall_seconds'], case_samples=s['case_samples'])
        manifests[mode], protocols[mode] = m, protocol
    keys = ('parent_checkpoint_sha256', 'initial_checkpoint_sha256', 'prefix_checkpoint_sha256',
            'config', 'training_cases', 'evaluation_cases', 'closure_action_bias', 'controller_modifier',
            'adaptive_reference_command', 'seed', 'environments', 'quality_bonus_weight')
    matching = {k: manifests['average'][k] == manifests['bottleneck'][k] for k in keys}
    for key in ('partitions', 'additional_steps_per_arm'):
        matching[key] = protocols['average'][key] == protocols['bottleneck'][key]
    for key, value in protocols['average']['sources'].items():
        if key.replace('\\', '/').startswith('training/'):
            matching['source:'+key] = value == protocols['bottleneck']['sources'].get(key)
    assert all(matching.values()), matching
    rows = []
    for case, before in runs['average']['rows'].items():
        after = runs['bottleneck']['rows'][case]
        rows.append(dict(case=case, average=before, bottleneck=after))
    gained = [r['case'] for r in rows if r['bottleneck']['success'] and not r['average']['success']]
    lost = [r['case'] for r in rows if r['average']['success'] and not r['bottleneck']['success']]
    result = dict(checked_at=datetime.now().astimezone().isoformat(), matching_checks=matching, arms=runs,
        rows=rows, gained=gained, lost=lost, evidence_sha256=evidence, source_sha256=sha(__file__),
        automatic_promotion=False,
        scope='One seed, same initial actor/optimizer, budget, cases, calibration and controller. Bottleneck arm selected after observing average-arm results. Development comparison, not prospective randomization or statistical superiority. Eight local neighborhood cases are outside training and reported separately.')
    lines = ['固定校准后的奖励对照（单随机种子）', '',
             '共同起点、优化器、32,768 步预算、训练工况和固定校准一致；仅奖励聚合方式不同。',
             '最弱约束组是在平均奖励组结果出来后追加，因此不能称为前瞻随机试验，也不声称统计显著。', '']
    for mode, r in runs.items():
        lines.append(f'{mode}：完整开发工况 {r["successes"]}/{len(r["rows"])}；最小归一化余量 {r["minimum_scaled_margin"]:.6f}。')
    lines += [f'新增通过：{gained}；新增失败：{lost}。', '',
              '归一化余量取关节余量/1 mrad 与高度余量/0.1 mm 的较小值，不是概率或置信度。',
              '工况 | average 关节 mrad / 高度 mm | bottleneck 关节 mrad / 高度 mm']
    for row in rows:
        b, c = row['average'], row['bottleneck']
        lines.append(f'{row["case"]}: {b["closure_mrad"]} / {b["rise_max_mm"]} | {c["closure_mrad"]} / {c["rise_max_mm"]}')
    lines += ['', '失败必须保留；不自动替换现有模型。其他工况回归、连续爬升及邻近扰动结果单独计入。']
    a.output.mkdir(exist_ok=False)
    (a.output/'comparison.json').write_text(json.dumps(result, indent=2))
    (a.output/'report.txt').write_text('\n'.join(lines), encoding='utf-8-sig')
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(successes={k:v['successes'] for k,v in runs.items()}, gained=gained, lost=lost)))


if __name__ == '__main__':
    main()
