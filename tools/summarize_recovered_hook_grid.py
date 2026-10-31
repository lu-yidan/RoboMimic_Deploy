"""Supplement frozen grid outcomes with copy-only audit recovery evidence."""
import argparse
import copy
from pathlib import Path
from tools.validate_swing_reference import read, write, sha
from tools.summarize_compiled_hook_ablation import complete_metrics, ARMS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--recovery',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=copy.deepcopy(read(a.analysis));recovery=read(a.recovery/'summary.json')
    original_counts=data['passed_counts'].copy();supplements=[]
    for recovered in recovery['recoveries']:
        if not recovered['audit_passed']:continue
        assert recovered['original_unchanged'] and not recovered['physics_rerun']
        original=Path(recovered['original_row']['run']);run=Path(recovered['copy'])
        assert all(sha(original/name)==digest for name,digest in recovered['original_files'].items())
        assert sha(run/'compiled_pair_audit.json')==recovered['audit_sha256']
        audit=read(run/'compiled_pair_audit.json')
        assert audit['passed'] and audit['integrity_passed']
        case=next(c for c in data['cases'] if c['case']==recovered['case'])
        row=case['arms'][recovered['arm']]
        assert not row['passed'] and row['complete'] and not row.get('integrity_passed')
        assert row['trajectory_sha256']==audit['trajectory_sha256']==sha(run/'trajectory.npz')
        assert row['telemetry_sha256']==audit['telemetry_sha256']==sha(run/'physics_telemetry.npz')
        row['original_result_before_audit_recovery']=copy.deepcopy(row)
        row.update(passed=True,integrity_passed=True,status='audit_recovered_success',
            recovery_returncode=recovered['recovery_returncode'],recovery_audit_sha256=recovered['audit_sha256'],
            recovered_copy=str(run),complete_cycle_metrics=complete_metrics(run))
        case['all_three_complete_and_audited']=all(r['complete_cycle_metrics'] is not None for r in case['arms'].values())
        available=[r['complete_cycle_metrics'] for r in case['arms'].values() if r['complete_cycle_metrics'] is not None]
        for key in ('case','offsets','reference_sha256','checkpoint_sha256','scene_geometry_sha256'):
            assert all(item[key]==available[0][key] for item in available),key
        supplements.append(dict(case=case['case'],arm=recovered['arm'],original_status=recovered['original_row']['status']))
    data.update(original_execution_passed_counts=original_counts,
        passed_counts={arm:sum(c['arms'][arm]['passed'] for c in data['cases']) for arm in ARMS},
        supplemented_records=supplements,original_analysis_sha256=sha(a.analysis),
        recovery_summary_sha256=sha(a.recovery/'summary.json'),source_sha256=sha(__file__),
        scope='Supplemental scientific view after audit-only recovery. Original execution statuses retained in each recovered row and original analysis unchanged. No physics reruns, training, changed denominator or automatic promotion.')
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'summary.json',data)
    lines=['# 网格审计补齐后的结果视图','',
        '原执行报告和原图表不改写。下表分开列出首次执行通过数与补审后通过数；审计进程异常不等于物理失败。','',
        '| 方案 | 首次执行完整通过 | 补审后完整通过 | 总工况 |','|---|---:|---:|---:|']
    for arm in ARMS:lines.append(f"| {arm} | {original_counts[arm]} | {data['passed_counts'][arm]} | 16 |")
    lines+=['',f'本次补齐 {len(supplements)} 条记录；未重新运行任何物理轨迹。',
        '所有完整周期指标仍要求记录完整且通过完整性审计。原始错误、补审来源和哈希见 summary.json。',
        '这仍是预先声明的局部开发网格，不证明传感器鲁棒性、随机墙面泛化或硬件部署能力。']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(dict(original_counts=original_counts,supplemented_counts=data['passed_counts'],recoveries=len(supplements)),flush=True)


if __name__=='__main__':main()
