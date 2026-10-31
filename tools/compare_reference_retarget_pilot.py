"""Compare one new-reference pilot with its predeclared same-scene baseline."""
import argparse
from pathlib import Path
from tools.validate_swing_reference import read, write, sha
from tools.summarize_compiled_hook_ablation import complete_metrics


def inspect(run):
    result=dict(run=str(run),audit_exists=False,physical_policy_passed=False,complete_metrics=None)
    if not (run/'manifest.json').exists():
        result['status']='No run manifest; inspect execution log'
        return result
    manifest=read(run/'manifest.json')
    result.update(manifest=manifest,manifest_sha256=sha(run/'manifest.json'))
    if (run/'summary.json').exists():
        summary=read(run/'summary.json')
        result.update(reason=summary['reason'],time_s=summary['time_s'])
    audit_path=run/'compiled_pair_audit.json'
    if audit_path.exists():
        audit=read(audit_path)
        assert audit['manifest_sha256']==sha(run/'manifest.json')
        assert audit['trajectory_sha256']==sha(run/'trajectory.npz')
        assert audit['telemetry_sha256']==sha(run/'physics_telemetry.npz')
        result.update(audit_exists=True,audit_sha256=sha(audit_path),physical_policy_passed=audit['passed'])
        if audit['integrity_passed'] and abs(result.get('time_s',0)-227.32)<1e-7:
            result['complete_metrics']=complete_metrics(run)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--recovery',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    declared=read(a.protocol);baseline=Path(declared['baseline_run']);selected=baseline
    recovery=read(a.recovery/'summary.json')
    # An audit-only recovered copy can supplement the preserved native-error row.
    for row in recovery['recoveries']:
        if row['case']==declared['case'] and row['arm']=='adopted_v2' and row['audit_passed']:
            assert row['original_unchanged'] and not row['physics_rerun']
            selected=Path(row['copy'])
    old,new=inspect(selected),inspect(Path(declared['output']))
    if 'manifest' in old and 'manifest' in new:
        for key in ('case','offsets','checkpoint_sha256','prefix_checkpoint_sha256',
                    'scene_geometry_sha256','parent_manifest_sha256','arm'):
            assert old['manifest'][key]==new['manifest'][key],key
        assert old['manifest']['plan_sha256']!=new['manifest']['plan_sha256']==declared['plan_sha256']
    differences=None
    if old['complete_metrics'] is not None and new['complete_metrics'] is not None:
        x,y=old['complete_metrics']['metrics'],new['complete_metrics']['metrics']
        differences={key:(y[key]-x[key]) for key in x if isinstance(x[key],(int,float))}
        differences['support_net_force_peaks_n']={key:y['support_net_force_peaks_n'][key]-value for key,value in x['support_net_force_peaks_n'].items()}
    report=dict(protocol_sha256=sha(a.protocol),recovery_summary_sha256=sha(a.recovery/'summary.json'),
        baseline=old,candidate=new,complete_second_cycle_differences_candidate_minus_baseline=differences,
        baseline_original_location=str(baseline),baseline_uses_recovered_copy=selected!=baseline,
        source_sha256=sha(__file__),new_training_steps=0,automatic_promotion=False,
        scope='One predeclared oracle-target reference pilot. All outcomes retained. Complete-cycle differences only for complete integrity-audited records. No fixed-time or contact-confirmed loading candidate is mixed into this comparison. No camera or random-wall generalization claim.')
    write(a.output/'summary.json',report)
    lines=['# 全身手目标参考：单工况动力学试验','',
        f"- 原参考物理/策略审计通过：{old['physical_policy_passed']}",
        f"- 新参考物理/策略审计通过：{new['physical_policy_passed']}",
        '- 正式 v2 未替换；未训练。','',
        '若记录中断或缺完整性审计，不以局部峰值宣称完整周期改善。指标均为第二周期。']
    if differences is not None:
        lines += ['','| 指标 | 原参考 | 新参考 |','|---|---:|---:|']
        x,y=old['complete_metrics']['metrics'],new['complete_metrics']['metrics']
        for key in ('root_acceleration_peak_m_s2','joint_acceleration_peak_rad_s2','closure_mrad','summed_squared_torque_nm2_s'):
            lines.append(f'| {key} | {x[key]:.6g} | {y[key]:.6g} |')
        lines += ['','力矩平方积分是控制代价代理，不等于电耗。完整数值和各支撑组接触合力见 summary.json。']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(dict(baseline_passed=old['physical_policy_passed'],candidate_passed=new['physical_policy_passed'],complete_pair=differences is not None),flush=True)


if __name__=='__main__':main()
