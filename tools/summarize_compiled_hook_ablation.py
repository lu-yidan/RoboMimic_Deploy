"""Join all 48 predeclared records; incomplete traces never imply lower peaks."""
import argparse
from pathlib import Path
import json
import numpy as np
from tools.validate_swing_reference import read,write,sha
from tools.audit_timed_trace_coverage import audit as audit_coverage

ARMS=('adopted_v2','contact_confirmed','timed_open_loop')


def verified_rows(folder,timed):
    summary=read(folder/'suite_summary.json');protocol=read(folder/'protocol.json')
    assert summary['finished'] and summary['sources_unchanged']
    assert summary['protocol_sha256']==sha(folder/'protocol.json')
    assert summary['declared_runs']==(16 if timed else 32)
    for relative,digest in protocol['sources'].items():
        assert sha(folder/'source_snapshot'/relative)==digest
    for name,digest in summary['result_sha256'].items():assert sha(folder/'results'/name)==digest
    for row in summary['rows']:
        name=f"{row['case']}.json" if timed else f"{row['case']}_{row['arm']}.json"
        assert read(folder/'results'/name)==row
        if 'run' not in row:continue
        run=Path(row['run'])
        if 'audit_sha256' in row:
            assert sha(run/'compiled_pair_audit.json')==row['audit_sha256']
            audit=read(run/'compiled_pair_audit.json')
            assert audit['manifest_sha256']==sha(run/'manifest.json')
            assert audit['trajectory_sha256']==sha(run/'trajectory.npz')==row['trajectory_sha256']
            assert audit['telemetry_sha256']==sha(run/'physics_telemetry.npz')==row['telemetry_sha256']
            assert bool(row['process_returncode']==0 and audit['passed'])==row['passed']
        elif row['passed']:raise AssertionError('A successful row lacks an independent audit')
    return summary,protocol


def complete_metrics(run):
    m=read(run/'manifest.json');s=read(run/'summary.json')
    record=read(run/'physics_telemetry_record.json')
    assert abs(s['time_s']-227.32)<1e-7
    assert sha(run/'physics_telemetry.npz')==record['telemetry_sha256']
    with np.load(run/'physics_telemetry.npz') as f:
        t=f['time_s'];selected=t>113.66+1e-7
        assert len(t)==454640 and int(selected.sum())==227320
        acceleration=f['solver_qacc'][selected];torque=f['motor_torque_nm'][selected]
        forces=f['group_force_world_n'][selected]
    force_peaks={g:float(np.linalg.norm(forces[:,j],axis=1).max()) for j,g in enumerate(record['groups'])}
    metrics=dict(root_acceleration_peak_m_s2=float(np.linalg.norm(acceleration[:,:3],axis=1).max()),
        root_acceleration_rms_m_s2=float(np.sqrt(np.mean(np.sum(acceleration[:,:3]**2,axis=1)))),
        joint_acceleration_peak_rad_s2=float(abs(acceleration[:,6:]).max()),
        support_net_force_peaks_n=force_peaks,summed_squared_torque_nm2_s=float(np.sum(torque**2)*.0005),
        closure_mrad=s['cycle_results'][1]['closure_mrad'],rise_m=s['final_rise_m'])
    events=[]
    if m['arm']=='contact_confirmed':
        a=read(run/'hook_admission_audit.json')
        events=[dict(group=e['group'],delay_s=e['confirmed_s']-e['start_s']) for e in a['events'] if e['confirmed_s'] is not None]
    elif m['arm']=='timed_open_loop':
        a=read(run/'timed_loading_audit.json')
        events=[dict(group=e['group'],delay_s=e['release_s']-e['start_s']) for e in a['expected_events'] if e['release_s'] is not None]
    return dict(metrics=metrics,events=events,case=m['case'],offsets=m['offsets'],
        reference_sha256=m['plan_sha256'],checkpoint_sha256=m['checkpoint_sha256'],
        scene_geometry_sha256=m['scene_geometry_sha256'],manifest_sha256=sha(run/'manifest.json'),
        trajectory_sha256=sha(run/'trajectory.npz'),telemetry_sha256=record['telemetry_sha256'])


def summarize(paired,timed,out):
    two,p=verified_rows(paired,False);third,q=verified_rows(timed,True)
    assert q['paired_protocol_sha256']==sha(paired/'protocol.json')
    assert p['cases']==q['cases'] and len(p['cases'])==16
    assert p['single_manifest_sha256']==q['single_manifest_sha256']
    assert two['single_chain_sha256']==third['single_chain_sha256']
    out.mkdir(parents=True,exist_ok=False);(out/'trace_coverage').mkdir()
    rows=two['rows']+third['rows']
    assert len(rows)==len({(r['case'],r['arm']) for r in rows})==48
    cases=[];coverage_reports={}
    for case in p['cases']:
        arms={r['arm']:dict(r) for r in rows if r['case']==case};assert set(arms)==set(ARMS)
        if any(r['status']=='first_cycle_gate_failed' for r in arms.values()):
            assert all(r['status']=='first_cycle_gate_failed' and not r['passed'] for r in arms.values())
        for arm,row in arms.items():
            row['complete_cycle_metrics']=None
            if row['complete'] and row.get('integrity_passed'):
                run=Path(row['run'])
                if arm=='timed_open_loop':
                    target=out/'trace_coverage'/f'{case}.json'
                    coverage=audit_coverage(run,target)
                    coverage_reports[case]=dict(passed=coverage['passed'],sha256=sha(target))
                    assert coverage['passed'],f'Incomplete timed trace: {case}'
                row['complete_cycle_metrics']=complete_metrics(run)
        complete=[r['complete_cycle_metrics'] for r in arms.values() if r['complete_cycle_metrics']]
        if complete:
            for k in ('case','offsets','reference_sha256','checkpoint_sha256','scene_geometry_sha256'):
                assert all(r[k]==complete[0][k] for r in complete),f'Mismatched paired input: {case} {k}'
        cases.append(dict(case=case,offsets=p['cases'][case],arms=arms,
            all_three_complete_and_audited=all(r['complete_cycle_metrics'] is not None for r in arms.values())))
    counts={arm:sum(c['arms'][arm]['passed'] for c in cases) for arm in ARMS}
    report=dict(declared_cases=16,declared_records=48,passed_counts=counts,cases=cases,
        paired_summary_sha256=sha(paired/'suite_summary.json'),timed_summary_sha256=sha(timed/'suite_summary.json'),
        timed_trace_coverage=coverage_reports,source_sha256=sha(__file__),
        coverage_source_sha256=sha(Path(__file__).with_name('audit_timed_trace_coverage.py')),
        new_training_steps=0,automatic_promotion=False,
        scope='Full 16-case denominator for each arm. Complete second-cycle metrics only when the saved record has an integrity audit. Failed/incomplete records remain visible with no complete-cycle peak estimate. This deterministic oracle-geometry development grid is neither sensor noise nor random-wall generalization. Squared torque is effort proxy, not energy.')
    write(out/'summary.json',report);(out/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    lines=['# 冻结杆位扰动三方案比较','',
        '三方案使用相同参考、策略、场景偏移和电机权限。所有 16 个工况保留；未重训、未自动替换 v2。','',
        '| 方案 | 完整物理/策略通过 | 分母 |','|---|---:|---:|']
    labels={'adopted_v2':'已采用 v2','contact_confirmed':'接触确认候选','timed_open_loop':'固定延时对照'}
    for arm in ARMS:lines.append(f'| {labels[arm]} | {counts[arm]} | 16 |')
    lines+=['','## 逐工况结果','','| 工况 | 原 v2 | 接触确认 | 固定延时 |','|---|---|---|---|']
    for c in cases:
        statuses=[('通过' if c['arms'][a]['passed'] else c['arms'][a]['status']) for a in ARMS]
        lines.append('| '+c['case']+' | '+' | '.join(statuses)+' |')
    lines+=['','## 解释边界','',
        '定时参数来自已观察的 T12。接触确认使用理想力反馈及接触物体真值身份；原全身反馈在三方案中均保留。',
        '接触确认方案可以因确认超时而停止，定时方案没有对应确认超时。失败原因必须与成功率一起报告。',
        '完整第二周期的峰值、均方根、力矩平方积分、闭合误差及每次加载延时见 summary.json。短记录不能说明完整周期峰值降低。',
        '本结果不证明新颖性、硬件安全性、实际传感器可部署性或随机墙面泛化。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(passed_counts=counts,records=48,output=str(out))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paired',type=Path,required=True)
    p.add_argument('--timed',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();summarize(a.paired.resolve(),a.timed.resolve(),a.output.resolve())
