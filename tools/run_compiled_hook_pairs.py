"""Keep the complete frozen grid denominator while evaluating both controllers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from tools.validate_swing_reference import ROOT,read,write,sha

ARMS=('adopted_v2','contact_confirmed')
SOURCES=('tools/run_compiled_hook_pairs.py','tools/evaluate_compiled_hook_pair.py',
         'tools/audit_compiled_hook_pair.py','training/compiled_rung_geometry.py',
         'training/probe_hook_loading.py','training/physics_telemetry.py')


def prepare(single,plan,out):
    single=single.resolve();plan=plan.resolve();m=read(single/'manifest.json')
    protocol=read(single/'hook_uncertainty_protocol.json');pm=read(plan.parent/'manifest.json')
    assert m['scene_compiler']=='complete_mjspec_model_at_reset_v1'
    assert protocol['cases']==m['evaluation_cases'] and len(protocol['cases'])==16
    assert sha(single/'hook_uncertainty_protocol.json')==m['hook_protocol_sha256']
    assert pm['plan_sha256']==sha(plan) and pm['first_reference_sha256']==m['reference_sha256']
    assert read(plan.parent/'audit.json')['passed']
    with np.load(plan) as z:assert len(z['cycle_ends_s'])==2
    out.mkdir(parents=True,exist_ok=False)
    sources={}
    for relative in SOURCES:
        target=out/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target);sources[relative]=sha(target)
    write(out/'protocol.json',dict(single_cycle_screen=str(single),single_manifest_sha256=sha(single/'manifest.json'),
        hook_protocol_sha256=m['hook_protocol_sha256'],cases=protocol['cases'],arms=list(ARMS),
        plan=str(plan),plan_sha256=sha(plan),controller_parameters=protocol['controller_parameters'],sources=sources,
        selection='Run both arms after the SAME case passes the compiled single-cycle and prefix gates. Never require baseline two-cycle success before starting the candidate. Count a failed first cycle against both arms in the full 16-case denominator.',
        interrupted_records='Keep controller errors, timeout records, all partial physics telemetry and process failures. They cannot count as successful or audited complete rollouts.',
        scope='Frozen local development grid with oracle rod geometry. No retuning, retry selection, new training, automatic promotion, or unseen/random-wall generalization claim. Feedback-versus-timed-ramp ablation remains missing.'))
    print(json.dumps(dict(prepared=True,cases=16,arms=2,output=str(out))),flush=True)


def eligible(single,case):
    chain=read(single/'reference_chain_audit.json');compiled=read(single/'compiled_reference_chain_audit.json')
    assert chain['passed'] and compiled['passed']
    assert chain['manifest_sha256']==compiled['manifest_sha256']==sha(single/'manifest.json')
    assert compiled['base_chain_audit_sha256']==sha(single/'reference_chain_audit.json')
    physical=next(r for r in chain['rows'] if r['case']==case)
    prefix=next(r for r in chain['prefix_rows'] if r['case']==case)
    scene=next(r for r in compiled['rows'] if r['case']==case)
    return bool(physical['physical_v2_passed'] and prefix['passed'] and scene['passed'])


def outcome(target,case,arm,code,log):
    row=dict(case=case,arm=arm,process_returncode=code,log=str(log),run=str(target),passed=False,
             status='execution_or_audit_error',complete=False)
    if (target/'interrupted.json').exists():
        row.update(status='interrupted',interruption=read(target/'interrupted.json'),
                   interrupted_record_sha256=sha(target/'interrupted.json'))
    if (target/'summary.json').exists():
        s=read(target/'summary.json')
        row.update(reason=s['reason'],time_s=s['time_s'],complete=abs(s['time_s']-227.32)<1e-7,
                   trajectory_sha256=sha(target/'trajectory.npz'))
    if (target/'compiled_pair_audit.json').exists():
        audit=read(target/'compiled_pair_audit.json')
        assert audit['case']==case and audit['arm']==arm
        assert audit['trajectory_sha256']==sha(target/'trajectory.npz')
        assert audit['manifest_sha256']==sha(target/'manifest.json')
        row.update(passed=bool(code==0 and audit['passed']),integrity_passed=audit['integrity_passed'],
            status='audited_success' if code==0 and audit['passed'] else 'audited_failure',
            audit_sha256=sha(target/'compiled_pair_audit.json'))
    if (target/'physics_telemetry_record.json').exists():
        record=read(target/'physics_telemetry_record.json')
        assert record['telemetry_sha256']==sha(target/'physics_telemetry.npz')
        row['telemetry_sha256']=record['telemetry_sha256']
        with np.load(target/'physics_telemetry.npz') as z:
            later=z['time_s']>113.66+1e-7
            row['later_recorded_samples']=int(later.sum())
            row['recorded_segment_peaks']=None if not later.any() else dict(
                root_acceleration_m_s2=float(np.linalg.norm(z['solver_qacc'][later,:3],axis=1).max()),
                joint_acceleration_rad_s2=float(abs(z['solver_qacc'][later,6:]).max()),
                support_net_force_n={g:float(np.linalg.norm(z['group_force_world_n'][later,j],axis=1).max())
                    for j,g in enumerate(record['groups'])})
        row['peak_interpretation']='Only recorded time span; failed/truncated runs cannot demonstrate a lower complete-cycle peak.'
    return row


def run(out,workers):
    protocol=read(out/'protocol.json');single=Path(protocol['single_cycle_screen'])
    assert sha(single/'manifest.json')==protocol['single_manifest_sha256']
    assert sha(single/'hook_uncertainty_protocol.json')==protocol['hook_protocol_sha256']
    for relative,digest in protocol['sources'].items():
        assert sha(ROOT/relative)==digest==sha(out/'source_snapshot'/relative)
    assert sha(protocol['plan'])==protocol['plan_sha256']
    assert not (out/'runs').exists() and not (out/'suite_summary.json').exists()
    (out/'runs').mkdir();(out/'logs').mkdir();(out/'results').mkdir()
    rows=[];jobs=[]
    for case in protocol['cases']:
        for arm in ARMS:
            if eligible(single,case):jobs.append((case,arm))
            else:
                row=dict(case=case,arm=arm,passed=False,status='first_cycle_gate_failed',complete=False)
                rows.append(row);write(out/'results'/f'{case}_{arm}.json',row)
    def one(case,arm):
        target=out/'runs'/f'{case}_{arm}';log=out/'logs'/f'{case}_{arm}.log'
        with log.open('w',encoding='utf8') as stream:
            result=subprocess.run([sys.executable,'-m','tools.evaluate_compiled_hook_pair',
                '--candidate',str(single),'--plan',protocol['plan'],'--case',case,'--arm',arm,
                '--output',str(target)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        try:row=outcome(target,case,arm,result.returncode,log)
        except Exception as exc:
            row=dict(case=case,arm=arm,passed=False,complete=False,status='result_integrity_error',
                     error=repr(exc),process_returncode=result.returncode,log=str(log),run=str(target))
        write(out/'results'/f'{case}_{arm}.json',row)
        print(json.dumps(dict(case=case,arm=arm,status=row['status'],passed=row['passed'])),flush=True)
        return row
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one,*job) for job in jobs]):rows.append(future.result())
    assert len(rows)==32 and len({(r['case'],r['arm']) for r in rows})==32
    rows.sort(key=lambda r:(r['case'],r['arm']))
    counts={arm:sum(r['passed'] for r in rows if r['arm']==arm) for arm in ARMS}
    comparison=[]
    for case in protocol['cases']:
        pair={r['arm']:r for r in rows if r['case']==case}
        comparison.append(dict(case=case,baseline_passed=pair[ARMS[0]]['passed'],candidate_passed=pair[ARMS[1]]['passed'],
            both_complete=all(r['complete'] for r in pair.values())))
    write(out/'suite_summary.json',dict(finished=True,declared_cases=16,declared_runs=32,passed_counts=counts,
        all_records_successful=all(r['passed'] for r in rows),rows=rows,paired_outcomes=comparison,
        protocol_sha256=sha(out/'protocol.json'),single_chain_sha256=sha(single/'compiled_reference_chain_audit.json'),
        result_sha256={p.name:sha(p) for p in sorted((out/'results').glob('*.json'))},
        sources_unchanged=all(sha(ROOT/k)==v for k,v in protocol['sources'].items()),
        new_training_steps=0,automatic_promotion=False,
        scope=protocol['scope']))
    print(json.dumps(dict(finished=True,cases=16,passed_counts=counts)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','run'))
    p.add_argument('--single',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=(1,2,4),default=4);a=p.parse_args()
    if a.command=='prepare':prepare(a.single,a.plan,a.output.resolve())
    else:run(a.output.resolve(),a.workers)
