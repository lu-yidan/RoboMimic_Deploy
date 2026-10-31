"""Read-only aggregation of audited continuous climbing records.

Endpoint support telemetry is 50 Hz over the last two seconds of each cycle.
Acceleration is sampled at 50 Hz throughout; it does not bound impact peaks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_suite_member(run, case, cycles, protocol, parent_manifest):
    """Bind each result to the declared frozen arm, including reused records."""
    manifest=read(run/'manifest.json')
    assert manifest['case']==case and manifest['cycles']==cycles
    assert Path(manifest['parent_experiment']).resolve()==Path(protocol['candidate']).resolve()
    assert manifest['parent_manifest_sha256']==protocol['parent_manifest_sha256']
    assert manifest['checkpoint_sha256']==protocol['checkpoint_sha256']
    assert manifest['offsets']==parent_manifest['evaluation_cases'][case]
    assert manifest['plan_sha256']==protocol['two_plan_sha256' if cycles==2 else 'eight_plan_sha256']
    assert sha(run/'reference.npz')==manifest['plan_sha256']
    assert manifest['new_training_steps']==0 and not manifest.get('contact_admission')
    assert not manifest.get('timed_loading') and not manifest.get('scene_compiler')


def summarize(run):
    m=read(run/'manifest.json');s=read(run/'summary.json')
    a=read(run/'v2_multicycle_audit.json');physical=read(run/'independent_audit.json')
    assert a['manifest_sha256']==sha(run/'manifest.json')
    assert a['summary_sha256']==sha(run/'summary.json')
    assert a['trajectory_sha256']==s['trajectory_sha256']==sha(run/'trajectory.npz')
    assert a['independent_audit_sha256']==sha(run/'independent_audit.json')
    assert a['policy_audit_sha256']==sha(run/'policy_transfer_audit.json')
    assert s['endpoint_samples_sha256']==sha(run/'endpoint_samples.json')
    z=np.load(run/'trajectory.npz');ref=np.load(run/'reference.npz')
    tail=read(run/'endpoint_samples.json');previous=0.;rows=[]
    for i,result in enumerate(s['cycle_results']):
        end=float(ref['cycle_ends_s'][i]);start=0. if i==0 else float(ref['cycle_ends_s'][i-1])
        index=int(np.argmin(abs(z['time_s']-end)))
        assert abs(z['time_s'][index]-end)<1e-7
        rise=float(z['qpos'][index,2]-z['qpos'][0,2])
        samples=[r for r in tail if r['cycle']==i+1]
        selection=(z['time_s'][1:]>start+1e-7)&(z['time_s'][1:]<=end+1e-7)
        acceleration=z['qacc_50hz'][selection]
        trans=np.linalg.norm(acceleration[:,:3],axis=1)
        row=dict(cycle=i+1,passed=result['passed'],closure_mrad=result['closure_mrad'],
            closure_warning=result['closure_mrad']>35,cumulative_rise_mm=rise*1000,
            cumulative_error_mm=(rise-.25*(i+1))*1000,actual_increment_mm=(rise-previous)*1000,
            min_normal_n=min(min(r['normal_n'].values()) for r in samples),
            min_vertical_n=min(min(r['vertical_n'].values()) for r in samples),
            peak_sampled_base_acceleration_m_s2=float(trans.max()),
            p99_sampled_base_acceleration_m_s2=float(np.percentile(trans,99)),
            peak_sampled_joint_acceleration_rad_s2=float(abs(acceleration[:,6:]).max()),
            endpoint_effector_error_mm=physical['cycle_endpoints'][i]['independent_endpoint']['effector_error_m']*1000,
            endpoint_linear_speed_m_s=physical['cycle_endpoints'][i]['independent_endpoint']['linear_speed_m_s'])
        rows.append(row);previous=rise
    return dict(case=m['case'],passed=a['passed'],completed_cycles=len(rows),declared_cycles=m['cycles'],
        actual_horizon_s=float(z['time_s'][-1]),final_rise_m=s['final_rise_m'],
        minimum_native_joint_margin_rad=s['minimum_native_joint_margin_rad'],
        runtime_peaks=s['peaks'],physical_reset_count=s['physical_reset_count'],
        first_cycle_prefix_exact=s['first_cycle_prefix_exact'],cycle_metrics=rows,
        run=str(run.resolve()),audit_sha256=sha(run/'v2_multicycle_audit.json'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--no-plot',action='store_true');a=p.parse_args()
    if not a.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    protocol=read(a.suite/'protocol.json');cases=protocol['cases']
    if (a.suite/'runner_source.py').exists():
        result=read(a.suite/'suite_summary.json')
        assert result['protocol_sha256']==sha(a.suite/'protocol.json')
        assert sha(a.suite/'runner_source.py')==protocol['source_sha256']
        parent=Path(protocol['candidate']);parent_manifest=read(parent/'manifest.json')
        assert sha(parent/'manifest.json')==protocol['parent_manifest_sha256']
        assert sha(parent/'policy.zip')==protocol['checkpoint_sha256']
        assert len(cases)==len(set(cases))==result['declared_cases']
        assert result['passed_cases']==sum(r['passed'] for r in result['results'])
        assert result['passed']==all(r['passed'] for r in result['results'])
        assert [r['case'] for r in result['results']]==cases
        rows=[]
        for item in result['results']:
            if item.get('two_run'):
                two=Path(item['two_run'])
                assert sha(two/'v2_multicycle_audit.json')==item['two_audit_sha256']
                verify_suite_member(two,item['case'],2,protocol,parent_manifest)
            if item.get('eight_run'):
                run=Path(item['eight_run']);assert sha(run/'v2_multicycle_audit.json')==item['eight_audit_sha256']
                verify_suite_member(run,item['case'],8,protocol,parent_manifest)
                assert read(run/'manifest.json')['two_cycle_audit_sha256']==item['two_audit_sha256']
                row=summarize(run);assert row['case']==item['case'] and row['declared_cycles']==8
                assert row['passed']==item['passed']
                rows.append(row)
            else:
                rows.append(dict(case=item['case'],passed=False,declared_cycles=8,completed_cycles=0,
                    cycle_metrics=[],reason=item['reason'],screening_result=item))
    else:rows=[summarize(a.suite/c) for c in cases]
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(passed=all(r['passed'] for r in rows),cases=len(rows),passed_cases=sum(r['passed'] for r in rows),
        new_training_steps=0,rows=rows,protocol_sha256=sha(a.suite/'protocol.json'),source_sha256=sha(__file__),
        scope='Development cases only. Cumulative height at each endpoint is checked relative to the initial run pose; actual individual increments are reported separately. Closure is quality-only. Normal/vertical support is endpoint-tail telemetry, not whole-run force maxima. Acceleration sampled at 50 Hz can miss short impacts; no impact safety claim. No independent full-dynamics resimulation.')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    (a.output/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    if a.no_plot:
        print(json.dumps(dict(passed=report['passed'],cases=len(rows),output=str(a.output))),flush=True)
        return
    fig,axes=plt.subplots(3,2,figsize=(12,11),layout='constrained')
    fields=[('cumulative_rise_mm','Cumulative root rise (mm)'),('actual_increment_mm','Actual rise since previous endpoint (mm)'),
            ('closure_mrad','Maximum endpoint joint closure (mrad)'),('min_normal_n','Minimum endpoint support normal force (N)'),
            ('min_vertical_n','Minimum endpoint vertical support (N)'),('peak_sampled_base_acceleration_m_s2','Peak base acceleration sampled at 50 Hz (m/s²)')]
    many=len(rows)>8
    for row in rows:
        x=[r['cycle'] for r in row['cycle_metrics']]
        for ax,(field,label) in zip(axes.flat,fields):
            style=dict(marker='o',markersize=3,label=row['case'])
            if many:
                style=dict(color='gray',alpha=.2,linewidth=.7)
                if row['case'] in ('training_00','training_12'):
                    style=dict(label=row['case'],color='tab:orange' if row['case']=='training_00' else 'tab:green',linewidth=1.7)
            ax.plot(x,[r[field] for r in row['cycle_metrics']],**style)
            ax.set(xlabel='Completed cycle',ylabel=label);ax.grid(alpha=.2)
    x=np.arange(1,max(r['declared_cycles'] for r in rows)+1)
    if many:
        complete=[r for r in rows if len(r['cycle_metrics'])==len(x)]
        for ax,(field,_) in zip(axes.flat,fields):
            if complete:
                values=np.array([[v[field] for v in r['cycle_metrics']] for r in complete])
                ax.fill_between(x,values.min(axis=0),values.max(axis=0),alpha=.12,color='tab:blue',label='Completed-case range')
                ax.plot(x,np.median(values,axis=0),color='tab:blue',linewidth=2,label='Completed-case median')
    axes[0,0].plot(x,250*x,'k--',alpha=.5,label='Nominal cumulative rise')
    axes[0,1].axhline(250,color='k',linestyle='--',alpha=.5)
    axes[1,0].axhline(35,color='darkorange',linestyle='--',label='Quality warning (not failure)')
    for ax in (axes[1,1],axes[2,0]):ax.axhline(10,color='red',linestyle='--',alpha=.5)
    axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
    fig.suptitle(f'Lifted reference + frozen PPO: {report["passed_cases"]}/{len(rows)} audited development cases\nNo retraining; privileged geometry; sampled acceleration is not an impact bound',fontsize=12)
    fig.savefig(a.output/'cycle_metrics.png',dpi=160);fig.savefig(a.output/'cycle_metrics.pdf');plt.close(fig)
    print(json.dumps(dict(passed=report['passed'],cases=len(rows),output=str(a.output))),flush=True)


if __name__=='__main__':main()
