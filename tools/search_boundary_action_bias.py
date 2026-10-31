"""Fixed action-bias feasibility search inside existing policy action bounds."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from tools.compare_audited_closure_candidates import load_run, sha
from tools.evaluate_closure_action_bias import run_case_in_fresh_process
from training.full_cycle_residual_env import FullCycleResidualEnv


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--probe-source',type=Path,default=None)
    p.add_argument('--radius',type=float,choices=[.1,.2],default=.2)
    p.add_argument('--start',type=float,choices=[70.2,99.66],default=70.2)
    a=p.parse_args(); parent=load_run(a.parent)
    manifest=json.loads((a.parent/'manifest.json').read_text())
    cases=['boundary_04','boundary_05','boundary_06','boundary_07']
    assert set(cases)<=set(parent['cases'])
    center=np.zeros(2)
    lower=np.maximum(-a.radius,-.2-center);upper=np.minimum(a.radius,.2-center)
    probes=[];axes=[]
    for axis in range(2):
        labels=[];offsets=[]
        for sign in [-1,1]:
            step=sign*.05
            if not lower[axis]<=step<=upper[axis]:
                labels.append('base');offsets.append(0.);continue
            delta=np.zeros(2);delta[axis]=step
            label=f'axis{axis}_{"minus" if sign<0 else "plus"}'
            probes.append((label,delta.tolist()));labels.append(label);offsets.append(step)
        assert offsets[0]!=offsets[1]
        axes.append(dict(labels=labels,offsets=offsets))
    sources={str(Path(__file__).relative_to(Path.cwd())):sha(__file__),
             'tools/evaluate_closure_action_bias.py':sha('tools/evaluate_closure_action_bias.py')}
    sources['training/closure_action_bias_policy.py']=sha('training/closure_action_bias_policy.py')
    sources.update({k:v for k,v in manifest['sources'].items() if k.replace('\\','/').startswith('training/')})
    for f,digest in sources.items():assert sha(f)==digest,f
    protocol=dict(parent=str(a.parent.resolve()),parent_audit_sha256=parent['audit_sha256'],
        checkpoint_sha256=parent['checkpoint_sha256'],cases=cases,center=center.tolist(),
        delta_bounds=np.c_[lower,upper].tolist(),probes=probes,axes=axes,sources=sources,
        proposal_fractions=[.25,.5,1.],radius=a.radius,start_seconds=a.start,kind='action_bias',
        probe_source=str(a.probe_source.resolve()) if a.probe_source else None,
        scope='Fixed left-knee/ankle action offsets applied with a smooth time and existing high-rod geometry gate. '
              'Same actor checkpoint, five constant reference commands, 26D action bounds, native physical limits and acceptance. '
              'Mandatory full zero-adapter replay; all probes start at time zero. This is policy calibration, not PPO learning or generalization.')
    protocol=json.loads(json.dumps(protocol))
    if a.probe_source:
        prior=json.loads((a.probe_source/'protocol.json').read_text())
        assert prior['parent_audit_sha256']==protocol['parent_audit_sha256']
        assert prior['kind']=='action_bias' and prior['start_seconds']==a.start
        assert prior['center']==protocol['center'] and prior['cases']==protocol['cases'] and prior['probes']==protocol['probes']
        for f,digest in prior['sources'].items():assert sha(f)==digest
        protocol['probe_source_protocol_sha256']=sha(a.probe_source/'protocol.json')
    if a.resume:
        old=json.loads((a.output/'protocol.json').read_text());assert old==protocol
        final=a.output/'summary.json'
        if final.exists():
            existing=json.loads(final.read_text())
            assert existing['completed'] and existing['protocol_sha256']==sha(a.output/'protocol.json')
            print(json.dumps(dict(completed=True,best=existing['best'],already_complete=True)),flush=True)
            return
        (a.output/'STOP_REQUESTED').unlink(missing_ok=True)
    else:
        a.output.mkdir(exist_ok=False)
        (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
        (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    env=FullCycleResidualEnv(plan=manifest['reference'],cases=manifest['evaluation_cases'])
    qadr=env.env.qadr.copy();initial=env.plan['qpos'][0].copy();env.close()
    rows=[];vectors={};bounds=None
    def inspect(label,case,path,info):
        nonlocal bounds
        with np.load(path) as z,np.load(a.parent/f'after_{case}.npz') as old:
            if label=='zero_gate':
                assert all(np.array_equal(z[key],old[key]) for key in old.files)
                assert info['success']==parent['rows'][case]['success']
            for key in ['qpos','qvel','time_s','observations','actions','torques','rewards']:
                count=round(a.start/.02)+(1 if key in ['qpos','qvel','time_s'] else 0)
                assert np.array_equal(z[key][:count],old[key][:count]),(label,case,key)
            complete=abs(float(z['time_s'][-1])-113.66)<1e-6
            if complete:
                tail=(z['time_s']>111.66)&(z['time_s']<=113.6600001)
                joint=z['qpos'][tail][:,qadr]-initial[qadr]
                height=z['qpos'][tail,2]-initial[2]-.25
                signed=np.r_[joint.ravel()/.001,-joint.ravel()/.001,height/.0001,-height/.0001]
                limits=np.r_[np.full(2*joint.size,40.),np.full(2*height.size,100.)]
                if bounds is None:bounds=limits
                assert np.array_equal(bounds,limits)
                vectors[label,case]=signed
                margin=float((limits-signed).min())
                closure=float(abs(joint).max()*1000);rise=float((height+.25).max()*1000)
            else: margin=-1e9;closure=None;rise=None
        row=dict(label=label,case=case,complete=complete,success=bool(info['success']),
                 minimum_margin=margin,closure_mrad=closure,rise_max_mm=rise,
                 file=str(path.resolve()),trajectory_sha256=sha(path))
        rows.append(row)
        (a.output/'progress.json').write_text(json.dumps(dict(completed=False,rows=rows),indent=2))
        return row
    parent_rows={r['case']:r for r in json.loads((a.parent/'summary.json').read_text())['after']}
    for case in cases:inspect('base',case,a.parent/f'after_{case}.npz',parent_rows[case])
    def batch(commands):
        for label,delta in commands:
            if (a.output/'STOP_REQUESTED').exists():return False
            raw=center+np.asarray(delta);assert np.all(abs(raw)<=.2+1e-12)
            command=np.clip(raw,-.2,.2)
            directory=a.output/label;directory.mkdir(exist_ok=True)
            m=copy.deepcopy(manifest)
            bias=np.zeros(26);bias[3:5]=command
            m['policy_adapter']='closure-action-bias-v1'
            m['closure_action_bias']=dict(normalized_bias=bias.tolist(),start_seconds=a.start,ramp_seconds=3.,geometry_gate='existing_high_rod')
            def job(case):
                destination=directory/case
                if a.probe_source and label.startswith('axis'):
                    destination=a.probe_source/label/case
                    assert (destination/'result.json').exists(), 'Only completed probes may be reused'
                result_file=destination/'result.json'
                if result_file.exists():
                    result=json.loads(result_file.read_text())
                    assert sha(destination/f'after_{case}.npz')==result['trajectory_sha256']
                    assert result['closure_action_bias']==m['closure_action_bias']
                else:
                    result=run_case_in_fresh_process((m,str((a.parent/'policy.zip').resolve()),parent['checkpoint_sha256'],case,str(destination.resolve())))
                return case,result,destination/f'after_{case}.npz'
            with ThreadPoolExecutor(max_workers=4) as pool:
                jobs=[pool.submit(job,case) for case in cases]
                for future in as_completed(jobs):
                    case,result,path=future.result();row=inspect(label,case,path,result)
                    print(json.dumps(row),flush=True)
        return True
    if not batch([('zero_gate',[0.,0.])]):return
    if not batch(probes):return
    proposals={};commands=[]
    for name,selected in [('common',cases)]+[(case,[case]) for case in cases]:
        if not all((label,case) in vectors for label in ['base']+[r[0] for r in probes] for case in selected):
            proposals[name]=dict(success=False,reason='Incomplete probe trajectory');continue
        base=np.concatenate([vectors['base',case] for case in selected]);limit=np.tile(bounds,len(selected))
        jacobian=np.stack([(np.concatenate([vectors[axis['labels'][1],case] for case in selected])-
                           np.concatenate([vectors[axis['labels'][0],case] for case in selected]))/
                          (axis['offsets'][1]-axis['offsets'][0]) for axis in axes],axis=1)
        fit=linprog([0,0,-1],A_ub=np.c_[jacobian,np.ones(len(base))],b_ub=limit-base,
                    bounds=protocol['delta_bounds']+[(None,None)],method='highs')
        proposals[name]=dict(success=bool(fit.success),message=fit.message)
        if fit.success:
            proposals[name].update(delta=fit.x[:2].tolist(),predicted_minimum_margin=float(fit.x[2]))
            if name=='common':commands=[(f'common_{fraction:g}',(fraction*fit.x[:2]).tolist()) for fraction in protocol['proposal_fractions']]
    (a.output/'proposals.json').write_text(json.dumps(proposals,indent=2))
    if not batch(commands):return
    groups=[]
    for label,delta in [('base',[0.]*2),('zero_gate',[0.]*2)]+probes+commands:
        group=[r for r in rows if r['label']==label]
        assert len(group)==4
        groups.append(dict(label=label,delta=delta,command=(center+delta).tolist(),
            successes=sum(r['success'] for r in group),minimum_margin=min(r['minimum_margin'] for r in group),
            neighbor_preserved=next(r['success'] for r in group if r['case']=='boundary_04')))
    best=max(groups,key=lambda r:(r['neighbor_preserved'],r['successes'],r['minimum_margin']))
    for f,digest in sources.items():assert sha(f)==digest
    result=dict(completed=True,rows=rows,groups=groups,best=best,proposals=proposals,
                protocol_sha256=sha(a.output/'protocol.json'),scope=protocol['scope'])
    (a.output/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(completed=True,best=best)),flush=True)


if __name__=='__main__':main()
