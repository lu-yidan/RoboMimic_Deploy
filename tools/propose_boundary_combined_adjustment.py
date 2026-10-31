"""Local, unvalidated proposals combining measured reference and action effects."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from tools.compare_audited_closure_candidates import load_run,sha
from training.full_cycle_residual_env import FullCycleResidualEnv


def main():
    p=argparse.ArgumentParser()
    for name in ('reference-search','action-search','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    rp=json.loads((a.reference_search/'protocol.json').read_text())
    ap=json.loads((a.action_search/'protocol.json').read_text())
    rs=json.loads((a.reference_search/'progress.json').read_text())
    ac=json.loads((a.action_search/'summary.json').read_text())
    assert ac['completed'] and ac['protocol_sha256']==sha(a.action_search/'protocol.json')
    for key in ('parent','parent_audit_sha256','checkpoint_sha256','cases'):assert rp[key]==ap[key]
    for protocol in (rp,ap):
        for f,h in protocol['sources'].items():assert sha(f)==h
    parent=load_run(Path(rp['parent']));assert parent['audit_sha256']==rp['parent_audit_sha256']
    m=json.loads((Path(rp['parent'])/'manifest.json').read_text())
    env=FullCycleResidualEnv(plan=m['reference'],cases=m['evaluation_cases'])
    qadr=env.env.qadr.copy();initial=env.plan['qpos'][0].copy();env.close()
    bounds=None
    def vector(row):
        nonlocal bounds
        assert sha(row['file'])==row['trajectory_sha256'] and row['complete']
        with np.load(row['file']) as z:
            tail=(z['time_s']>111.66)&(z['time_s']<=113.6600001)
            joint=z['qpos'][tail][:,qadr]-initial[qadr]
            height=z['qpos'][tail,2]-initial[2]-.25
        limits=np.r_[np.full(2*joint.size,40.),np.full(2*height.size,100.)]
        if bounds is None:bounds=limits
        assert np.array_equal(bounds,limits)
        return np.r_[joint.ravel()/.001,-joint.ravel()/.001,height/.0001,-height/.0001]
    refs={(r['label'],r['case']):vector(r) for r in rs['rows']}
    acts={(r['label'],r['case']):vector(r) for r in ac['rows']}
    center=np.asarray(rp['center']);cases=rp['cases'];models=[]
    assert next(g['command'] for g in ac['groups'] if g['label']=='common_1')==[-.2,-.2]
    assert next(g['command'] for g in ac['groups'] if g['label']=='common_0.5')==[-.1,-.1]
    for cap in (.2,.3,.4,.5):
        for name,selected in [('common',cases),('boundary_07',['boundary_07'])]:
            base=np.concatenate([acts['common_1',case] for case in selected])
            jac=[]
            for axis in rp['axes']:
                jac.append(np.concatenate([(refs[axis['labels'][1],c]-refs[axis['labels'][0],c])/(axis['offsets'][1]-axis['offsets'][0]) for c in selected]))
            # Increasing scalar means a more negative, tied knee/ankle offset.
            jac.append(np.concatenate([(acts['common_1',c]-acts['common_0.5',c])/.1 for c in selected]))
            jac=np.stack(jac,axis=1)
            limits=np.tile(bounds,len(selected))
            delta_bounds=np.c_[-1-center,1-center].tolist()+[[0.,cap-.2]]
            fit=linprog([0.]*6+[-1.],A_ub=np.c_[jac,np.ones(len(base))],b_ub=limits-base,bounds=delta_bounds+[(None,None)],method='highs')
            row=dict(name=name,action_bias_cap=cap,optimization_success=bool(fit.success))
            if fit.success:row.update(reference_command=np.clip(center+fit.x[:5],-1,1).tolist(),tied_action_bias=float(-.2-fit.x[5]),predicted_minimum_margin=float(fit.x[6]))
            models.append(row)
    result=dict(physically_validated=False,models=models,parent=rp['parent'],parent_audit_sha256=rp['parent_audit_sha256'],
        reference_protocol_sha256=sha(a.reference_search/'protocol.json'),action_protocol_sha256=sha(a.action_search/'protocol.json'),
        reference_progress_sha256=sha(a.reference_search/'progress.json'),action_summary_sha256=sha(a.action_search/'summary.json'),source_sha256=sha(__file__),
        scope='Local additive approximation anchored at the measured -0.2/-0.2 action candidate. Reference derivatives come from the original parent. Interactions and extrapolation are unvalidated. Increasing calibration-search radius would not change final action bounds, but requires a separately declared adapter and physical evaluation.')
    a.output.mkdir(exist_ok=False);(a.output/'proposals.json').write_text(json.dumps(result,indent=2));(a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(models,indent=2),flush=True)


if __name__=='__main__':main()
