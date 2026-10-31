"""Export independently audited full-cycle terminal metrics and traces."""
import argparse,json
from pathlib import Path
import numpy as np
from tools.report_selected_continuation import read_run
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from tools.continue_vector_full_cycle_training import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    m=json.loads((a.candidate/'manifest.json').read_text());parent=Path(m['parent_experiment']);a.output.mkdir(exist_ok=False)
    e=FullCycleResidualEnv(plan=m['reference'],cases=m['evaluation_cases']);e.reset(options={'case':'foot_z_plus15'});arrays={};rows={}
    try:
        for label,path in [('original',parent),('modified',a.candidate)]:
            metrics,_,_=read_run(path);rows[label]=metrics
            with np.load(path/'after_foot_z_plus15.npz') as z:
                t=z['time_s'];keep=t>=99.66;q=z['qpos']
                arrays[label+'_time_s']=t[keep]
                arrays[label+'_closure_mrad']=np.max(abs(q[keep][:,e.env.qadr]-e.env.initial[e.env.qadr]),axis=1)*1000
                arrays[label+'_rise_mm']=(q[keep,2]-e.env.initial[2])*1000
        for label in ('original','modified'):
            mask=arrays[label+'_time_s']>111.66
            assert abs(arrays[label+'_closure_mrad'][mask].max()-rows[label]['foot_z_plus15']['joint_closure_mrad'])<1e-8
            assert abs(arrays[label+'_rise_mm'][mask].max()-rows[label]['foot_z_plus15']['root_rise_max_mm'])<1e-8
        np.savez_compressed(a.output/'traces.npz',**arrays)
        s=dict(metrics=rows,original=str(parent),candidate=str(a.candidate),candidate_audit_sha256=sha(a.candidate/'independent_training_audit.json'),
            traces_sha256=sha(a.output/'traces.npz'),source_sha256=sha(__file__),scope='Audited original vs search bias plus added waist/body commands. Not a new PPO gain; narrow deterministic margins.')
        (a.output/'summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(rows),flush=True)
    finally:e.close()

if __name__=='__main__':main()
