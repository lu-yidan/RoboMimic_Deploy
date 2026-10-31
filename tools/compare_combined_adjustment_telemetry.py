"""Compare independently verified 2000 Hz telemetry without rerunning dynamics."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--modified',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    arrays={};rows={};prefix={};traces={};manifests=[];checkpoints=[]
    for label,root in [('original',a.original),('modified',a.modified)]:
        audit=json.loads((root/'telemetry_audit.json').read_text());summary=json.loads((root/'summary.json').read_text())
        assert audit['passed'] and audit['summary_sha256']==sha(root/'summary.json') and summary['exact_reproduction_passed']
        checkpoints.append(summary['checkpoint_sha256'])
        manifests.append(json.loads((Path(summary['parent_experiment'])/'manifest.json').read_text()))
        assert len(audit['rows'])==1
        row=audit['rows'][0];path=root/row['case']/'physics_telemetry.npz';assert sha(path)==row['trace_sha256']
        with np.load(path) as z:traces[label]={k:z[k].copy() for k in z.files}
        z=traces[label];t=z['time_s'];tail=t>111.66;norm=np.linalg.norm(z['solver_qacc'][:,:3],axis=1)
        rows[label]=dict(**row,final_root_acceleration_norm_peak_m_s2=float(norm[tail].max()),fullcycle_root_acceleration_norm_rms_m_s2=float(np.sqrt(np.mean(norm**2))))
        bins=len(t)//20;trim=bins*20
        arrays[label+'_time_s']=t[:trim].reshape(bins,20).mean(axis=1)
        for key,value in [('root_acceleration',norm),('support_vertical',z['group_force_world_n'][:,:,2])]:
            blocks=value[:trim].reshape((bins,20)+value.shape[1:])
            for stat,fn in [('mean',np.mean),('min',np.min),('max',np.max)]:arrays[label+'_'+key+'_'+stat]=fn(blocks,axis=1)
    assert rows['original']['case']==rows['modified']['case']
    assert manifests[1]['suffix_start_seconds']==70.2
    assert manifests[0]['prefix_checkpoint_sha256']==manifests[1]['prefix_checkpoint_sha256']
    assert manifests[0]['reference_sha256']==manifests[1]['reference_sha256']
    assert manifests[0]['controller_modifier']==manifests[1]['controller_modifier']
    assert len(set(checkpoints))==1
    assert manifests[1].get('policy_adapter')=='closure-action-bias-v2'
    for key in traces['original']:
        mask=traces['original']['time_s']<=70.20000001
        prefix[key]=bool(np.array_equal(traces['original'][key][mask],traces['modified'][key][mask]))
    assert all(prefix.values())
    a.output.mkdir(exist_ok=False);np.savez_compressed(a.output/'plot_data.npz',**arrays)
    result=dict(labels={'original':'PPO parent','modified':'Joint/reference calibration'},reference_commands=[m.get('adaptive_reference_command') for m in manifests],action_biases=[m.get('closure_action_bias') for m in manifests],rows=rows,prefix_exact=prefix,plot_data_sha256=sha(a.output/'plot_data.npz'),source_sha256=sha(__file__),
        scope='Same actor checkpoint and original physical/action limits; joint/reference calibration versus parent, no new PPO learning; exact unchanged prefix through 70.2 seconds. 2000 Hz source telemetry; 100 Hz block envelopes for plotting only. Root acceleration is not COM acceleration; one deterministic known case, no hardware or robustness claim.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
