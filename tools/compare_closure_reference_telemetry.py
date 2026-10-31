"""Compare independently verified 2000 Hz telemetry without rerunning dynamics."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--modified',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    arrays={};rows={};prefix={};traces={};manifests=[]
    for label,root in [('original',a.original),('modified',a.modified)]:
        audit=json.loads((root/'telemetry_audit.json').read_text());summary=json.loads((root/'summary.json').read_text())
        assert audit['passed'] and audit['summary_sha256']==sha(root/'summary.json') and summary['exact_reproduction_passed']
        manifests.append(json.loads((Path(summary['parent_experiment'])/'manifest.json').read_text()))
        row=audit['rows'][0];path=root/row['case']/'physics_telemetry.npz';assert sha(path)==row['trace_sha256']
        with np.load(path) as z:traces[label]={k:z[k].copy() for k in z.files}
        z=traces[label];t=z['time_s'];tail=t>111.66;norm=np.linalg.norm(z['solver_qacc'][:,:3],axis=1)
        rows[label]=dict(**row,final_root_acceleration_norm_peak_m_s2=float(norm[tail].max()),fullcycle_root_acceleration_norm_rms_m_s2=float(np.sqrt(np.mean(norm**2))))
        bins=len(t)//20;trim=bins*20
        arrays[label+'_time_s']=t[:trim].reshape(bins,20).mean(axis=1)
        for key,value in [('root_acceleration',norm),('support_vertical',z['group_force_world_n'][:,:,2])]:
            blocks=value[:trim].reshape((bins,20)+value.shape[1:])
            for stat,fn in [('mean',np.mean),('min',np.min),('max',np.max)]:arrays[label+'_'+key+'_'+stat]=fn(blocks,axis=1)
    assert manifests[0]['reference_sha256']==manifests[1]['reference_sha256']
    assert manifests[0]['controller_modifier']==manifests[1]['controller_modifier']
    assert manifests[0].get('adaptive_reference_command') is None and len(manifests[1]['adaptive_reference_command'])==5
    assert sha(Path(json.loads((a.original/'summary.json').read_text())['parent_experiment'])/'policy.zip')==sha(Path(json.loads((a.modified/'summary.json').read_text())['parent_experiment'])/'policy.zip')
    for key in traces['original']:
        mask=traces['original']['time_s']<=99.66000001
        prefix[key]=bool(np.array_equal(traces['original'][key][mask],traces['modified'][key][mask]))
    assert all(prefix.values())
    a.output.mkdir(exist_ok=False);np.savez_compressed(a.output/'plot_data.npz',**arrays)
    result=dict(labels={'original':'Search seed','modified':'Added fixed references'},rows=rows,prefix_exact=prefix,plot_data_sha256=sha(a.output/'plot_data.npz'),source_sha256=sha(__file__),
        scope='Same actor with changed fixed reference authority; no new PPO learning. 2000 Hz source telemetry; 100 Hz block envelopes for plotting only. Root acceleration is not COM acceleration; one deterministic known case, no hardware or robustness claim.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
