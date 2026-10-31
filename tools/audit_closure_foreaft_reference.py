"""Independently reconstruct recorded added commands and total reference bounds."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.continue_vector_full_cycle_training import sha


def audit(directory):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text());summary=json.loads((directory/'summary.json').read_text())
    assert not summary['evaluation_pending']
    command=np.asarray(manifest['adaptive_reference_command'],dtype=float)
    assert command.shape==(5,) and np.isfinite(command).all() and abs(command).max()<=1.
    scales=np.array([.003,.003,.003,.00015,.001]);rows=[]
    assert manifest['adaptive_reference_scales']==scales.tolist()
    for row in summary['after']:
        path=directory/f"after_{row['case']}.npz";assert sha(path)==row['trajectory_sha256']
        with np.load(path) as data:
            trace=data['adaptive_reference_trace'];fixed_waist=data['waist_reference_trace'];times=data['time_s']
            geometry=np.float32(manifest['evaluation_cases'][row['case']].get('rung_01',[0,0,0])[2]/.02)
            gain=float(np.clip((float(geometry)-.625)/.125,0.,1.))
            assert trace.ndim==2 and trace.shape[1]==21 and len(trace)>0
            u=np.clip((trace[:,0]-99.66)/3.,0.,1.);ramp=u*u*(3-2*u)
            expected_command=np.broadcast_to(command,(len(trace),5));target=expected_command*scales*gain*ramp[:,None]
            residual=np.zeros_like(target);last=np.zeros(5)
            for i,value in enumerate(target):last=last+.05*(value-last);residual[i]=last
            velocity=np.diff(np.vstack([np.zeros(5),residual]),axis=0)/.005
            waist=np.zeros((len(trace),3))
            if len(fixed_waist):
                indices=np.rint(fixed_waist[:,0]/.005).astype(int)
                assert np.all(indices>=0) and np.all(indices<len(trace)) and np.max(abs(trace[indices,0]-fixed_waist[:,0]))<1e-7
                waist[indices]=fixed_waist[:,4:7]
            total_waist=waist+residual[:,:3]
            total_height=-gain*manifest['controller_modifier']['body_drop_m']*ramp+residual[:,3]
            checks=dict(finite=bool(np.isfinite(trace).all()),commands=bool(np.array_equal(trace[:,1:6],expected_command)),
                cadence=bool(np.allclose(np.diff(trace[:,0]),.005,atol=1e-7,rtol=0)),start=abs(float(trace[0,0]))<1e-12,
                coverage=bool(-1e-7<=float(times[-1]-trace[-1,0])<=.0050001 and (not row['success'] or abs(float(trace[-1,0]+.005-times[-1]))<1e-7)),
                target=bool(np.allclose(trace[:,6:11],target,atol=1e-15,rtol=0)),filter=bool(np.allclose(trace[:,11:16],residual,atol=1e-15,rtol=0)),
                velocity=bool(np.allclose(trace[:,16:21],velocity,atol=1e-13,rtol=0)),
                foreaft_bound=bool(abs(residual[:,4]).max()<=.001+1e-12),total_waist_bound=bool(abs(total_waist).max()<=.015+1e-12),total_body_bound=bool(total_height.max()<=1e-12 and total_height.min()>=-.015-1e-12),
                inactive_before_closure=bool(np.array_equal(residual[trace[:,0]<=99.66],np.zeros_like(residual[trace[:,0]<=99.66]))))
            rows.append(dict(case=row['case'],passed=all(checks.values()),checks=checks,geometry_gain=gain,
                maximum_combined_waist_offset_rad=float(abs(total_waist).max()),combined_height_range_m=[float(total_height.min()),float(total_height.max())]))
    result=dict(passed=all(r['passed'] for r in rows),rows=rows,manifest_sha256=sha(directory/'manifest.json'),summary_sha256=sha(directory/'summary.json'),source_sha256=sha(__file__),
        scope='Added controller command reconstruction and combined reference bounds. Task success remains separately audited; not an independent dynamics replay.')
    (directory/'adaptive_reference_audit.json').write_text(json.dumps(result,indent=2));(directory/'adaptive_reference_audit_source.py').write_bytes(Path(__file__).read_bytes())
    assert result['passed'];return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);audit(parser.parse_args().directory)
