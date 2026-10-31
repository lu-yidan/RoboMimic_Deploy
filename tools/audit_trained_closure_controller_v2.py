"""Reconstruct added controller commands; task success remains separate."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.continue_vector_full_cycle_training import sha
from training.closure_body_target import body_target_offset


def audit(out):
    out=Path(out);m=json.loads((out/'manifest.json').read_text())
    s=json.loads((out/'summary.json').read_text());c=m['controller_modifier']
    bias=np.asarray(c['waist_bias']);rows=[]
    for result in s['after']:
        path=out/f"after_{result['case']}.npz"
        assert sha(path)==result['trajectory_sha256']
        with np.load(path) as z:
            obs=z['observations'];ts=z['time_s'];trace=z['waist_reference_trace']
            gain=float(np.clip((float(obs[0,138])-.625)/.125,0,1))
            assert np.all(obs[:,138]==obs[0,138])
            target=[];filtered=[];velocity=[];res=np.zeros(3)
            for t in trace[:,0]:
                u=float(np.clip((t-99.66)/c['waist_ramp_seconds'],0,1))
                goal=.015*gain*u*u*(3-2*u)*bias
                old=res.copy();res+=.05*(goal-old)
                target.append(goal);filtered.append(res.copy());velocity.append((res-old)/.005)
            errors={}
            for name,expected,actual in [('target',target,trace[:,1:4]),('residual',filtered,trace[:,4:7]),('velocity',velocity,trace[:,7:10])]:
                expected=np.asarray(expected).reshape(-1,3)
                errors[name]=float(abs(expected-actual).max()) if len(trace) else 0.
            expected_body=np.asarray([body_target_offset(t,gain*c['body_drop_m']) for t in ts])
            errors['body']=float(abs(expected_body-z['body_reference_trace']).max())
            # A declared failure can terminate before this controller activates.
            nonzero=bool(gain and np.any(bias) and ts[-1] > 99.66)
            cadence=bool(len(trace)>0 and np.all(abs(np.diff(trace[:,0])-.005)<1e-8)) if nonzero else len(trace)==0
            span=bool(len(trace) and 99.66<=trace[0,0]<=99.6700001 and
                      -1e-7 <= ts[-1]-trace[-1,0] <= .0050001 and
                      (not result['success'] or abs(trace[-1,0]-(ts[-1]-.005))<1e-7)) if nonzero else len(trace)==0
            checks=dict(command_reconstruction=max(errors.values())<1e-12,command_cadence=cadence,
                command_span=span,geometry_gain=gain==float(z['controller_geometry_gain']),
                bounded_waist=not len(trace) or float(abs(trace[:,4:7]).max())<=.015+1e-12)
            if gain==0 and m.get('require_inactive_case_exact_parent',True):
                parent=Path(m['parent_experiment'])/path.name
                with np.load(parent) as old:
                    checks['inactive_case_exact_parent']=all(np.array_equal(z[k],old[k]) for k in ('qpos','qvel','time_s','actions','observations','torques','rewards'))
            rows.append(dict(case=result['case'],passed=all(checks.values()),checks=checks,errors=errors,geometry_gain=gain))
    report=dict(passed=all(r['passed'] for r in rows),rows=rows,manifest_sha256=sha(out/'manifest.json'),summary_sha256=sha(out/'summary.json'),source_sha256=sha(__file__),
        scope='Added reference-command reconstruction. Inactive-case equality is required only for fixed-network search candidates; a trained actor may change those cases. Not a separate dynamics replay or a task-success certificate.')
    destination=out/'controller_modifier_audit_v2.json'
    assert not destination.exists()
    destination.write_text(json.dumps(report,indent=2))
    (out/'controller_modifier_audit_v2_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report),flush=True)
    assert report['passed']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('experiment',type=Path);audit(p.parse_args().experiment)
