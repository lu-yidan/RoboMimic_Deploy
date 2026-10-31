"""Check every limited command, including absence of missing middle trace rows."""
import argparse
from pathlib import Path
import json
import numpy as np
from tools.validate_swing_reference import read,write,sha


def reconstruct(reference,physics_times,protocol):
    clock=np.r_[0.,physics_times[9:-1:10]]
    indices=np.clip(np.searchsorted(reference['time'],clock,side='right')-1,0,len(reference['time'])-2)+1
    active=reference['active'][indices].astype(bool)
    caps=np.full((len(clock),4),350.)
    expected={}
    for group,column in (('left_hook',2),('right_hook',3)):
        starts=np.flatnonzero(active[1:,column]&~active[:-1,column])+1
        for i in starts:
            if clock[i]<=reference['cycle_ends_s'][0]+1e-7:continue
            start=float(clock[i]);off=np.flatnonzero(~active[i:,column])
            end=i+int(off[0]) if len(off) else len(clock)
            release_index=np.searchsorted(clock,start+protocol['delays_seconds'][group]-1e-8)
            release=float(clock[release_index]) if release_index<end else None
            interval=np.arange(i,end)
            fraction=np.zeros(len(interval)) if release is None else np.clip((clock[interval]-release)/protocol['ramp_seconds'],0,1)
            weight=fraction*fraction*(3-2*fraction)
            caps[interval,column]=(protocol['probe_force_n']+(700.-protocol['probe_force_n'])*weight)/2
            expected[(group,start)]=release
    selected=np.any((caps<350.-1e-10)&active,axis=1)
    return clock[selected],caps[selected],active[selected],expected


def audit(run,output):
    m=read(run/'manifest.json');protocol=read(run/'timed_loading_protocol.json')
    record=read(run/'timed_loading_record.json');task=read(run/'v2_multicycle_audit.json')
    assert protocol==m['timed_loading'] and not m.get('contact_admission')
    assert sha(run/'timed_loading_protocol.json')==m['timed_loading_protocol_sha256']==record['protocol_sha256']
    assert sha(run/'physics_telemetry.npz')==record['telemetry_sha256']
    assert sha(run/'timed_loading_trace.json')==record['trace_sha256']
    assert task['manifest_sha256']==sha(run/'manifest.json')
    with np.load(run/'reference.npz') as f:reference={k:f[k] for k in f.files}
    with np.load(run/'physics_telemetry.npz') as f:times=f['time_s']
    expected,caps,active,events=reconstruct(reference,times,protocol)
    rows=read(run/'timed_loading_trace.json')
    actual=np.array([r['time_s'] for r in rows])
    same_clock=bool(np.array_equal(actual,expected))
    checks=dict(command_clock_complete=same_clock,no_duplicates=len(np.unique(actual))==len(actual),
        cap_values=False,active_values=False,controller_label=all(r.get('controller_type')=='timed_open_loop' for r in rows),
        task_binding=task['trajectory_sha256']==sha(run/'trajectory.npz'))
    if len(rows)==len(expected):
        checks['cap_values']=bool(np.allclose(np.array([r['caps_per_contact_n'] for r in rows]),caps,rtol=0,atol=1e-10))
        checks['active_values']=bool(np.array_equal(np.array([r['active'] for r in rows]),active))
    report=dict(passed=all(checks.values()),checks=checks,expected_limited_commands=len(expected),
        recorded_limited_commands=len(rows),events=len(events),source_sha256=sha(__file__),
        manifest_sha256=sha(run/'manifest.json'),trace_sha256=record['trace_sha256'],
        telemetry_sha256=record['telemetry_sha256'],task_audit_sha256=sha(run/'v2_multicycle_audit.json'),
        scope='Additional trace completeness and cap audit. Task success is separate; a truncated run can have a complete recorded command trace and still fail the task.')
    assert not output.exists();write(output,report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(audit(a.run.resolve(),a.output.resolve())),flush=True)
