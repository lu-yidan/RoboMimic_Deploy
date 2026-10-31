"""Supplement the contact audit with every limited-command coverage check."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.validate_swing_reference import read,write,sha


def reconstruct(reference,physics_times,parameters,events):
    clock=np.r_[0.,physics_times[9:-1:10]]
    indices=np.clip(np.searchsorted(reference['time'],clock,side='right')-1,0,len(reference['time'])-2)+1
    active=reference['active'][indices].astype(bool)
    caps=np.full((len(clock),4),350.)
    expected=[]
    for group,column in (('left_hook',2),('right_hook',3)):
        for i in np.flatnonzero(active[1:,column]&~active[:-1,column])+1:
            if clock[i]<=reference['cycle_ends_s'][0]+1e-7:continue
            start=float(clock[i]);matches=[e for e in events if e['group']==group and abs(e['start_s']-start)<1e-8]
            assert len(matches)==1,'Missing or duplicate contact activation'
            event=matches[0];expected.append((group,start))
            off=np.flatnonzero(~active[i:,column]);end=i+int(off[0]) if len(off) else len(clock)
            confirmed=event['confirmed_s']
            if confirmed is not None:
                assert start<=confirmed<clock[end-1]+1e-8
                assert np.any(abs(clock[i:end]-confirmed)<1e-8),'Confirmation is not on the command clock'
            interval=np.arange(i,end)
            u=np.zeros(len(interval)) if confirmed is None else np.clip((clock[interval]-confirmed)/parameters['ramp_seconds'],0,1)
            caps[interval,column]=(parameters['probe_force_n']+(700.-parameters['probe_force_n'])*u*u*(3-2*u))/2
    assert len(expected)==len(events),'Unexpected recorded activation'
    selected=np.any((caps<350.-1e-10)&active,axis=1)
    return clock[selected],caps[selected],active[selected],expected


def audit(run,output,event_audit_name='hook_admission_audit.json'):
    assert Path(event_audit_name).name==event_audit_name
    m=read(run/'manifest.json');p=read(run/'hook_admission_protocol.json')
    record=read(run/'hook_admission_record.json');event_audit=read(run/event_audit_name)
    assert m['contact_admission']==p and not m.get('timed_loading')
    assert sha(run/'manifest.json')==event_audit['manifest_sha256']
    assert sha(run/'hook_admission_protocol.json')==m['hook_admission_protocol_sha256']==record['protocol_sha256']
    assert sha(run/'hook_admission_trace.json')==record['trace_sha256']==event_audit['trace_sha256']
    assert sha(run/'physics_telemetry.npz')==record['telemetry_sha256']==event_audit['telemetry_sha256']
    assert sha(run/'trajectory.npz')==record['trajectory_sha256']==event_audit['trajectory_sha256']
    with np.load(run/'reference.npz') as z:ref={k:z[k] for k in z.files}
    with np.load(run/'physics_telemetry.npz') as z:times=z['time_s']
    expected,caps,active,events=reconstruct(ref,times,p['parameters'],event_audit['events'])
    rows=read(run/'hook_admission_trace.json');actual=np.array([r['time_s'] for r in rows])
    checks=dict(command_clock_complete=bool(np.array_equal(actual,expected)),
                no_duplicate_commands=len(np.unique(actual))==len(actual),cap_values=False,active_values=False,
                original_trace_audit_passed=event_audit['trace_checks_passed'])
    if len(rows)==len(expected):
        checks['cap_values']=bool(np.allclose(np.array([r['caps_per_contact_n'] for r in rows]),caps,rtol=0,atol=1e-10))
        checks['active_values']=bool(np.array_equal(np.array([r['active'] for r in rows]),active))
    report=dict(passed=all(checks.values()),checks=checks,events=len(events),
                expected_limited_commands=len(expected),recorded_limited_commands=len(rows),
                manifest_sha256=sha(run/'manifest.json'),event_audit_sha256=sha(run/event_audit_name),
                trace_sha256=record['trace_sha256'],telemetry_sha256=record['telemetry_sha256'],
                source_sha256=sha(__file__),
                scope='Supplemental trace completeness using independently reconstructed activation clocks and the existing confirmation audit. No new dynamics, no independent reconstruction of normal force or contact identity, no controller promotion.')
    assert not output.exists();write(output,report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--event-audit-name',default='hook_admission_audit.json');a=p.parse_args()
    print(json.dumps(audit(a.run.resolve(),a.output.resolve(),a.event_audit_name)),flush=True)
