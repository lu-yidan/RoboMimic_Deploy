"""Check the timed ablation's cap schedule against pilot command timestamps.

This replays only the scheduling state machine, with a stub QP. It cannot
establish dynamic success, collision-force improvement or feedback necessity.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from training import timed_hook_loading as timed
from tools.validate_swing_reference import read,write,sha,ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();pilot=a.pilot.resolve()
    m=read(pilot/'manifest.json');audit=read(pilot/'hook_admission_audit_v2.json')
    assert m['case']=='training_12' and m['cycles']==2 and audit['passed']
    assert audit['manifest_sha256']==sha(pilot/'manifest.json')
    assert audit['telemetry_sha256']==sha(pilot/'physics_telemetry.npz')
    assert audit['trace_sha256']==sha(pilot/'hook_admission_trace.json')
    with np.load(pilot/'reference.npz') as f:ref={k:f[k] for k in f.files}
    with np.load(pilot/'physics_telemetry.npz') as f:clock=np.r_[0.,f['time_s'][9:-1:10]]
    event_checks={event['group']:abs(event['confirmed_s']-event['start_s']-timed.PILOT_DELAYS[event['group']])<1e-8
                  for event in audit['events']}
    assert set(event_checks)==set(timed.PILOT_DELAYS) and all(event_checks.values())
    original_calls=[]
    def original(q,v,active):original_calls.append(float(base.env.data.time));return np.zeros(29),{}
    # Deliberately provide no contact_metrics, model, or contact-force sensor.
    # Any feedback access would fail this test instead of silently supplying
    # the pilot's force measurements to the open-loop comparator.
    base=SimpleNamespace(ends=ref['cycle_ends_s'],env=SimpleNamespace(data=SimpleNamespace(time=0.)),
                         baseline=SimpleNamespace(command=original))
    state=timed.install_timed_hook_loading(base)
    def qp(controller,q,v,active,caps):
        return np.zeros(29),dict(force_plan=[],base_equilibrium_residual=0.)
    with patch.object(timed,'capped_command',side_effect=qp):
        for t in clock:
            base.env.data.time=float(t)
            i=int(np.clip(np.searchsorted(ref['time'],t,side='right')-1,0,len(ref['time'])-2))+1
            base.baseline.command(None,None,ref['active'][i])
    expected=read(pilot/'hook_admission_trace.json');actual=state['trace']
    same_len=len(expected)==len(actual)
    time_exact=same_len and all(x['time_s']==y['time_s'] for x,y in zip(actual,expected))
    caps_exact=same_len and all(x['caps_per_contact_n']==y['caps_per_contact_n'] for x,y in zip(actual,expected))
    active_exact=same_len and all(x['active']==y['active'] for x,y in zip(actual,expected))
    checks=dict(delay_matches_pilot=all(event_checks.values()),time_exact=time_exact,
        caps_exact=caps_exact,active_exact=active_exact,
        first_cycle_original_only=bool(np.array_equal(
            np.asarray(original_calls)[np.asarray(original_calls)<ref['cycle_ends_s'][0]],
            clock[clock<ref['cycle_ends_s'][0]]))
            and all(t>ref['cycle_ends_s'][0] for t in [r['time_s'] for r in actual]))
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'schedule_check.json',dict(passed=all(checks.values()),checks=checks,
        rows=len(actual),delays_seconds=timed.PILOT_DELAYS,
        command_timestamps=len(clock),source_sha256=sha(__file__),
        controller_source_sha256=sha(ROOT/'training/timed_hook_loading.py'),
        pilot_audit_sha256=sha(pilot/'hook_admission_audit_v2.json'),
        pilot_trace_sha256=sha(pilot/'hook_admission_trace.json'),
        scope='Recorded T12 command clocks and planned active contacts; state-machine-only comparison with a stub QP and no sensor object. Exact caps do not constitute a fresh physical rollout. Delays were selected from this already observed pilot before any new grid results.'))
    (a.output/'schedule_check_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(passed=all(checks.values()),checks=checks,rows=len(actual))),flush=True)
    assert all(checks.values())


if __name__=='__main__':main()
