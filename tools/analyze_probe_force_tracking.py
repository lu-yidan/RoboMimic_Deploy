"""Compare modeled vertical load limits with actual 2 kHz net support forces."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.validate_swing_reference import read,write,sha


def analyze(run,out,audit_name):
    m=read(run/'manifest.json');record=read(run/'hook_admission_record.json')
    audit=read(run/audit_name);trace=read(run/'hook_admission_trace.json')
    assert audit['passed'] and audit['trace_checks_passed']
    assert sha(run/'manifest.json')==audit['manifest_sha256']
    assert sha(run/'hook_admission_trace.json')==audit['trace_sha256']==record['trace_sha256']
    assert sha(run/'physics_telemetry.npz')==audit['telemetry_sha256']==record['telemetry_sha256']
    with np.load(run/'physics_telemetry.npz') as z:
        times=z['time_s'];forces=z['group_force_world_n']
    rows=[]
    for r in trace:
        active=np.asarray(r['active'],bool)
        planned=np.asarray(r['force_plan']).reshape(-1,2,3)
        assert len(planned)==int(active.sum())
        for g,column in (('left_hook',2),('right_hook',3)):
            if not active[column] or r['caps_per_contact_n'][column]>=350.-1e-10:continue
            hook=r['hooks'][g];t=r['time_s'];cap=2*r['caps_per_contact_n'][column]
            model_slot=list(np.flatnonzero(active)).index(column)
            predicted=float(planned[model_slot,:,2].sum())
            assert predicted<=cap+2e-3
            index=int(np.searchsorted(times,t+1e-8));before=index-1
            following=forces[index:index+10,record['groups'].index(g),2]
            assert len(following)==10 and abs(times[before]-t)<1e-8
            assert abs(times[index+9]-(t+.005))<1e-8
            measured=float(forces[before,record['groups'].index(g),2])
            assert abs(measured-r['measurements'][g]['vertical_n'])<1e-8
            maximum=float(following.max())
            rows.append(dict(cycle=r['cycle'],group=g,time_s=t,phase=r['states'][g],
                activation_s=hook['start'],confirmed_s=hook['confirmed'],
                modeled_total_vertical_cap_n=float(cap),modeled_vertical_force_n=predicted,
                actual_before_command_vertical_n=measured,actual_next_5ms_peak_vertical_n=maximum,
                actual_next_5ms_mean_vertical_n=float(following.mean()),
                peak_minus_modeled_cap_n=maximum-cap))
    groups={}
    for g in ('left_hook','right_hook'):
        selected=[r for r in rows if r['group']==g]
        groups[g]={}
        for phase in ('probe','ramp'):
            subset=[r for r in selected if r['phase']==phase]
            groups[g][phase]=None if not subset else dict(commands=len(subset),
                maximum_actual_vertical_n=max(r['actual_next_5ms_peak_vertical_n'] for r in subset),
                maximum_positive_cap_excess_n=max(0.,max(r['peak_minus_modeled_cap_n'] for r in subset)),
                commands_exceeding_modeled_cap=sum(r['peak_minus_modeled_cap_n']>1e-3 for r in subset),
                largest_excess_record=max(subset,key=lambda r:r['peak_minus_modeled_cap_n']))
    report=dict(groups=groups,rows=rows,run=str(run),manifest_sha256=sha(run/'manifest.json'),
        event_audit_sha256=sha(run/audit_name),trace_sha256=record['trace_sha256'],
        telemetry_sha256=record['telemetry_sha256'],source_sha256=sha(__file__),
        scope='Recorded limited commands only, after the unchanged first cycle. Actual net world-vertical group force over each following 5 ms interval is compared with the modeled two-contact vertical cap. Exceeding a modeled cap is not a solver bound violation: physical contact force is not directly limited. Net vertical force is not individual contact magnitude, stress or impact safety. No retuning or new dynamics.')
    out.mkdir(parents=True,exist_ok=False);write(out/'summary.json',report)
    (out/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(groups=groups,output=str(out))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--audit-name',default='hook_admission_audit.json');a=p.parse_args()
    analyze(a.run.resolve(),a.output.resolve(),a.audit_name)
