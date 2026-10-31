"""Check high-rate vertical-force dips between 200 Hz confirmation samples."""
import argparse
import json
from pathlib import Path
import numpy as np
from tools.validate_swing_reference import read,write,sha


def audit(run,out,event_name):
    m=read(run/'manifest.json');p=read(run/'hook_admission_protocol.json')
    e=read(run/event_name);record=read(run/'hook_admission_record.json')
    assert e['passed'] and e['trace_checks_passed']
    assert sha(run/'manifest.json')==e['manifest_sha256']
    assert sha(run/'physics_telemetry.npz')==record['telemetry_sha256']==e['telemetry_sha256']
    assert sha(run/'hook_admission_trace.json')==record['trace_sha256']==e['trace_sha256']
    trace=read(run/'hook_admission_trace.json');cfg=p['parameters']
    with np.load(run/'physics_telemetry.npz') as z:
        t=z['time_s'];force=z['group_force_world_n']
    rows=[]
    for event in e['events']:
        end=event['confirmed_s'];assert end is not None
        start=end-cfg['confirm_seconds'];g=event['group']
        mask=(t>=start-1e-8)&(t<=end+1e-8)
        samples=force[mask,record['groups'].index(g),2];times=t[mask]
        assert len(samples)==round(cfg['confirm_seconds']/.0005)+1
        low=samples<=cfg['confirm_force_n'];indices=np.flatnonzero(low)
        runs=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1) if len(indices) else []
        observed=[r['measurements'][g]['vertical_n'] for r in trace
            if start-1e-8<=r['time_s']<=end+1e-8 and g in r['measurements']]
        assert len(observed)==round(cfg['confirm_seconds']/.005)+1
        assert min(observed)>cfg['confirm_force_n']
        rows.append(dict(cycle=event['cycle'],group=g,start_s=start,confirmed_s=end,
            sampled_200hz_confirmation_passed=True,minimum_sampled_200hz_vertical_n=min(observed),
            full_rate_samples=len(samples),minimum_2000hz_vertical_n=float(samples.min()),
            samples_at_or_below_threshold=int(low.sum()),
            low_sample_occupancy_ms=float(low.sum()*.5),
            longest_low_sample_run_occupancy_ms=float(max((len(r) for r in runs),default=0)*.5),
            first_low_sample_time_s=float(times[indices[0]]) if len(indices) else None,
            full_rate_vertical_condition_sustained=bool(not low.any())))
    report=dict(events=len(rows),events_with_between_sample_vertical_dips=sum(not r['full_rate_vertical_condition_sustained'] for r in rows),
        rows=rows,threshold_n=cfg['confirm_force_n'],window_s=cfg['confirm_seconds'],
        controller_sampling_hz=200,telemetry_sampling_hz=2000,
        event_audit_sha256=sha(run/event_name),telemetry_sha256=record['telemetry_sha256'],
        manifest_sha256=sha(run/'manifest.json'),source_sha256=sha(__file__),
        scope='Additional temporal-resolution diagnostic, not a replacement gate or rerun. The controller requires 11 qualifying 200 Hz samples. A full-rate vertical-force threshold dip disproves an uninterrupted vertical-threshold claim, but does not prove all physical contact was lost. High-rate normal force and contact identity are not recorded. Occupancy counts samples at 0.5 ms each; crossing times between samples are unknown.')
    out.mkdir(parents=True,exist_ok=False);write(out/'summary.json',report)
    (out/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--event-audit-name',default='hook_admission_audit.json');a=p.parse_args()
    audit(a.run.resolve(),a.output.resolve(),a.event_audit_name)
