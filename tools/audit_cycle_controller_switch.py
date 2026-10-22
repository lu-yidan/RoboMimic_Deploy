"""Verify controller selection and uninterrupted clock for repeated cycles."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--first-cycle',type=Path)
    p.add_argument('--require-events',action='store_true');a=p.parse_args()
    manifest=json.loads((a.run/'manifest.json').read_text());ref=dict(np.load(a.run/'reference.npz'))
    z=dict(np.load(a.run/'trajectory.npz'));rows=list(csv.DictReader((a.run/'control_log.csv').open()))
    times=ref['time'];ends=ref['cycle_ends_s'];mismatches=[]
    eligible=np.isin(ref['phase'][1:],['weight_transfer','lf_unload','lf_swing','lf_land'])
    switch_times=[]
    for start,end in zip(np.r_[0.,ends[:-1]],ends):
        candidates=times[:-1][eligible & (times[:-1]>=start-1e-8) & (times[:-1]<end-1e-8)]
        switch_times.append(float(candidates[0]) if len(candidates) else float('inf'))
    for row in rows:
        command_t=float(row['time_s'])-.005
        cycle=int(np.searchsorted(ends,command_t+1e-8,side='right'))
        expected='contact_frame' if command_t>=switch_times[cycle]-1e-8 else 'fixed_contact_points'
        if expected!=row['controller_mode']:mismatches.append(dict(time_s=row['time_s'],expected=expected,actual=row['controller_mode']))
    first_difference=None
    if a.first_cycle:
        first=dict(np.load(a.first_cycle/'trajectory.npz'));n=len(first['time_s'])
        np.testing.assert_allclose(z['time_s'][:n],first['time_s'],atol=1e-8,rtol=0)
        first_difference=float(np.max(abs(z['qpos'][:n]-first['qpos'])))
    checks=dict(strategy_enabled=bool(manifest['config'].get('reset_contact_frame_each_cycle')),
        uninterrupted_clock=bool(np.allclose(np.diff(z['time_s']),.02,atol=1e-7,rtol=0)),
        exact_mode_schedule=not mismatches,first_cycle_unchanged=first_difference is None or first_difference<1e-12)
    events_path=a.run/'controller_switch_events.json'
    if a.require_events or events_path.exists():
        events=json.loads(events_path.read_text()) if events_path.exists() else []
        correct=len(events)==2*len(ends)
        for k,start in enumerate(np.r_[0.,ends[:-1]]):
            for mode,phase,expected_time in [('fixed_contact_points','settle',start),('contact_frame','weight_transfer',switch_times[k])]:
                selected=[r for r in events if r['cycle']==k+1 and r['mode']==mode]
                correct &= len(selected)==1
                if selected:
                    event=selected[0]
                    correct &= event['phase']==phase and -1e-10<=event['time_s']-expected_time<=.005+1e-7
                    correct &= abs(event['leg_acceleration_kp']-manifest['config']['joint_kp'])<1e-9
        checks['complete_phase_aligned_switch_events']=bool(correct)
    result=dict(passed=all(checks.values()),checks=checks,first_cycle_qpos_max_difference=first_difference,mismatches=mismatches,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Controller schedule and recorded-state continuity audit; does not replace dynamics/force/endpoint acceptance.')
    (a.run/'controller_switch_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    (a.run/'controller_switch_audit_source.py').write_bytes(Path(__file__).read_bytes())
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
