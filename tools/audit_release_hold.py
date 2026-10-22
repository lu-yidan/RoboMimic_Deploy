"""Verify a protected stop against its unprotected physical counterpart."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--guarded',type=Path,required=True)
    p.add_argument('--unguarded',type=Path,required=True);a=p.parse_args()
    sg=json.loads((a.guarded/'summary.json').read_text());su=json.loads((a.unguarded/'summary.json').read_text())
    events=json.loads((a.guarded/'release_events.json').read_text());denied=[e for e in events if not e['allowed']]
    if not denied:raise ValueError('No denied release to audit')
    event=denied[0];t=event['time_s'];g=np.load(a.guarded/'trajectory.npz');u=np.load(a.unguarded/'trajectory.npz')
    before=np.flatnonzero(g['time_s']<=t+1e-8)
    if len(before)>len(u['qpos']):raise ValueError('Unprotected run ended before comparison point')
    same_times=bool(np.allclose(g['time_s'][before],u['time_s'][before],atol=1e-8))
    difference=float(np.max(abs(g['qpos'][before]-u['qpos'][before]))) if same_times else None
    rows=list(csv.DictReader((a.guarded/'control_log.csv').open(encoding='utf-8')))
    hold=[r for r in rows if float(r['time_s'])>t+1e-8]
    groups=('left_foot','right_foot','left_hook','right_hook')
    kept=[group for group,active in zip(groups,event['hold_active']) if active]
    minima={group:min(float(r[group+'_vertical_n']) for r in hold) for group in groups} if hold else {}
    checks=dict(same_scene=sg['scene_variant']==su['scene_variant'],
                no_unconfirmed_support_added=bool(kept) and all(not h or (b and c) for h,b,c in zip(event['hold_active'],event.get('active_before',event['hold_active']),event.get('confirmed',event['hold_active']))),
                same_preintervention_trajectory=same_times and difference<1e-10,
                denied_support_retained=all(group in kept for group in event['released']),
                full_half_second_hold=sg['reason']=='release_interlock_stop' and abs(sg['hold_duration_s']-.5)<1e-7,
                retained_contacts_loaded=bool(hold) and all(minima[g]>10. for g in kept),
                no_extra_or_self_contact=sg['other_contact_peak_n']<1. and sg['self_contact_peak_n']<1.,
                base_tracking=sg['base_error_peak_m']<.03,torque_bounds=sg['max_torque_ratio']<=1.+1e-5)
    result=dict(passed=bool(all(checks.values())),checks=checks,event=event,
                preintervention_qpos_max_difference=difference,hold_minimum_vertical_n=minima,
                hold_sample_count=len(hold),hold_duration_s=sg['hold_duration_s'],
                unprotected_reason=su['reason'],unprotected_time_s=su['simulation_time_s'],
                scope='Finite 500 ms holding demonstration only; not an infinite safety invariant or automatic retreat.')
    (a.guarded/'hold_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
