"""Independently check early-contact switching logs and their saved-state schedule."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    ref = dict(np.load(a.run / 'reference.npz'))
    trajectory = dict(np.load(a.run / 'trajectory.npz'))
    switches = json.loads((a.run / 'controller_switch_events.json').read_text())
    gates = json.loads((a.run / 'contact_gate_events.json').read_text())
    rows = list(csv.DictReader((a.run / 'control_log.csv').open()))
    ends = ref['cycle_ends_s']
    phase_at = lambda t: str(ref['phase'][min(max(int(np.searchsorted(ref['time'], t, side='right')), 1), len(ref['phase']) - 1)])
    starts = np.r_[0., ends[:-1]]
    errors = []
    for i, event in enumerate(switches):
        t = event['time_s']
        k = min(int(np.searchsorted(ends, t, side='right')), len(ends) - 1)
        if event['cycle'] != k + 1 or event['phase'] != phase_at(t):
            errors.append(dict(event=i, error='phase_or_cycle'))
        if i and switches[i-1]['mode'] == event['mode']:
            errors.append(dict(event=i, error='duplicate_mode'))
        if event['mode'] == 'fixed_contact_points':
            is_start = abs(t - starts[k]) <= .0050001
            gated = any(abs(g['time_s'] - t) < 1e-8 and g['missing'] for g in gates)
            if not is_start and not gated:
                errors.append(dict(event=i, error='unexplained_baseline'))
    for k, start in enumerate(starts):
        candidates = [e for e in switches if e['cycle'] == k + 1 and e['mode'] == 'contact_frame']
        eligible = ref['time'][:-1][(ref['phase'][1:] == 'step_preparation') & (ref['time'][:-1] >= start - 1e-8) & (ref['time'][:-1] < ends[k] - 1e-8)]
        if not candidates or not len(eligible) or not -1e-8 <= candidates[0]['time_s'] - eligible[0] <= .0050001:
            errors.append(dict(cycle=k+1, error='early_entry'))
    event_times = np.array([e['time_s'] for e in switches])
    for row in rows:
        # A row is emitted after the ten physics steps driven by this command.
        t = float(row['time_s']) - .005
        i = max(int(np.searchsorted(event_times, t + 1e-8, side='right')) - 1, 0)
        if row['controller_mode'] != switches[i]['mode']:
            errors.append(dict(time_s=float(row['time_s']), error='saved_mode'))
    checks = dict(schedule_consistent=not errors, uninterrupted_saved_clock=bool(np.allclose(np.diff(trajectory['time_s']), .02, atol=1e-7, rtol=0)), chronological_events=bool(np.all(np.diff(event_times) > 0)))
    result = dict(passed=all(checks.values()), checks=checks, errors=errors, gate_count=len(gates), switch_count=len(switches), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scope='Checks recorded timing, phase, controller transitions and gate explanations. Does not independently reconstruct sub-sample contact geometry or certify hardware safety.')
    (a.run / 'early_schedule_audit.json').write_text(json.dumps(result, indent=2))
    (a.run / 'early_schedule_audit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
