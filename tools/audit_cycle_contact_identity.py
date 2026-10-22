"""Audit loaded contact rod identity throughout saved cycles, not just endpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

PHASES = ['settle', 'rh_lift', 'rh_retreat', 'rh_raise', 'rh_approach', 'rh_seat', 'rh_load', 'lh_lift', 'lh_retreat', 'lh_raise', 'lh_approach', 'lh_seat', 'lh_load', 'step_preparation', 'rf_unload', 'rf_swing', 'rf_land', 'rf_load', 'weight_transfer', 'lf_unload', 'lf_swing', 'lf_land', 'lf_load', 'body_rise', 'final_center', 'final_hold', 'posture_closure', 'closed_hold']


def expected_rungs(cycle, phase):
    i = PHASES.index(phase)
    return {group: f'rung_{cycle + offset + int(i >= PHASES.index(transition)):02d}' for group, offset, transition in [('left_hook', 4, 'lh_approach'), ('right_hook', 4, 'rh_approach'), ('left_foot', 0, 'lf_swing'), ('right_foot', 0, 'rf_swing')]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    ref = dict(np.load(a.run / 'reference.npz'))
    ends = ref.get('cycle_ends_s', np.array([ref['time'][-1]]))
    path = a.run / 'contact_log.json'
    logs = json.loads(path.read_text())
    violations = []; loaded = 0
    for row in logs:
        # Phase is the command phase before the last 5 ms integration interval.
        cycle = min(int(np.searchsorted(ends, row['time_s'] - .005, side='right')), len(ends) - 1)
        expected = expected_rungs(cycle, row['phase'])
        for contact in row['contacts']:
            force = float(np.linalg.norm(contact['force_on_robot_world_n']))
            if force <= 1.:
                continue
            loaded += 1
            if contact['group'] not in expected or contact['external_geom'] != expected[contact['group']]:
                violations.append(dict(time_s=row['time_s'], phase=row['phase'], group=contact['group'], actual=contact['external_geom'], expected=expected.get(contact['group']), force_n=force))
    result = dict(passed=bool(loaded and not violations), loaded_contact_samples=loaded, violation_count=len(violations), violations=violations, contact_log_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scope='50 Hz recorded contact identities above1N. Before approach/foot swing use old rod; afterward use target rod. Does not assert force confirmation duration or between-sample correctness.')
    (a.run / 'contact_identity_audit.json').write_text(json.dumps(result, indent=2))
    (a.run / 'contact_identity_audit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: v for k, v in result.items() if k != 'violations'}, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
