"""Classify recorded training resets using all available pre-reset evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from training.wall_support_env import WallSupportEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--cases', nargs='+', required=True)
    p.add_argument('--time', type=float, default=49.)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    env = WallSupportEnv(ROOT / 'outputs/wall_support_bundle_pd15_v2')
    results = []
    for name in a.cases:
        run = a.suite / name
        z = dict(np.load(run / 'trajectory.npz'))
        i = int(np.argmin(abs(z['time_s'] - a.time)))
        rows = [r for r in csv.DictReader((run / 'control_log.csv').open()) if float(r['time_s']) <= a.time + 1e-7]
        peaks = {key: max(float(r[key]) for r in rows) for key in ['other_contact_peak_n', 'self_contact_peak_n', 'flight_contact_peak_n', 'base_error_peak_m']}
        q = z['qpos'][:i+1, env.qadr]
        margin = float(np.min(np.minimum(q - env.model.jnt_range[env.joints, 0], env.model.jnt_range[env.joints, 1] - q)))
        checks = dict(reset_time_available=bool(abs(z['time_s'][i] - a.time) < 1e-7), no_prior_unintended_contact=peaks['other_contact_peak_n'] < 1., no_prior_self_contact=peaks['self_contact_peak_n'] < 1., no_prior_swing_collision=peaks['flight_contact_peak_n'] < 1., prior_base_tracking=peaks['base_error_peak_m'] < .03, saved_native_positions=margin >= -1e-7)
        results.append(dict(case=name, eligible=all(checks.values()), checks=checks, prior_peaks=peaks, minimum_saved_joint_margin_rad=margin, trajectory_sha256=hashlib.sha256((run / 'trajectory.npz').read_bytes()).hexdigest()))
    env.close()
    result = dict(reset_time_s=a.time, results=results, eligible_cases=[r['case'] for r in results if r['eligible']], excluded_cases=[r['case'] for r in results if not r['eligible']], scope='Runtime cumulative collision/tracking peaks through reset and saved-state joint ranges. Exclusion is a failed prefix, not an improved success rate; report separately. No between-sample position guarantee.', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():raise FileExistsError(a.out)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
