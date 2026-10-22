"""Measure recorded-prefix reset divergence from uninterrupted physics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.run / 'manifest.json').read_text())
    rows = []
    for case in manifest['cases']:
        prefix_path = Path(manifest['suite']) / case / 'trajectory.npz'
        full = dict(np.load(prefix_path))
        segment = dict(np.load(a.run / (case + '.npz')))
        indices = np.searchsorted(full['time_s'], segment['time_s'])
        indices = np.clip(indices, 1, len(full['time_s']) - 1)
        indices -= (abs(full['time_s'][indices - 1] - segment['time_s']) < abs(full['time_s'][indices] - segment['time_s'])).astype(int)
        dt = float(np.max(abs(full['time_s'][indices] - segment['time_s'])))
        if dt > 1e-7:
            raise ValueError('Different trajectory sample clocks')
        rows.append(dict(case=case, samples=len(indices), timestamp_max_difference_s=dt, initial_qpos_exact=bool(np.array_equal(full['qpos'][indices[0]], segment['qpos'][0])), initial_qvel_exact=bool(np.array_equal(full['qvel'][indices[0]], segment['qvel'][0])), qpos_max_difference=float(np.max(abs(full['qpos'][indices] - segment['qpos']))), qvel_max_difference=float(np.max(abs(full['qvel'][indices] - segment['qvel']))), prefix_sha256=hashlib.sha256(prefix_path.read_bytes()).hexdigest()))
    result = dict(results=rows, scope='Measured divergence, not a pass/fail threshold. Reset restores qpos/qvel only; solver cache and contact confirmation history restart. Segment success cannot replace continuous validation.', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.run / 'continuity_comparison.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
