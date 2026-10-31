"""Plot calibration evidence; these figures do not show policy learning."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--critic', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    s = json.loads((a.critic / 'summary.json').read_text())
    z = np.load(a.critic / 'value_targets.npz')
    selected = z['splits'] == 'validation'
    y, before, after = (z[k].ravel() for k in ('returns', 'values_before', 'values_after'))
    cases = list(dict.fromkeys(z['cases'].tolist()))
    rows = []
    for case in cases:
        mask = selected & (z['cases'] == case)
        rows.append([float(np.mean((v[mask] - y[mask]) ** 2)) for v in (before, after)])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    h = s['history']
    axes[0].plot([r['updates'] for r in h],
                 [r['validation']['explained_variance'] for r in h], label='Validation')
    axes[0].axhline(.8, color='tab:red', linestyle='--', label='Declared gate: 0.8')
    axes[0].axvline(s['selected_update'], color='gray', linestyle=':', label='Selected checkpoint')
    axes[0].set(xlabel='Additional critic gradient updates', ylabel='Explained variance',
                title='Resumed critic fitting (actor frozen)')
    axes[0].legend(fontsize=8)
    positions = np.arange(len(cases))
    values = np.asarray(rows)
    axes[1].barh(positions - .18, values[:, 0], height=.36, label='Original seed')
    axes[1].barh(positions + .18, values[:, 1], height=.36, label='Recalibrated critic')
    axes[1].set_yticks(positions, cases)
    axes[1].set(xscale='log', xlabel='Value prediction MSE (log scale)',
                title='Known-case validation errors')
    axes[1].legend(fontsize=8)
    fig.suptitle('Calibration only: reused development episodes, no PPO improvement claim')
    a.output.mkdir(exist_ok=False)
    for suffix in ('png', 'pdf'):
        fig.savefig(a.output / ('critic_calibration.' + suffix), dpi=160)
    plt.close(fig)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest = dict(source_sha256=digest(__file__), critic_summary_sha256=digest(a.critic/'summary.json'),
                    targets_sha256=digest(a.critic/'value_targets.npz'), cases=cases, mse=rows)
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (a.output/'plot_source.py').write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
    main()
