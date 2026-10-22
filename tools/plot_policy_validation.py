"""Plot independently audited paired validation without statistical claims."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=Path)
    args = parser.parse_args()
    path = args.experiment
    summary = json.loads((path/'policy_validation_summary.json').read_text())
    assert sha(path/'independent_training_audit.json') == summary['audit_sha256']
    assert sha(path/'policy.zip') == summary['checkpoint_sha256']
    rows = summary['rows']
    heights = sorted({row['offset_mm'][2] for row in rows})
    offsets = sorted({row['offset_mm'][0] for row in rows})
    assert len(rows) == len(heights)*len(offsets)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, label, name in zip(axes, ('zero', 'policy'), ('Reference + controller', 'Same reference + learned residual')):
        grid = np.zeros((len(offsets), len(heights)))
        for yi, x in enumerate(offsets):
            for xi, h in enumerate(heights):
                found = [r for r in rows if r['offset_mm'][0] == x and r['offset_mm'][2] == h]
                assert len(found) == 1
                row = found[0][label]
                grid[yi, xi] = int(row['accepted'])
                ax.text(xi, yi, 'PASS' if row['accepted'] else 'FAIL', ha='center', va='center', fontsize=11, color='#222222')
        ax.imshow(grid, cmap=ListedColormap(['#eac0a4', '#b8d8cb']), vmin=0, vmax=1, aspect='auto')
        ax.set(xticks=range(len(heights)), xticklabels=[f'{h:g}' for h in heights],
               yticks=range(len(offsets)), yticklabels=[f'{x:+g}' for x in offsets],
               xlabel='Rod height offset (mm)', ylabel='Rod x offset (mm)',
               title=f'{name}\nStrict acceptance: {int(grid.sum())} / {len(rows)}')
    fig.suptitle('Frozen 12-case validation | full-cycle acceptance', fontsize=13)
    fig.text(.5, .025, 'Deterministic simulated offsets, oracle observations and privileged contact control. Model-selection validation, not a test set.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .93))
    for extension in ('png', 'pdf'):
        fig.savefig(path/f'policy_validation_grid.{extension}', dpi=180)
    plt.close(fig)
    (path/'policy_validation_figure.json').write_text(json.dumps(dict(summary_sha256=sha(path/'policy_validation_summary.json'), source_sha256=sha(__file__)), indent=2))
    (path/'policy_validation_figure_source.py').write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
    main()
