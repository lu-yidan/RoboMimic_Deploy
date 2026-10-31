"""Plot audited endpoint results from the matched reward comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('comparison', type=Path)
    args = parser.parse_args()
    source = args.comparison/'summary.json'
    report = json.loads(source.read_text())
    assert report['passed'] and all(report['matched_checks'].values())
    rows = report['rows']
    x = np.arange(len(rows))
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, layout='constrained')
    for label, name, color, offset in [
        ('parent', 'Common starting checkpoint', '#777777', -.18),
        ('average', 'Average quality reward', '#c56a35', 0),
        ('bottleneck', 'Bottleneck quality reward', '#167f98', .18),
    ]:
        for ax, key in zip(axes, ['closure_mrad', 'rise_max_mm', 'scaled_margin']):
            values = [np.nan if row[label+'_'+key] is None else row[label+'_'+key] for row in rows]
            ax.plot(x+offset, values, 'o', color=color, ms=5, label=name)
            failed = [i for i, row in enumerate(rows) if not row[label+'_success']]
            ax.scatter(x[failed]+offset, np.array(values)[failed], s=85, facecolors='none', edgecolors=color)
    axes[0].axhline(40, color='crimson', ls='--', lw=1)
    axes[1].axhline(260, color='crimson', ls='--', lw=1)
    axes[2].axhline(0, color='crimson', ls='--', lw=1)
    for ax, title in zip(axes, ['Final joint error (mrad)', 'Maximum final root rise (mm)\nDashed: upper limit', 'Min(joint slack / 1 mrad,\nheight slack / 0.1 mm)']):
        ax.set_ylabel(title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9, ncol=3)
    axes[2].set_xticks(x, [r['case']+('*' if r['used_for_training'] else '') for r in rows], rotation=50, ha='right')
    axes[2].set_xlabel('* Used for training; circled points fail task acceptance. All cases are development cases.')
    fig.suptitle('Matched 65,536-step reward experiment (one seed)\nJoint and height constraints remain unchanged; no automatic model promotion')
    for suffix in ['png', 'pdf']:
        fig.savefig(args.comparison/f'endpoints.{suffix}', dpi=180)
    plt.close(fig)
    metadata = dict(summary_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.comparison/'plot_provenance.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
