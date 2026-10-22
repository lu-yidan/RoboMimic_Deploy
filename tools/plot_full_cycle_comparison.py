"""Plot matched finite-case metrics without statistical/generalization claims."""
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
    parser.add_argument('experiment', type=Path)
    args = parser.parse_args()
    source = args.experiment/'paired_metrics.json'
    rows = json.loads(source.read_text())['cases']
    names = {'nominal': 'Nominal', 'foot_z_plus5': 'Foot bar\nZ +5 mm', 'foot_x_minus10': 'Foot bar\nX -10 mm'}
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    colors = ['#8197ad', '#268978']
    for ax, field, ylabel in zip(axes, ['base_peak_m', 'minimum_joint_margin_rad'],
                                ['Peak base tracking error (mm)', 'Minimum native joint margin (mrad)']):
        for i, label in enumerate(['zero', 'after']):
            key = label+'_'+field
            values = [1000*row[key] for row in rows]
            bars = ax.bar(x+(i-.5)*.34, values, width=.34, color=colors[i],
                          label='Zero residual' if i == 0 else 'PPO after 8192 steps')
            ax.bar_label(bars, fmt='%.2f', padding=4, fontsize=9)
        ax.set_xticks(x, [names.get(row['case'], row['case']) for row in rows])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 6 if field == 'base_peak_m' else 23)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_title('Lower is better; changes are mixed', fontsize=11)
    axes[1].set_title('Larger is better; all three margins decrease', fontsize=11)
    axes[0].legend(loc='upper left', fontsize=9, frameon=False)
    fig.suptitle('Full-cycle residual PPO: both methods pass 3 of 3 declared cases', fontsize=13)
    fig.text(.5, .03, 'Single seed; one completed training episode. No success-count gain or held-out generalization claim.',
             ha='center', fontsize=9, color='#424b55')
    fig.tight_layout(rect=[0, .08, 1, .93])
    for suffix in ('png', 'pdf'):
        fig.savefig(args.experiment/('paired_metrics.'+suffix), dpi=180)
    plt.close(fig)
    (args.experiment/'paired_metrics_figure_manifest.json').write_text(json.dumps(dict(
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
