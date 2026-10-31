"""Plot audited endpoint outcomes with the unchanged acceptance threshold."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('comparison', type=Path)
    p.add_argument('--before-label', default='Fixed-reference baseline')
    p.add_argument('--after-label', default='Landing-window PPO')
    a = p.parse_args()
    s = json.loads((a.comparison/'comparison.json').read_text())
    rows = s['rows']
    x = np.arange(len(rows))
    names = [r['case'].replace('foot_z_plus', 'z+') for r in rows]
    before = np.array([r['before_closure_mrad'] if r['before_closure_mrad'] is not None else np.nan for r in rows])
    after = np.array([r['after_closure_mrad'] if r['after_closure_mrad'] is not None else np.nan for r in rows])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout='constrained', height_ratios=[2, 1])
    for values, offset, color, label in [(before, -.18, '#b75242', a.before_label), (after, .18, '#24758b', a.after_label)]:
        bars = axes[0].bar(x+offset, values, width=.34, color=color, label=label)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                axes[0].text(bar.get_x()+bar.get_width()/2, value+.25, f'{value:.2f}', ha='center', fontsize=8)
    axes[0].axhline(40., color='#333333', ls='--', lw=1, label='Unchanged 40 mrad threshold')
    axes[0].set_ylabel('Worst joint error in final 2 s (mrad)')
    axes[0].set_ylim(0, max(44., float(np.nanmax(np.r_[before, after]))+4))
    axes[0].legend(loc='lower left', fontsize=9)
    delta = after-before
    axes[1].bar(x, delta, color=['#24758b' if v <= 0 else '#b75242' for v in delta], width=.5)
    axes[1].axhline(0, color='#333333', lw=1)
    axes[1].set_ylabel('Change (mrad)\nNegative is lower error')
    axes[1].set_xticks(x, names)
    axes[1].set_xlabel('Declared rod offsets (mm); case labels retained')
    axes[1].tick_params(axis='x', rotation=15)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    successful = lambda key: sum(r[key] for r in rows)
    fig.suptitle(f'Complete-cycle development cases: {successful("before_success")}/{len(rows)} to '
                 f'{successful("after_success")}/{len(rows)} accepted\n'
                 'Joint error shown; task acceptance also requires pose, height, speed and physical limits', fontsize=12)
    fig.savefig(a.comparison/'closure_outcomes.png', dpi=180)
    fig.savefig(a.comparison/'closure_outcomes.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
