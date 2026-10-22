"""Cycle endpoint errors with explicit unexecuted cycles; not success rates."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('comparison', type=Path)
    args = parser.parse_args()
    report = json.loads(args.comparison.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={'width_ratios': [2.2, 1]})
    colors = ['#8899aa', '#157f77']
    names = ['Zero residual', 'Learned residual (8,192 steps)']
    for row, color, name in zip(report['records'], colors, names):
        cycles = row['executed_cycles']
        axes[0].plot([r['cycle'] for r in cycles], [1000*r['effector_error_m'] for r in cycles],
                     marker='o', color=color, label=name)
        for r in cycles:
            if not r['passed']:
                axes[0].scatter(r['cycle'],1000*r['effector_error_m'],marker='x',s=110,color='#b33d40',zorder=5)
        if row['unexecuted_cycles']:
            axes[0].text(7.25, 13.5, 'Zero: cycles 7–8\nnot executed', ha='center', color='#586776',fontsize=9)
    axes[0].axhline(10, linestyle='--', color='#b33d40', linewidth=1)
    axes[0].text(1.1,10.35,'10 mm strict endpoint limit',color='#b33d40',fontsize=9)
    axes[0].set(xlim=(.7,8.3),ylim=(0,17),xticks=range(1,9),xlabel='Cycle',ylabel='Final 2 s peak effector error (mm)')
    axes[0].legend(loc='lower left',frameon=False,fontsize=9)
    bars=axes[1].bar(['Zero','Learned'],[r['measured_root_rise_m'] for r in report['records']],color=colors,width=.6)
    axes[1].bar_label(bars,labels=[f"{r['measured_root_rise_m']:.3f} m\n{r['time_s']:.2f} s" for r in report['records']],padding=5,fontsize=10)
    axes[1].set(ylim=(0,2.6),ylabel='Measured root rise (m)')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Continuous nominal climb: learned 8/8 cycles accepted; zero fails at cycle 6',fontsize=12)
    fig.text(.5,.025,'One deterministic pair; scripted reference and ideal contact feedback. Not multi-seed robustness or autonomous climbing.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.07,1,.94])
    out=args.comparison.parent
    for suffix in ('png','pdf'):
        fig.savefig(out/f'long_horizon_comparison.{suffix}',dpi=170)
    plt.close(fig)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    (out/'figure_manifest.json').write_text(json.dumps(dict(input_sha256=sha(args.comparison),source_sha256=sha(__file__),
        files={s:sha(out/f'long_horizon_comparison.{s}') for s in ('png','pdf')}),indent=2))
    (out/'plot_source.py').write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
    main()
