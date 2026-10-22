"""Plot audited case results without hiding failed or incomplete trajectories."""
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
    source = args.experiment/'case_comparison.json'
    report = json.loads(source.read_text())
    audit_file = args.experiment/'independent_training_audit.json'
    if hashlib.sha256(audit_file.read_bytes()).hexdigest() != report['audit_sha256']:
        raise ValueError('Audit changed after comparison was generated')
    cases = report['cases']
    x = np.arange(len(cases))
    width = .36
    fig, axes = plt.subplots(3, 1, figsize=(max(9, len(cases)*1.6), 10), sharex=True)
    colors = ['#8799aa', '#227f77']
    for index, label in enumerate(('zero', 'after')):
        offset = x+(index-.5)*width
        legend = 'Zero residual' if label == 'zero' else f"PPO: {report['actual_steps']:,} total steps"
        accepted = [int(row[label]['accepted']) for row in cases]
        bars = axes[0].bar(offset, accepted, width, color=colors[index], label=legend)
        for position, passed in zip(offset, accepted):
            axes[0].text(position, passed+.06, 'PASS' if passed else 'FAIL',
                         ha='center', fontsize=9, color=colors[index])
        duration = [row[label]['simulation_time_s'] for row in cases]
        bars = axes[1].bar(offset, duration, width, color=colors[index])
        axes[1].bar_label(bars, fmt='%.1f', padding=3, fontsize=9)
        for position, row in zip(offset, cases):
            checks = row[label]['endpoint_checks']
            errors = [item['effector_error_m'] for item in checks]
            if not checks or any(error is None or item['sample_count'] < 99 for error, item in zip(errors, checks)):
                axes[2].text(position, .5, 'N/A', ha='center', fontsize=9, color=colors[index])
                continue
            value = max(errors)*1000
            bar = axes[2].bar(position, value, width, color=colors[index])
            axes[2].bar_label(bar, fmt='%.2f', padding=3, fontsize=9)
    axes[0].set_ylim(0, 1.6)
    axes[0].set_yticks([0, 1], ['Fail', 'Pass'])
    axes[0].set_ylabel('Independent acceptance')
    axes[0].legend(loc='upper center', ncol=2, frameon=False)
    axes[1].set_ylabel('Simulated duration (s)')
    axes[1].set_ylim(0, max(row[label]['simulation_time_s'] for row in cases for label in ('zero', 'after'))*1.2)
    axes[2].axhline(10, color='#b33d40', linestyle='--', linewidth=1, label='10 mm acceptance limit')
    axes[2].set_ylabel('Final 2 s peak effector error (mm)')
    axes[2].set_ylim(bottom=0)
    axes[2].margins(y=.2)
    axes[2].legend(frameon=False, loc='upper right')
    names = [row['case'].replace('foot_', 'foot bar\n').replace('hand_', 'hand bar\n')
             .replace('minus', ' -').replace('plus', ' +').replace('_', ' ')
             + (' mm' if row['case'] != 'nominal' else '')
             + ('\n(train case)' if row['in_training_set'] else '\n(evaluation only)') for row in cases]
    axes[2].set_xticks(x, names, fontsize=9)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle(f"Full-cycle residual PPO | zero {report['zero_accepted']}/{len(cases)}, trained {report['after_accepted']}/{len(cases)} accepted", fontsize=14)
    fig.text(.5, .018, 'One seed; declared geometries, not blind tests. N/A: no complete final window. Endpoint error alone does not imply success.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .96])
    for extension in ('png', 'pdf'):
        fig.savefig(args.experiment/f'audited_case_comparison.{extension}', dpi=160)
    plt.close(fig)
    (args.experiment/'audited_case_comparison_figure.json').write_text(json.dumps(dict(
        input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
