"""Plot audited endpoint margins for known-case reward experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-smooth', action='store_true')
    args = parser.parse_args()
    experiments = [('support_reference_policy_transfer_v1', 'Parent'), ('closure_margin_control98304_v1', 'Zero-cost continuation'), ('closure_margin_penalty98304_v1', 'Capped-cost continuation')]
    if args.include_smooth:
        experiments.append(('closure_smooth_penalty98304_v1', 'Smooth-cost continuation'))
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.6))
    sources, counts = {}, {}
    for name, label in experiments:
        path = Path('outputs')/name
        audit = json.loads((path/'independent_training_audit.json').read_text())
        manifest = json.loads((path/'manifest.json').read_text())
        slack = json.loads((path/'endpoint_slack_diagnostic.json').read_text())
        assert audit['passed'] and sha(path/'independent_training_audit.json') == slack['audit_sha256']
        assert sha(path/'policy.zip') == audit['checkpoint_sha256']
        rows = sorted([r for r in slack['rows'] if r['label']=='after'], key=lambda r: manifest['evaluation_cases'][r['case']]['rung_01'][2])
        assert len(rows) == 5 and all(r['endpoint_available'] for r in rows)
        heights = [1000*manifest['evaluation_cases'][r['case']]['rung_01'][2] for r in rows]
        count = sum(r['accepted'] for r in rows)
        counts[label] = count
        axes[0].plot(heights, [1000*r['signed_slack']['joint_closure_rad'] for r in rows], 'o-', lw=1.7, markersize=4, label=f'{label} ({count}/5)')
        axes[1].plot(heights, [1000*r['signed_slack']['root_rise_m'] for r in rows], 'o-', lw=1.7, markersize=4, label=label)
        sources[str(path/'endpoint_slack_diagnostic.json')] = sha(path/'endpoint_slack_diagnostic.json')
    for ax in axes:
        ax.axhline(0., color='#a44238', ls='--', lw=1.2)
        ax.set(xlabel='Known rod height offset (mm)', xticks=[5, 7.5, 10, 12.5, 15])
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set(ylabel='Joint-closure slack (mrad)', title='40 mrad minus observed maximum')
    axes[1].set(ylabel='Root-rise slack (mm)', title='10 mm minus maximum absolute rise error')
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('Known-case endpoint slack | positive is within this criterion', fontsize=13)
    fig.text(.5, .025, 'Original acceptance also requires contact, tracking, speed and transient constraints. One seed and equal additional budget; no statistical claim.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, .94))
    args.output.mkdir(exist_ok=False)
    for extension in ('png', 'pdf'):
        fig.savefig(args.output/f'endpoint_slack.{extension}', dpi=180)
    plt.close(fig)
    (args.output/'provenance.json').write_text(json.dumps(dict(source_hashes=sources, counts=counts, source_sha256=sha(__file__)), indent=2))
    (args.output/'figure_source.py').write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
    main()
