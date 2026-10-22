"""Illustrate bounded scalar reward sensitivity; not an action feasibility test."""
import json
from pathlib import Path
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = Path('outputs/closure_reward_signal_figure_v1')
    output.mkdir(exist_ok=False)
    audit_path = Path('outputs/support_retarget_calibration_v1/foot_z_plus15/independent_audit.json')
    # Metric locations come from the already audited training zero baseline.
    training_audit = Path('outputs/closure_margin_control98304_v1/independent_training_audit.json')
    audit = json.loads(training_audit.read_text())
    assert audit['passed']
    row = next(r for r in audit['trajectories'] if r['label'] == 'zero' and r['case'] == 'foot_z_plus15')
    endpoint = row['independent_endpoints'][0]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    specs = [(np.linspace(28, 60, 401), 32, 8, 'Maximum joint closure (mrad)'),
             (np.linspace(0, 16, 401), 6, 4, 'Absolute root-rise error (mm)')]
    for ax, (x, hinge, scale, label) in zip(axes, specs):
        normalized = np.maximum((x-hinge)/scale, 0)
        capped = .01*np.minimum(normalized, 1)**2
        smooth = .01*normalized**2/(1+normalized**2)
        ax.plot(x, capped, color='#b46b45', lw=2, label='Capped square (original experiment)')
        ax.plot(x, smooth, color='#327c81', lw=2, label='Bounded rational (follow-up candidate)')
        ax.axvline(hinge+scale, ls='--', color='#777777', lw=1, label='Strict endpoint limit')
        ax.set(xlabel=label, ylabel='Per-term penalty at full weight', ylim=(-.0004, .0107))
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=7.8, loc='upper left')
    fig.suptitle('Terminal reward sensitivity: equal 0.02 total cost cap', fontsize=13)
    fig.text(.5, .025, 'Scalar metric perturbations only. Unchanged physical task gates; no claim that actions can realize these improvements.', ha='center', fontsize=8.4)
    fig.tight_layout(rect=(0, .07, 1, .94))
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'reward_signal.{extension}', dpi=180)
    plt.close(fig)
    record = dict(training_audit_sha256=digest(training_audit), source_sha256=digest(__file__),
                  endpoint_reference=endpoint, parameters=dict(total_weight=.02, joint_hinge_rad=.032, joint_scale_rad=.008, height_hinge_m=.006, height_scale_m=.004),
                  scope='Formula illustration, not a policy result, action gradient, or physical feasibility claim.')
    (output / 'figure_manifest.json').write_text(json.dumps(record, indent=2))
    (output / 'figure_source.py').write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
    main()
