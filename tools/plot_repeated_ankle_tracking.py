"""Compare saved left-ankle tracking at the second right-foot loading event."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    # G1 29-joint order: left ankle pitch is joint index 4, after free joint qpos.
    index = 11
    runs = [('Original', 'clearance_two_cycle_baseline_v1'), ('Contact from transfer', 'contact_frame_per_cycle_two_v1'), ('Early contact', 'early_contact_two_cycle_v1')]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    values = []
    for label, folder in runs:
        run = ROOT / 'outputs' / folder / 'nominal'
        z = dict(np.load(run / 'trajectory.npz'))
        audit = json.loads((run / 'joint_limit_audit.json').read_text())
        t = z['time_s']; q = z['qpos'][:, index]
        selected = (t >= 145) & (t <= 165)
        axes[0].plot(t[selected], q[selected], label=label)
        axes[1].plot(t[selected], .5236 - q[selected], label=label)
        values.append(dict(label=label, global_margin_rad=audit['minimum_margin_rad'], position_passed=audit['passed']))
    ref = dict(np.load(ROOT / 'outputs/early_contact_two_cycle_v1/nominal/reference.npz'))
    selected = (ref['time'] >= 145) & (ref['time'] <= 165)
    axes[0].plot(ref['time'][selected], ref['qpos'][selected, index], '--', color='gray', label='Common reference')
    axes[0].axhline(.5236, color='black', linestyle=':', label='Native upper bound')
    axes[1].axhline(0, color='black', linestyle=':')
    axes[0].set_ylabel('Left ankle pitch (rad)')
    axes[1].set_ylabel('Upper-bound margin (rad)')
    axes[1].set_xlabel('Continuous simulation time (s)')
    for ax in axes:
        ax.axvline(158.28, color='gray', alpha=.4)
        ax.grid(alpha=.25)
    axes[0].legend(loc='best', fontsize=9)
    fig.suptitle('Second-cycle ankle tracking: saved 50 Hz physics states\nNominal geometry; exact-contact control; not a learned policy')
    out = ROOT / 'outputs/research_progress_20261017'
    for extension in ['png', 'pdf']:
        fig.savefig(out / ('repeated_ankle_tracking_v1.' + extension), dpi=160)
    (out / 'repeated_ankle_tracking_v1.json').write_text(json.dumps(values, indent=2))


if __name__ == '__main__':
    main()
