"""Plot audited single-cycle training changes and separate frozen-policy transfer."""
import argparse
import json
from pathlib import Path
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    comparison = Path('outputs/calibrated_landing_ppo32768_compare16_20261027_v1/comparison.json')
    run = Path('outputs/boundary_combined_eight_handoff_20261027_v1')
    c = json.loads(comparison.read_text())
    physical = json.loads((run/'independent_audit.json').read_text())
    policy = json.loads((run/'policy_transfer_audit.json').read_text())
    summary = json.loads((run/'summary.json').read_text())
    assert physical['passed'] and policy['passed'] and summary['passed']
    assert sha(run/'trajectory.npz') == physical['trajectory_sha256'] == policy['trajectory_sha256']
    assert not c['lost'] and len(c['rows']) == 16
    rows = c['rows']
    x = np.arange(len(rows))
    with np.load(run/'trajectory.npz') as z:
        t = z['time_s'].copy()
        rise = z['qpos'][:, 2]-z['qpos'][0, 2]
    args.output.mkdir(exist_ok=False)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout='constrained')
    axes[0].scatter(x-.08, [r['before_closure_mrad'] for r in rows], label='Fixed calibration, original PPO', color='#64748b', s=30)
    axes[0].scatter(x+.08, [r['after_closure_mrad'] for r in rows], label='Same calibration, +32,768 PPO steps', color='#007d80', s=30)
    axes[0].axhline(40, color='#ba3838', linestyle='--', label='Unchanged 40 mrad limit')
    axes[0].set_xticks(x, [r['case'].replace('interior_', 'I').replace('boundary_', 'B') for r in rows])
    axes[0].set_ylabel('Maximum closure error (mrad)')
    axes[0].set_title('Single-cycle development suite: 16/16 retained; B07 margin remains small')
    axes[0].legend(fontsize=8, loc='lower right')
    axes[0].grid(axis='y', alpha=.2)
    axes[1].plot(t[::10], rise[::10], color='#27669e', linewidth=1.5)
    for endpoint in summary['cycle_results']:
        end = endpoint['time_s']
        i = np.argmin(abs(t-end))
        axes[1].scatter(t[i], rise[i], color='#007d80', s=22)
    axes[1].set_xlabel('Continuous simulated time (s)')
    axes[1].set_ylabel('Measured root rise (m)')
    axes[1].set_title('Separate frozen-policy transfer: 8 cycles, 909.28 s, 1.9985 m; no physical resets')
    axes[1].grid(alpha=.2)
    fig.suptitle('G1 closure follow-up: trained single-cycle policy and frozen-policy transfer are distinct', fontsize=12)
    fig.savefig(args.output/'training_and_transfer.png', dpi=170)
    fig.savefig(args.output/'training_and_transfer.pdf')
    plt.close(fig)
    manifest = dict(evidence_sha256={str(p):sha(p) for p in (comparison, run/'summary.json', run/'independent_audit.json', run/'policy_transfer_audit.json')},
                    source_sha256=sha(__file__), scope='Development cases; transfer uses original frozen PPO and scripted reference. No robustness or autonomous perception claim.')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
    main()
