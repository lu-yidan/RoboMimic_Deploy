"""Compare audited continuous handoff recordings without changing acceptance."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_run(path):
    manifest = json.loads((path/'manifest.json').read_text())
    summary = json.loads((path/'summary.json').read_text())
    audit = json.loads((path/'independent_audit.json').read_text())
    assert sha(path/'trajectory.npz') == summary['trajectory_sha256'] == audit['trajectory_sha256']
    assert sha(path/'reference.npz') == manifest['plan_sha256']
    assert all(sha(path/'source_snapshot'/name) == digest for name, digest in manifest['sources'].items())
    if 'geometry_observation_roles' in manifest:
        policy_audit = json.loads((path/'policy_transfer_audit.json').read_text())
        assert policy_audit['passed'] and policy_audit['trajectory_sha256'] == summary['trajectory_sha256']
    with np.load(path/'trajectory.npz') as z:
        arrays = {name: z[name].copy() for name in ('qpos', 'qvel', 'time_s', 'actions', 'torques')}
    return manifest, summary, audit, arrays


def main():
    p = argparse.ArgumentParser()
    for name in ('before', 'after', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--before-label', default='Classical continuation')
    p.add_argument('--after-label', default='Frozen policy continuation')
    a = p.parse_args()
    paths = (a.before, a.after)
    runs = [read_run(path) for path in paths]
    assert runs[0][0]['plan_sha256'] == runs[1][0]['plan_sha256']
    assert runs[0][0]['offsets'] == runs[1][0]['offsets']
    assert runs[0][0]['fade_seconds'] == runs[1][0]['fade_seconds']
    with np.load(a.before/'reference.npz') as z:
        ends = z['cycle_ends_s'].copy()
    rows = []
    for i in range(len(ends)):
        row = dict(cycle=i+1)
        for label, run in zip(('before', 'after'), runs):
            endpoint = run[2]['cycle_endpoints'][i]
            row.update({label+'_'+key: endpoint[key] for key in
                ('passed', 'sample_count', 'joint_closure_rad', 'effector_error_m', 'tilt_deg', 'linear_speed_m_s')})
        rows.append(row)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    colors = ('#b75242', '#24758b')
    labels = (a.before_label, a.after_label)
    cycles = np.arange(1, len(ends)+1)
    for run, label, color in zip(runs, labels, colors):
        data = run[3]
        axes[0, 0].plot(data['time_s'][::10], data['qpos'][::10, 2]-data['qpos'][0, 2], label=label, color=color)
        for ax, key, scale in ((axes[0, 1], 'joint_closure_rad', 1000),
                               (axes[1, 0], 'effector_error_m', 1000), (axes[1, 1], 'tilt_deg', 1)):
            values = [row[key]*scale if row[key] is not None else np.nan for row in run[2]['cycle_endpoints']]
            ax.plot(cycles, values, 'o-', color=color, label=label)
    axes[0, 0].set(xlabel='Recorded simulation time (s)', ylabel='Root rise (m)')
    for ax, threshold, title in ((axes[0, 1], 40, 'Worst joint closure (mrad)'),
                                 (axes[1, 0], 10, 'Worst effector alignment (mm)'),
                                 (axes[1, 1], 3, 'Worst root tilt (deg)')):
        ax.axhline(threshold, color='#333333', ls='--', lw=1, label='Original acceptance limit')
        ax.set(xlabel='Cycle endpoint', ylabel=title, xticks=cycles)
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Continuous recorded physics on the same reference and scene\n'
                 'Endpoint metrics use the final 2 s; all original acceptance checks remain required.', fontsize=12)
    a.output.mkdir(exist_ok=False)
    fig.savefig(a.output/'cycle_comparison.png', dpi=180)
    fig.savefig(a.output/'cycle_comparison.pdf')
    first_count = round(float(ends[0])/.02)
    first = {}
    for key in runs[0][3]:
        count = first_count+1 if key in ('qpos', 'qvel', 'time_s') else first_count
        first[key] = bool(np.array_equal(runs[0][3][key][:count], runs[1][3][key][:count]))
    result = dict(rows=rows, first_cycle_equal=first, labels=labels,
        physical_passed=[run[2]['passed'] for run in runs],
        final_rise_m=[float(run[3]['qpos'][-1, 2]-run[3]['qpos'][0, 2]) for run in runs],
        summary_sha256=[sha(path/'summary.json') for path in paths],
        audit_sha256=[sha(path/'independent_audit.json') for path in paths], source_sha256=sha(__file__),
        scope='One paired deterministic scenario. Frozen one-cycle policy transfer and/or classical continuation; '
              'no new multicycle training, robustness probability, camera or hardware claim.')
    (a.output/'comparison.json').write_text(json.dumps(result, indent=2))
    with (a.output/'comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(physical_passed=result['physical_passed'], final_rise_m=result['final_rise_m'])))


if __name__ == '__main__':
    main()
