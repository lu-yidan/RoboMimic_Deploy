"""Compare an audited development intervention with its full-rate baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def series(directory, report):
    assert report['telemetry_sha256'] == sha(directory/'physics_telemetry.npz')
    assert report['trajectory_sha256'] == sha(directory/'trajectory.npz')
    with np.load(directory/'physics_telemetry.npz') as data:
        return {key: data[key] for key in data.files}


def metrics(data, mask, groups):
    index = np.flatnonzero(mask)
    assert len(index)
    acceleration = np.linalg.norm(data['solver_qacc'][:, :3], axis=1)
    force = np.linalg.norm(data['group_force_world_n'], axis=2)
    peak = index[np.argmax(acceleration[index])]
    return dict(samples=len(index), start_s=float(data['time_s'][index[0]]),
                end_s=float(data['time_s'][index[-1]]),
                peak_root_acceleration_m_s2=float(acceleration[mask].max()),
                peak_root_acceleration_time_s=float(data['time_s'][peak]),
                peak_joint_acceleration_rad_s2=float(abs(data['solver_qacc'][mask, 6:]).max()),
                peak_support_net_force_n={group: float(force[mask, j].max())
                                          for j, group in enumerate(groups)},
                peak_hook_net_force_n={group: float(force[mask, j].max())
                                       for j, group in enumerate(groups) if 'hook' in group})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--candidate-audit', default='hook_admission_audit.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    baseline, candidate = args.baseline.resolve(), args.candidate.resolve()
    ba = read(baseline/'telemetry_audit.json')
    if Path(args.candidate_audit).name!=args.candidate_audit:
        raise ValueError('Candidate audit must be a local filename')
    candidate_audit=candidate/args.candidate_audit
    ca = read(candidate_audit)
    record = read(candidate/'hook_admission_record.json')
    protocol = read(candidate/'hook_admission_protocol.json')
    bm, cm = read(baseline/'manifest.json'), read(candidate/'manifest.json')
    assert ba['passed'] and ca['trace_checks_passed']
    assert ca['manifest_sha256'] == sha(candidate/'manifest.json')
    assert ca['task_audit_sha256'] == sha(candidate/'v2_multicycle_audit.json')
    assert record['groups'] == ba['groups']
    assert protocol['baseline_trajectory_sha256'] == ba['trajectory_sha256']
    assert all(bm[key] == cm[key] for key in ('case', 'cycles', 'plan_sha256',
               'checkpoint_sha256', 'prefix_checkpoint_sha256', 'offsets'))
    data = {'baseline': series(baseline, ba), 'candidate': series(candidate, ca)}
    summaries = {'baseline': read(baseline/'summary.json'),
                 'candidate': read(candidate/'summary.json')}
    ends = np.load(candidate/'reference.npz')['cycle_ends_s']
    times = data['baseline']['time_s']
    assert np.array_equal(times, data['candidate']['time_s'])
    first = times <= float(ends[0])+1e-7
    first_exact = {key: bool(np.array_equal(data['baseline'][key][first],
                                          data['candidate'][key][first]))
                   for key in data['baseline']}
    assert all(first_exact.values())
    cycles = []
    for i, end in enumerate(ends):
        start = float(ends[i-1]) if i else 0.
        mask = (times > start+1e-7) & (times <= float(end)+1e-7)
        cycles.append(dict(cycle=i+1, **{name: dict(metrics(value, mask, ba['groups']),
                            endpoint_closure_mrad=summaries[name]['cycle_results'][i]['closure_mrad'])
                                        for name, value in data.items()}))
    events = []
    for event in ca['events']:
        # Include the declared confirmation timeout and ramp, so a delayed peak
        # cannot disappear merely because a narrow plotting window excludes it.
        lo = event['start_s']-.2
        hi = event['start_s']+protocol['parameters']['timeout_seconds']+protocol['parameters']['ramp_seconds']
        mask = (times >= lo) & (times <= hi)
        events.append(dict(**event, window_s=[lo, hi],
                           metrics={name: metrics(value, mask, ba['groups'])
                                    for name, value in data.items()}))
    report = dict(case=cm['case'], task_passed=ca['task_passed'],
                  first_cycle_full_rate_exact=first_exact, cycles=cycles, events=events,
                  baseline_audit_sha256=sha(baseline/'telemetry_audit.json'),
                  candidate_audit_sha256=sha(candidate_audit),candidate_audit_name=args.candidate_audit,
                  source_sha256=sha(__file__), new_training_steps=0,
                  scope='Paired complete-run development comparison at 2000 Hz. Same reference, policy and case; the controller intervention changes later-cycle hook loading. No hardware or held-out generalization claim, no automatic adoption, and no new pass threshold derived from these peaks.')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    (args.output/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    if not args.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # Keep the transient figure readable on eight-cycle runs. All fourteen
        # event windows remain in JSON; the plot shows the first changed cycle.
        plotted_events = events[:2]
        fig, axes = plt.subplots(3, len(plotted_events), figsize=(6*len(plotted_events), 9),
                                 squeeze=False, layout='constrained')
        for column, event in enumerate(plotted_events):
            group = event['group']; j = ba['groups'].index(group)
            # Index is declared for this frozen 29-DoF G1 package only.
            motor = 15 if group == 'left_hook' else 22
            lo, hi = event['window_s']; mask = (times >= lo) & (times <= hi)
            for name, value in data.items():
                local = (times[mask]-event['start_s'])*1000
                values = [np.linalg.norm(value['group_force_world_n'][mask, j], axis=1),
                          np.linalg.norm(value['solver_qacc'][mask, :3], axis=1),
                          value['motor_torque_nm'][mask, motor]]
                for row, y in enumerate(values):
                    axes[row, column].plot(local, y, label=name, linewidth=1)
            for row, label in enumerate(('Net hook force (N)', 'Root acceleration (m/s²)',
                                          'Shoulder pitch torque (Nm)')):
                ax = axes[row, column]; ax.set_ylabel(label); ax.grid(alpha=.2)
                ax.axvline(0, color='gray', linestyle='--', alpha=.5)
                ax.axvline((event['confirmed_s']-event['start_s'])*1000,
                           color='green', linestyle=':', alpha=.7)
                ax.legend(fontsize=8)
            axes[0, column].set_title(f"Cycle {event['cycle']}: {group}")
            axes[-1, column].set_xlabel('Time from planned load activation (ms)')
        fig.suptitle(f"{cm['case']}: contact-confirmed loading candidate, frozen policy\n"
                     'Full 2000 Hz comparison; dashed: activation, dotted: confirmation')
        fig.savefig(args.output/'hook_admission_comparison.png', dpi=160)
        fig.savefig(args.output/'hook_admission_comparison.pdf')
        plt.close(fig)
        if len(cycles)>2:
            fig, axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
            fields=[('peak_root_acceleration_m_s2','Root acceleration peak (m/s²)',None),
                    ('peak_joint_acceleration_rad_s2','Joint acceleration peak (rad/s²)',None),
                    ('peak_support_net_force_n','Left hook net force peak (N)','left_hook'),
                    ('peak_support_net_force_n','Right hook net force peak (N)','right_hook'),
                    ('endpoint_closure_mrad','Endpoint closure (mrad)',None),
                    ('peak_support_net_force_n','Maximum foot net force peak (N)','feet')]
            for ax,(field,label,group) in zip(axes.flat,fields):
                for name in data:
                    values=[row[name][field] for row in cycles]
                    if group=='feet':values=[max(v['left_foot'],v['right_foot']) for v in values]
                    elif group:values=[v[group] for v in values]
                    ax.plot([row['cycle'] for row in cycles],values,marker='o',label=name)
                ax.set(xlabel='Cycle',ylabel=label);ax.grid(alpha=.2);ax.legend(fontsize=8)
            axes[2,0].axhline(35,color='gray',linestyle='--',alpha=.5)
            fig.suptitle(f"{cm['case']}: complete-cycle 2000 Hz peaks and endpoint quality\n"
                         'First cycle unchanged; one development case, no hardware claim')
            fig.savefig(args.output/'whole_run_comparison.png',dpi=160)
            fig.savefig(args.output/'whole_run_comparison.pdf');plt.close(fig)
    print(json.dumps(dict(case=cm['case'], task_passed=ca['task_passed'], cycles=cycles)), flush=True)


if __name__ == '__main__':
    main()
