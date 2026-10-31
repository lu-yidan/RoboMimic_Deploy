"""Local bounded constant-reference refinement, explicitly not PPO training."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linprog
from training.seeded_reference_delta_env import SeededReferenceDeltaEnv
from tools.compare_audited_closure_candidates import load_run, sha


def evaluate(payload):
    rank, protocol, commands, output = payload
    torch.set_num_threads(1)
    env = SeededReferenceDeltaEnv(Path(protocol['worker_shards'])/f'worker_{rank}',
        controller_modifier=protocol['controller_modifier'], teacher_checkpoint=protocol['teacher_checkpoint'],
        teacher_sha256=protocol['teacher_sha256'], seed_command=protocol['seed_command'],
        delta_scale=protocol['delta_scale'], weight=.1)
    case = [case for case in env.cases if case in protocol['active_cases']]
    assert len(case) == 1
    case = case[0]
    rows = []
    try:
        for label, delta in commands:
            delta = np.asarray(delta, dtype=np.float64)
            assert delta.shape == (5,) and abs(delta).max() <= 1.
            env.reset(options={'case': case})
            base = env.env.unwrapped
            physical = base.env
            q, v, times = [physical.data.qpos.copy()], [physical.data.qvel.copy()], [float(physical.data.time)]
            while True:
                _, _, done, truncated, info = env.step(delta)
                q.append(physical.data.qpos.copy())
                v.append(physical.data.qvel.copy())
                times.append(float(physical.data.time))
                if done or truncated:
                    break
            q, v, times = np.asarray(q), np.asarray(v), np.asarray(times)
            assert all(np.isfinite(array).all() for array in (q, v, times))
            tail = times > 111.66
            complete = bool(len(q) == 701 and tail.sum() == 101)
            joint = q[tail][:, physical.qadr]-physical.initial[physical.qadr]
            height = q[tail, 2]-physical.initial[2]-.25
            signed = np.r_[joint.reshape(-1)/.001, -joint.reshape(-1)/.001, height/.0001, -height/.0001]
            bounds = np.r_[np.full(2*joint.size, 40.), np.full(2*len(height), 100.)]
            if label == 'zero':
                with np.load(Path(protocol['parent'])/f'after_{case}.npz') as old:
                    assert np.max(abs(old['qpos'][4983:]-q)) < 1e-10
                    assert np.max(abs(old['qvel'][4983:]-v)) < 1e-8
            path = Path(output)/f'{label}_rank{rank}.npz'
            np.savez_compressed(path, qpos=q, qvel=v, time_s=times, delta=delta, signed=signed, bounds=bounds)
            row = dict(case=case, label=label, delta=delta.tolist(), complete=complete,
                success=bool(info['success']), reason=info['reason'], info=info,
                closure_rad=float(abs(joint).max()) if len(joint) else None,
                height_error_m=float(abs(height).max()) if len(height) else None,
                minimum_margin=float((bounds-signed).min()) if complete else -1e9,
                trajectory_sha256=sha(path), file=path.name)
            path.with_suffix('.json').write_text(json.dumps(row, indent=2))
            rows.append(row)
            print(json.dumps({key: row[key] for key in ('case', 'label', 'success', 'minimum_margin')}), flush=True)
    finally:
        env.close()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, choices=(1, 2, 4), default=2)
    a = p.parse_args()
    protocol = json.loads((a.seed/'protocol.json').read_text())
    seed = json.loads((a.seed/'summary.json').read_text())
    assert seed['passed'] and seed['protocol_sha256'] == sha(a.seed/'protocol.json')
    parent = load_run(Path(protocol['parent']))
    assert parent['audit_sha256'] == protocol['parent_audit_sha256']
    for source, digest in protocol['sources'].items():
        assert sha(source) == digest
    assert sha(protocol['teacher_checkpoint']) == protocol['teacher_sha256']
    assert sha(Path(protocol['worker_shards'])/'summary.json') == protocol['shards_summary_sha256']
    active = {case: offsets for case, offsets in protocol['cases'].items() if offsets['rung_01'][2] > .0125}
    assert len(active) == 4
    center = np.asarray(protocol['seed_command'])
    assert center[0] == 1. and center[4] == -1.
    epsilon = .25
    probes = [('zero', np.zeros(5).tolist())]
    axis_labels = []
    for axis in range(5):
        labels, offsets = [], []
        for sign in (-1, 1):
            step = sign*epsilon
            if (axis == 0 and sign == 1) or (axis == 4 and sign == -1):
                labels.append('zero'); offsets.append(0.)
            else:
                delta = np.zeros(5); delta[axis] = step
                label = f'axis{axis}_{sign:+d}'
                probes.append((label, delta.tolist())); labels.append(label); offsets.append(step)
        axis_labels.append(dict(labels=labels, offsets=offsets))
    assert len(probes) == 9
    protocol.update(active_cases=active, probe_delta=epsilon, probes=probes, axis_labels=axis_labels,
        proposal_fractions=[.25, .5, .75, 1.],
        delta_bounds=[[-1., 0.], [-1., 1.], [-1., 1.], [-1., 1.], [0., 1.]],
        workers=a.workers, source_sha256=sha(__file__),
        scope='36 local 14-second constant-delta probes and at most 16 proposal checks on four known cases. '
              'Original reference/delta limits, teacher and acceptance unchanged. Clipping-aware one-sided derivatives '
              'at saturated coordinates. Linearized margin is a proposal, not a certificate. No actor training or automatic promotion.')
    a.output.mkdir(exist_ok=False)
    (a.output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    (a.output/'search_source.py').write_bytes(Path(__file__).read_bytes())
    rows = []
    def batch(commands):
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(evaluate, (rank, protocol, commands, str(a.output.resolve()))) for rank in range(4)]
            for future in as_completed(futures):
                rows.extend(future.result())
    batch(probes)
    assert len(rows) == 36
    proposal = None
    if all(row['complete'] for row in rows):
        cases = sorted(active)
        lookup = {(row['case'], row['label']): row for row in rows}
        def vector(label, field):
            values = []
            for case in cases:
                row = lookup[case, label]
                assert sha(a.output/row['file']) == row['trajectory_sha256']
                with np.load(a.output/row['file']) as z:
                    values.append(z[field].copy())
            return np.concatenate(values)
        base, bounds = vector('zero', 'signed'), vector('zero', 'bounds')
        jacobian = np.stack([(vector(axis['labels'][1], 'signed')-vector(axis['labels'][0], 'signed')) /
                            (axis['offsets'][1]-axis['offsets'][0]) for axis in axis_labels], axis=1)
        solved = linprog([0, 0, 0, 0, 0, -1], A_ub=np.c_[jacobian, np.ones(len(base))],
            b_ub=bounds-base, bounds=protocol['delta_bounds']+[(None, None)], method='highs')
        proposal = dict(success=bool(solved.success), message=solved.message)
        if solved.success:
            proposal.update(delta=solved.x[:5].tolist(), predicted_minimum_margin=float(solved.x[5]))
            (a.output/'proposal.json').write_text(json.dumps(proposal, indent=2))
            batch([(f'proposal_{fraction:g}', (solved.x[:5]*fraction).tolist()) for fraction in protocol['proposal_fractions']])
    grouped = []
    for label in dict.fromkeys(row['label'] for row in rows):
        group = [row for row in rows if row['label'] == label]
        assert len(group) == 4
        grouped.append(dict(label=label, delta=group[0]['delta'], successes=sum(row['success'] for row in group),
            highest_preserved=next(row['success'] for row in group if row['case'] == 'foot_z_plus15'),
            minimum_margin=min(row['minimum_margin'] for row in group)))
    best = max(grouped, key=lambda row: (row['highest_preserved'], row['successes'], row['minimum_margin']))
    result = dict(completed=True, rows=rows, groups=grouped, best=best, proposal=proposal,
        protocol_sha256=sha(a.output/'protocol.json'), scope=protocol['scope'])
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(completed=True, best=best, proposal=proposal)), flush=True)


if __name__ == '__main__':
    main()
