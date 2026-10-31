"""Verify zero learned reference delta reproduces the audited fixed candidate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from tools.compare_audited_closure_candidates import load_run, sha
from training.seeded_reference_delta_env import SeededReferenceDeltaEnv
from training.held_reference_command import HeldReferenceCommand


def worker(a):
    torch.set_num_threads(1)
    verified = load_run(a.parent)
    m = json.loads((a.parent / 'manifest.json').read_text())
    env = SeededReferenceDeltaEnv(a.shards / f'worker_{a.worker}',
        controller_modifier=m['controller_modifier'], teacher_checkpoint=a.parent/'policy.zip',
        teacher_sha256=verified['checkpoint_sha256'], seed_command=m['adaptive_reference_command'], weight=.1)
    held = HeldReferenceCommand(env, repeat=5, physical_gamma=.9995)
    a.output.mkdir(exist_ok=False)
    records = []
    original = env.step

    def record(action):
        prior = env.last_obs.copy()
        result = original(action)
        base = env.env.unwrapped
        records.append(dict(qpos=base.env.data.qpos.copy(), qvel=base.env.data.qvel.copy(),
                            time_s=float(base.env.data.time), observations=prior,
                            actions=base.previous_action.copy(), torques=base.last_torque.copy(),
                            base_rewards=result[4]['base_reward'], rewards=float(result[1])))
        return result

    env.step = record
    rows = []
    try:
        for case in env.cases:
            records.clear()
            held.reset(options={'case': case})
            base = env.env.unwrapped
            initial = dict(qpos=base.env.data.qpos.copy(), qvel=base.env.data.qvel.copy(),
                           time_s=float(base.env.data.time))
            macro_rewards = []
            while True:
                _, reward, done, truncated, info = held.step(np.zeros(5, dtype=np.float32))
                macro_rewards.append(float(reward))
                if done or truncated:
                    break
            arrays = {key: np.asarray([r[key] for r in records]) for key in records[0]}
            for key in initial:
                arrays[key] = np.concatenate([np.asarray(initial[key])[None], arrays[key]], axis=0)
            start = round(99.66/.02)
            differences = {}
            tolerances = dict(qpos=1e-7, qvel=1e-5, time_s=1e-8, observations=1e-5,
                              actions=1e-5, torques=1e-3, base_rewards=1e-8)
            with np.load(a.parent/f'after_{case}.npz') as old:
                for key in tolerances:
                    reference = old['rewards' if key == 'base_rewards' else key][start:]
                    assert reference.shape == arrays[key].shape
                    differences[key] = float(abs(reference-arrays[key]).max())
            raw_return = sum(.9995**i*r for i, r in enumerate(arrays['rewards']))
            macro_return = sum(held.macro_gamma**i*r for i, r in enumerate(macro_rewards))
            checks = dict(reproduces_fixed_candidate=all(differences[k] <= tolerances[k] for k in tolerances),
                          success_unchanged=bool(info['success']) == verified['rows'][case]['success'],
                          teacher_steps=len(records) == 700, macro_steps=len(macro_rewards) == 140,
                          discounted_return=bool(abs(raw_return-macro_return) < 1e-10))
            path = a.output/f'{case}.npz'
            np.savez_compressed(path, **arrays)
            rows.append(dict(case=case, passed=all(checks.values()), checks=checks,
                             differences=differences, tolerances=tolerances,
                             task_success=bool(info['success']), trajectory_sha256=sha(path)))
            print(json.dumps(rows[-1]), flush=True)
    finally:
        env.close()
    result = dict(passed=all(r['passed'] for r in rows), rows=rows)
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    assert result['passed']


def main():
    p = argparse.ArgumentParser()
    for name in ('parent', 'shards', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--worker', type=int, choices=range(4))
    a = p.parse_args()
    if a.worker is not None:
        worker(a)
        return
    a.output.mkdir(exist_ok=False)
    rows = []
    for rank in range(4):
        output = a.output/f'worker_{rank}'
        with (a.output/f'worker_{rank}.log').open('w') as log:
            subprocess.run([sys.executable, '-u', '-m', 'tools.check_seeded_reference_delta_v2',
                '--parent', str(a.parent), '--shards', str(a.shards), '--output', str(output),
                '--worker', str(rank)], check=True, stdout=log, stderr=subprocess.STDOUT)
        s = json.loads((output/'summary.json').read_text())
        assert s['passed']
        rows.extend(s['rows'])
    assert len(rows) == 8 and len({r['case'] for r in rows}) == 8
    sources = {name: sha(name) for name in ('tools/check_seeded_reference_delta_v2.py',
        'training/seeded_reference_delta_env.py', 'training/held_reference_command.py',
        'training/closure_foreaft_curriculum_env.py', 'training/closure_foreaft_reference.py',
        'training/closure_combined_curriculum_env.py', 'training/closure_curriculum_env_v3.py')}
    result = dict(passed=True, rows=rows, sources=sources,
                  parent_audit_sha256=load_run(a.parent)['audit_sha256'],
                  shards_summary_sha256=sha(a.shards/'summary.json'),
                  scope='Zero delta at 10 Hz reproduces the declared fixed candidate within original cache tolerances. No learned feedback performance or new full-cycle dynamics claim.')
    (a.output/'summary.json').write_text(json.dumps(result, indent=2))
    for name in sources:
        target = a.output/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(name).read_bytes())


if __name__ == '__main__':
    main()
