"""Run calibration, gated PPO and full-cycle evaluation without supervision."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    parent = 'outputs/closure_foreaft_fullcycle_20261024_v1'
    grid = 'outputs/closure_foreaft_grid_20261024_v1'
    seed = 'outputs/closure_combined_training_seed_20261024_v1'
    shards = 'outputs/landing_window_worker_shards_20261024_v1'
    window = 'outputs/landing_window_foreaft_seed_gate_20261024_v1'
    workers = 'outputs/landing_window_worker_gate_20261024_v1'
    dataset, critic, training = (out / n for n in ('dataset', 'critic', 'training'))
    stages = [
        ('collect', 'tools.collect_landing_window_critic_data',
         ['--parent', parent, '--seed', seed, '--shards', shards,
          '--window-gate', window, '--worker-gate', workers, '--output', dataset], dataset),
        ('critic', 'tools.warmstart_landing_window_critic',
         ['--seed', seed, '--dataset', dataset, '--output', critic], critic),
        ('train_and_evaluate', 'tools.train_landing_window_curriculum',
         ['--parent', parent, '--grid', grid, '--critic', critic, '--dataset', dataset,
          '--shards', shards, '--window-gate', window, '--worker-gate', workers,
          '--steps', '98304', '--output', training], training),
    ]
    status = dict(pid=os.getpid(), started_at=datetime.now().astimezone().isoformat(),
                  state='starting', completed_stages=[], requested_steps=98304)

    def save():
        status['updated_at'] = datetime.now().astimezone().isoformat()
        temp = out / 'status.tmp'
        temp.write_text(json.dumps(status, indent=2), encoding='utf-8')
        temp.replace(out / 'status.json')

    save()
    try:
        for name, module, options, result_dir in stages:
            if (out / 'STOP_REQUESTED').exists():
                status['state'] = 'stopped_before_next_stage'
                save()
                return
            command = [sys.executable, '-u', '-m', module, *map(str, options)]
            status.update(state='running', stage=name, command=command)
            save()
            with (out / f'{name}.log').open('w', encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=root, stdout=log,
                                           stderr=subprocess.STDOUT)
                status['child_pid'] = process.pid
                save()
                code = process.wait()
            if code:
                raise RuntimeError(f'{name} exited with code {code}; see {name}.log')
            if (result_dir / 'stopped_summary.json').exists():
                status['state'] = 'stopped_by_request'
                save()
                return
            result = json.loads((result_dir / 'summary.json').read_text())
            if name in ('collect', 'critic') and not result.get('passed'):
                raise RuntimeError(f'{name} validation failed; later stages not started')
            if name == 'train_and_evaluate':
                assert not result['evaluation_pending']
                status['successes'] = sum(r['success'] for r in result['after'])
                status['cases'] = len(result['after'])
            status['completed_stages'].append(name)
            save()
        status['state'] = 'completed'
        save()
    except Exception:
        status.update(state='failed', error=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
