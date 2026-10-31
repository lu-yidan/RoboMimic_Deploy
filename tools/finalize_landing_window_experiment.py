"""Finish a declared run's comparisons, regressions and telemetry in order."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from tools.compare_audited_closure_candidates import load_run, sha


def main():
    p = argparse.ArgumentParser()
    for name in ('training', 'baseline', 'grid-baseline', 'geometry-baseline', 'geometry-protocol', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    a.output.mkdir(exist_ok=False)
    status = dict(state='starting', completed_stages=[], source_sha256=sha(__file__))
    (a.output/'runner_source.py').write_bytes(Path(__file__).read_bytes())

    def save():
        status['updated_at'] = datetime.now().astimezone().isoformat()
        temp = a.output/'status.tmp'
        temp.write_text(json.dumps(status, indent=2))
        temp.replace(a.output/'status.json')

    def wait_for(directory, marker, label):
        status.update(state='waiting', stage=label)
        save()
        deadline = time.monotonic()+6*3600
        while not (directory/marker).exists():
            if (a.output/'STOP_REQUESTED').exists():
                raise RuntimeError('Stop requested before next evaluation stage')
            if any((directory/name).exists() for name in ('interrupted_status.json', 'stopped_summary.json')):
                raise RuntimeError(f'Upstream task interrupted: {directory}')
            if time.monotonic() > deadline:
                raise TimeoutError(f'Upstream result not ready: {directory/marker}')
            time.sleep(15)

    def run(label, module, options, python=sys.executable):
        if (a.output/'STOP_REQUESTED').exists():
            raise RuntimeError('Stop requested before next evaluation stage')
        command = [str(python), '-u', '-m', module, *map(str, options)]
        status.update(state='running', stage=label, command=command)
        save()
        with (a.output/f'{label}.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
        status['completed_stages'].append(label)
        save()

    try:
        wait_for(a.training, 'adaptive_reference_audit.json', 'waiting_for_full_cycle_training_audits')
        trained = load_run(a.training)
        for name in ('adaptive_reference_audit.json', 'controller_modifier_audit.json', 'composite_prefix_audit.json'):
            assert json.loads((a.training/name).read_text())['passed']
        run('eight_case_comparison', 'tools.compare_audited_closure_candidates',
            ['--before', a.baseline, '--after', a.training, '--output', a.output/'eight_case_comparison'])
        if trained['rows']['foot_z_plus15']['complete']:
            telemetry = a.output/'telemetry'
            run('telemetry', 'tools.evaluate_closure_foreaft_telemetry',
                ['--experiment', a.training, '--output', telemetry, '--cases', 'foot_z_plus15', '--workers', '1', '--verify-recorded'])
            run('telemetry_audit', 'tools.audit_physics_telemetry', [telemetry])
            physics = a.output/'physics_comparison'
            run('physics_comparison', 'tools.compare_landing_training_telemetry',
                ['--original', 'outputs/closure_foreaft_telemetry_20261024_v1', '--modified', telemetry, '--output', physics])
            run('physics_plot', 'tools.plot_landing_training_telemetry', [physics],
                python=root.parents[1]/'.venv/Scripts/python.exe')
        else:
            status['telemetry_skipped'] = 'Highest case ended before full cycle; final-two-second comparison is inapplicable'
            save()
        run('animation', 'tools.render_closure_training_comparison',
            ['--before', a.baseline, '--after', a.training, '--output', a.output/'video',
             '--before-label', '固定参考基线', '--after-label', '落脚窗口 PPO 训练后', '--speed', '4'])
        run('old_grid', 'tools.evaluate_landing_controller_grid',
            ['--candidate', a.training, '--grid-parent', a.grid_baseline, '--output', a.output/'grid'])
        wait_for(a.geometry_baseline, 'geometry_result.json', 'waiting_for_geometry_baseline')
        run('new_geometries', 'tools.evaluate_landing_geometry_holdout',
            ['--protocol', a.geometry_protocol, '--candidate', a.training,
             '--baseline-result', a.geometry_baseline, '--output', a.output/'geometry', '--workers', '2'])
        status.update(state='completed', automatic_promotion=False,
                      training_audit_sha256=trained['audit_sha256'])
        save()
    except Exception:
        status.update(state='failed', error=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
