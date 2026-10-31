"""Finish already-authorized feedback validation without promoting the model."""
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
    for name in ('training', 'baseline', 'grid-baseline', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    a.output.mkdir(exist_ok=False)
    (a.output/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    status = dict(state='waiting', stage='full_cycle_feedback_audits', completed_stages=[], source_sha256=sha(__file__))

    def save():
        status['updated_at'] = datetime.now().astimezone().isoformat()
        temporary = a.output/'status.tmp'
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(a.output/'status.json')

    def run(label, module, options, python=sys.executable):
        if (a.output/'STOP_REQUESTED').exists():
            raise RuntimeError('Stop requested before next stage')
        command = [str(python), '-u', '-m', module, *map(str, options)]
        status.update(state='running', stage=label, command=command)
        save()
        with (a.output/f'{label}.log').open('w', encoding='utf-8') as stream:
            subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)
        status['completed_stages'].append(label)
        save()

    try:
        save()
        deadline = time.monotonic()+6*3600
        while not (a.training/'controller_modifier_audit.json').exists():
            if (a.output/'STOP_REQUESTED').exists():
                raise RuntimeError('Stop requested while waiting')
            if any((a.training/name).exists() for name in ('interrupted_status.json', 'stopped_summary.json')):
                raise RuntimeError('Upstream training interrupted')
            if (a.training/'independent_candidate_audit.json').exists():
                if not json.loads((a.training/'independent_candidate_audit.json').read_text())['passed']:
                    raise RuntimeError('Upstream trajectory audit failed')
            if time.monotonic() > deadline:
                raise TimeoutError('Training audits not ready')
            time.sleep(15)
        verified = load_run(a.training)
        for name in ('controller_modifier_audit.json', 'composite_prefix_audit.json'):
            assert json.loads((a.training/name).read_text())['passed']
        run('eight_case_comparison', 'tools.compare_reference_feedback_results',
            ['--before', a.baseline, '--after', a.training, '--output', a.output/'eight_case_comparison'])
        run('reference_effects', 'tools.audit_reference_feedback_effects',
            ['--baseline', a.baseline, '--candidate', a.training, '--output', a.output/'reference_effects'])
        run('outcome_plot', 'tools.plot_closure_outcome_comparison',
            [a.output/'eight_case_comparison', '--after-label', 'Learned reference feedback'],
            python=root.parents[1]/'.venv/Scripts/python.exe')
        if verified['rows']['foot_z_plus15']['complete']:
            run('telemetry', 'tools.record_reference_feedback_telemetry',
                ['--experiment', a.training, '--output', a.output/'telemetry'])
            run('telemetry_audit', 'tools.audit_physics_telemetry', [a.output/'telemetry'])
        run('animation', 'tools.render_closure_training_comparison',
            ['--before', a.baseline, '--after', a.training, '--output', a.output/'video',
             '--before-label', '固定参考基线', '--after-label', '五维参考反馈训练后', '--speed', '4'])
        run('old_grid', 'tools.evaluate_reference_feedback_grid',
            ['--candidate', a.training, '--grid-parent', a.grid_baseline, '--output', a.output/'grid'])
        status.update(state='completed', automatic_promotion=False, training_audit_sha256=verified['audit_sha256'])
        save()
    except Exception:
        status.update(state='failed', error=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
