"""T5 sweep over the QP torque weight, on the adopted reference, three weights in parallel."""
import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROPOSED = Path(r'D:\G1Climbing\.dsh-watch\proposed')
REPO = Path(r'D:\G1Climbing\external\RoboMimic_Deploy')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(r'D:\G1Climbing\.dsh-watch')))

import tools.evaluate_closure_action_bias_v2 as ev  # noqa: E402
from calibration_sweep import measure, reference_frame, sha  # noqa: E402

BASE = REPO / 'outputs/calibrated_geometry_training_baseline_20261027_v1'
PLAN = REPO / 'outputs/swing_lift_phase_sine_plan_20261029_v2/plan.npz'
OUT = Path(r'D:\G1Climbing\.dsh-watch\t5_torque_weight')
WEIGHTS = [10000.0, 3000.0, 1000.0]
CASE = 'training_13'


def run_worker(payload):
    directory = Path(payload[-1])
    case = payload[3]
    settings = json.loads((directory.parent.parent / 'settings.json').read_text())
    request = directory.parent / f'{case}_worker_payload.json'
    request.write_text(json.dumps(payload, indent=2))
    log = directory.parent / f'{case}_worker.log'
    with log.open('w', encoding='utf-8') as stream:
        subprocess.run([sys.executable, str(PROPOSED / 'torque_weight_worker.py'),
                        '--worker-payload', str(request.resolve()),
                        '--torque-weight', str(settings['weight'])],
                       stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=str(REPO))
    return json.loads((directory / 'result.json').read_text())


ev.run_case_in_fresh_process = run_worker


def build(arm, weight):
    exp = arm / 'experiment'
    if arm.exists():
        shutil.rmtree(arm)
    exp.mkdir(parents=True)
    manifest = json.loads((BASE / 'manifest.json').read_text())
    manifest['reference'] = str(PLAN)
    manifest['reference_sha256'] = sha(PLAN)
    (exp / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    shutil.copy2(BASE / 'policy.zip', exp / 'policy.zip')
    shutil.copytree(BASE / 'source_snapshot', exp / 'source_snapshot')
    (exp / 'summary.json').write_text(json.dumps(
        {'checkpoint_sha256': sha(exp / 'policy.zip'), 'evaluation_pending': True}, indent=2))
    (arm / 'settings.json').write_text(json.dumps(dict(weight=weight, case=CASE), indent=2))
    return exp


def one(weight):
    arm = OUT / f'w{int(weight)}'
    exp = build(arm, weight)
    result = ev.evaluate_parallel(exp, arm / 'eval', [CASE], 1)
    row = result['trajectories'][0]
    return dict(weight=weight, success=row['success'], reason=row['reason'], time_s=row['time_s'],
                peaks=row.get('peaks'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(WEIGHTS)) as pool:
        rows = list(pool.map(one, WEIGHTS))
    qadr, initial, end = reference_frame()
    for row in rows:
        arm = OUT / f"w{int(row['weight'])}"
        try:
            measured = measure(arm, [CASE], qadr, initial, end)[0]
            row['closure_mrad'] = measured.get('closure_mrad')
            row['rise_mm'] = measured.get('rise_mm')
        except Exception as exc:
            row['measure_error'] = str(exc)[:120]
        print('weight %6.0f  success=%-5s reason=%-26s closure=%s rise=%s' % (
            row['weight'], row['success'], str(row['reason'])[:26],
            round(row['closure_mrad'], 3) if row.get('closure_mrad') else '—',
            round(row['rise_mm'], 3) if row.get('rise_mm') else '—'), flush=True)
    (OUT / 'result.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
