"""How much residual authority is actually usable? Frozen policy, four gains, one case, in parallel.

The 131k-step run at 2x authority crashed 63% of its episodes within 30 s, which suggests the usable
authority is limited by stability rather than by the action-space size. This measures it directly:
the same frozen policy, only the authority gain changes.
"""
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROPOSED = Path(r'D:\G1Climbing\.dsh-watch\proposed')
WIDE = Path(r'D:\G1Climbing\.dsh-watch\wide')
REPO = Path(r'D:\G1Climbing\external\RoboMimic_Deploy')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(r'D:\G1Climbing\.dsh-watch')))

import tools.evaluate_closure_action_bias_v2 as ev  # noqa: E402
from calibration_sweep import measure, reference_frame, sha  # noqa: E402

BASE = REPO / 'outputs/calibrated_geometry_training_baseline_20261027_v1'
PLAN = REPO / 'outputs/swing_lift_phase_sine_plan_20261029_v2/plan.npz'
OUT = Path(r'D:\G1Climbing\.dsh-watch\authority_sweep')
CASE = 'training_13'
GAINS = [1.0, 1.25, 1.5, 2.0]


def run_worker(payload):
    directory = Path(payload[-1])
    case = payload[3]
    settings = json.loads((directory.parent.parent / 'settings.json').read_text())
    request = directory.parent / f'{case}_worker_payload.json'
    request.write_text(json.dumps(payload, indent=2))
    log = directory.parent / f'{case}_worker.log'
    environment = dict(os.environ)
    environment['PYTHONPATH'] = '%s;%s' % (WIDE, REPO)
    environment['G1_RESIDUAL_AUTHORITY'] = repr(0.015 * settings['gain'])
    environment['G1_RESIDUAL_MARKER'] = str(WIDE / 'patch_marker_sweep.txt')
    with log.open('w', encoding='utf-8') as stream:
        subprocess.run([sys.executable, str(PROPOSED / 'authority_worker.py'),
                        '--worker-payload', str(request.resolve())],
                       stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=str(REPO),
                       env=environment)
    return json.loads((directory / 'result.json').read_text())


ev.run_case_in_fresh_process = run_worker


def build(arm, gain):
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
    (arm / 'settings.json').write_text(json.dumps(dict(gain=gain, case=CASE,
                                                       authority=0.015 * gain), indent=2))
    return exp


def one(gain):
    arm = OUT / ('gain%.2f' % gain)
    exp = build(arm, gain)
    result = ev.evaluate_parallel(exp, arm / 'eval', [CASE], 1)
    row = result['trajectories'][0]
    return dict(gain=gain, success=row['success'], reason=row['reason'], time_s=row['time_s'],
                peaks=row.get('peaks'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    marker = WIDE / 'patch_marker_sweep.txt'
    if marker.exists():
        marker.unlink()
    with ThreadPoolExecutor(max_workers=len(GAINS)) as pool:
        rows = list(pool.map(one, GAINS))
    qadr, initial, end = reference_frame()
    print('\n%-6s %-8s %-28s %10s %10s' % ('gain', 'success', 'reason', 'closure', 'rise'))
    for row in rows:
        arm = OUT / ('gain%.2f' % row['gain'])
        try:
            measured = measure(arm, [CASE], qadr, initial, end)[0]
            row['closure_mrad'] = measured.get('closure_mrad')
            row['rise_mm'] = measured.get('rise_mm')
        except Exception as exc:
            row['measure_error'] = str(exc)[:160]
        print('%-6.2f %-8s %-28s %10s %10s' % (
            row['gain'], row['success'], str(row['reason'])[:28],
            ('%.2f' % row['closure_mrad']) if row.get('closure_mrad') else '—',
            ('%.2f' % row['rise_mm']) if row.get('rise_mm') else '—'))
    (OUT / 'result.json').write_text(json.dumps(rows, indent=2))
    print('\nwrote', OUT / 'result.json')


if __name__ == '__main__':
    main()
