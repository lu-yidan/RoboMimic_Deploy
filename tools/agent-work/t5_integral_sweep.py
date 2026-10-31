"""T5: integral action on the joint reference, on the ADOPTED reference (8 mm swing lift).

Earlier attempts (on the old reference) failed immediately with a joint-limit violation / 125 N contact
because the swing corridor was too tight. The adopted reference opens it by 8 mm, so a gentle integral
may now be tolerable. Three gains, one case, one run each.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROPOSED = Path(r'D:\G1Climbing\.dsh-watch\proposed')
REPO = Path(r'D:\G1Climbing\external\RoboMimic_Deploy')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(r'D:\G1Climbing\.dsh-watch')))

import tools.evaluate_closure_action_bias_v2 as ev  # noqa: E402
from calibration_sweep import sha  # noqa: E402

BASE = REPO / 'outputs/calibrated_geometry_training_baseline_20261027_v1'
PLAN = REPO / 'outputs/swing_lift_phase_sine_plan_20261029_v2/plan.npz'
OUT = Path(r'D:\G1Climbing\.dsh-watch\t5_integral')
CASE = 'training_13'
ARMS = [(0.3, 0.03), (0.6, 0.05), (1.0, 0.08)]


def build(arm_dir, gamma, limit):
    exp = arm_dir / 'experiment'
    if arm_dir.exists():
        shutil.rmtree(arm_dir)
    exp.mkdir(parents=True)
    manifest = json.loads((BASE / 'manifest.json').read_text())
    manifest['reference'] = str(PLAN)
    manifest['reference_sha256'] = sha(PLAN)
    (exp / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    shutil.copy2(BASE / 'policy.zip', exp / 'policy.zip')
    shutil.copytree(BASE / 'source_snapshot', exp / 'source_snapshot')
    (exp / 'summary.json').write_text(json.dumps(
        {'checkpoint_sha256': sha(exp / 'policy.zip'), 'evaluation_pending': True}, indent=2))
    (arm_dir / 'settings.json').write_text(json.dumps(dict(gamma=gamma, limit=limit, case=CASE), indent=2))
    return exp


def run_case(payload):
    directory = Path(payload[-1])
    case = payload[3]
    request = directory.parent / f'{case}_worker_payload.json'
    request.write_text(json.dumps(payload, indent=2))
    settings = json.loads((directory.parent.parent / 'settings.json').read_text())
    log = directory.parent / f'{case}_worker.log'
    with log.open('w', encoding='utf-8') as stream:
        subprocess.run([sys.executable, str(PROPOSED / 'integral_worker.py'),
                        '--worker-payload', str(request.resolve()),
                        '--gamma', str(settings['gamma']), '--limit', str(settings['limit'])],
                       stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=str(REPO))
    return json.loads((directory / 'result.json').read_text())


ev.run_case_in_fresh_process = run_case


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for gamma, limit in ARMS:
        arm_dir = OUT / f'g{gamma:.1f}_l{limit:.2f}'
        exp = build(arm_dir, gamma, limit)
        try:
            result = ev.evaluate_parallel(exp, arm_dir / 'eval', [CASE], 1)
            row = result['trajectories'][0]
            print('gamma %.1f limit %.2f -> success=%s reason=%s time=%.2f'
                  % (gamma, limit, row['success'], row['reason'], row['time_s']), flush=True)
        except Exception as exc:
            print('gamma %.1f limit %.2f -> FAILED: %s' % (gamma, limit, str(exc)[:160]), flush=True)


if __name__ == '__main__':
    main()
