"""Extend historical acceptance without altering any original results."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', type=Path, required=True)
    args = parser.parse_args()
    results = json.loads((args.suite / 'results.json').read_text())
    rows = []
    for case in results:
        run = ROOT / case['source'] if 'source' in case else args.suite / case['name']
        # Aggregated suites can point to runs located in constituent suites.
        if not (run / 'trajectory.npz').exists():
            raise FileNotFoundError(f'Run must be explicit: {run}')
        process = subprocess.run([sys.executable, str(ROOT / 'tools/audit_joint_limits.py'), '--run', str(run)], capture_output=True, text=True)
        (run / 'native_position_audit_console.log').write_text(process.stdout + process.stderr)
        if process.returncode not in (0, 1) or not (run / 'joint_limit_audit.json').exists():
            raise RuntimeError(f'Audit execution failed: {run}')
        audit = json.loads((run / 'joint_limit_audit.json').read_text())
        rows.append(dict(name=case['name'], legacy_passed=bool(case['passed']), native_position_passed=audit['passed'], extended_passed=bool(case['passed'] and audit['passed']), minimum_margin_rad=audit['minimum_margin_rad'], worst_joint=audit['worst_joint'], violating_samples=audit['violating_samples']))
        print(json.dumps(rows[-1]), flush=True)
    result = dict(legacy_passes=sum(x['legacy_passed'] for x in rows), extended_passes=sum(x['extended_passed'] for x in rows), count=len(rows), cases=rows, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scope='Additional saved-state position audit; historical results unchanged. No between-sample position guarantee.')
    (args.suite / 'extended_native_acceptance.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
