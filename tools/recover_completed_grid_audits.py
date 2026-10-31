"""One audit-only recovery on copies of complete records after native crashes.

Original result rows, summaries, trajectories and logs remain unchanged. This
supplement does not rerun physics or turn a physical failure into a success.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tools.validate_swing_reference import read, write, sha, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--timed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    suite_hashes = {}
    for suite in (args.paired, args.timed):
        summary = read(suite / 'suite_summary.json')
        protocol = read(suite / 'protocol.json')
        assert summary['finished'] and summary['sources_unchanged']
        assert sha(suite / 'protocol.json') == summary['protocol_sha256']
        assert all(sha(ROOT / name) == digest for name, digest in protocol['sources'].items())
        suite_hashes[str(suite / 'suite_summary.json')] = sha(suite / 'suite_summary.json')
        for row in summary['rows']:
            if row['passed'] or not row.get('complete') or 'run' not in row:
                continue
            run = Path(row['run'])
            log = Path(row['log'])
            # Strict recovery eligibility: no completed final audit, full saved
            # rollout, and explicit native audit subprocess exit in original log.
            if (run / 'compiled_pair_audit.json').exists():
                continue
            log_text = log.read_text(encoding='utf-8')
            if 'non-zero exit status 3221225480' not in log_text or 'tools.audit_compiled_' not in log_text:
                continue
            saved = read(run / 'summary.json')
            if not saved.get('passed') or not saved.get('success') or abs(saved['time_s'] - 227.32) > 1e-7:
                continue
            name = f"{row['case']}_{row['arm']}"
            dest = args.output / 'copied_runs' / name
            protected = {str(p.relative_to(run)): sha(p) for p in run.rglob('*') if p.is_file()}
            assert protected['trajectory.npz'] == row['trajectory_sha256']
            assert protected['physics_telemetry.npz'] == row['telemetry_sha256']
            shutil.copytree(run, dest)
            module = 'tools.audit_compiled_timed_hook' if row['arm'] == 'timed_open_loop' else 'tools.audit_compiled_hook_pair'
            recovery_log = args.output / f'{name}.log'
            with recovery_log.open('w', encoding='utf-8') as stream:
                result = subprocess.run([sys.executable, '-m', module, str(dest.resolve())],
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            unchanged = all(sha(run / p) == digest for p, digest in protected.items())
            assert unchanged
            audit_path = dest / 'compiled_pair_audit.json'
            audit = read(audit_path) if audit_path.exists() else None
            report = dict(case=row['case'], arm=row['arm'], original_row=row,
                original_log_sha256=sha(log), original_files=protected, original_unchanged=True,
                copy=str(dest.resolve()), recovery_returncode=result.returncode,
                audit_sha256=sha(audit_path) if audit else None,
                audit_passed=bool(result.returncode == 0 and audit and audit['passed']),
                recovery_log_sha256=sha(recovery_log), physics_rerun=False)
            if audit:
                assert audit['trajectory_sha256'] == protected['trajectory.npz']
                assert audit['telemetry_sha256'] == protected['physics_telemetry.npz']
            reports.append(report)
            write(args.output / f'{name}.json', report)
            print(json.dumps(dict(case=row['case'], arm=row['arm'], recovered=report['audit_passed'])), flush=True)
    assert all(sha(p) == digest for p, digest in suite_hashes.items())
    write(args.output / 'summary.json', dict(recoveries=reports,
        attempted=len(reports), passed=sum(r['audit_passed'] for r in reports),
        original_suite_hashes=suite_hashes, source_sha256=sha(__file__),
        scope='Supplement only. Original declared denominators and execution-error rows retained. One sequential audit-only attempt per eligible complete record, on copies. No rerun, training, promotion or amended main comparison.'))


if __name__ == '__main__':
    main()
