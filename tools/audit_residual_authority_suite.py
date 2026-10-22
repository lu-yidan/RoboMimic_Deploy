"""Audit every predeclared action including failed trials, without selecting a winner."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.suite/'manifest.json').read_text());results=json.loads((a.suite/'results.json').read_text())
    assert len(results)==len(manifest['actions'])==25
    assert {r['trial'] for r in results}=={name for name,_ in manifest['actions']}
    rows=[]
    for result in results:
        run=a.suite/result['trial']
        process=subprocess.run([sys.executable,str(ROOT/'tools/audit_constant_residual.py'),'--run',str(run)],capture_output=True,text=True)
        (run/'audit_console.log').write_text(process.stdout+process.stderr)
        audit=json.loads((run/'independent_audit.json').read_text()) if (run/'independent_audit.json').exists() else {'integrity_passed':False}
        rows.append(dict(trial=result['trial'],runtime_success=result['success'],audit_passed=bool(process.returncode==0 and audit['integrity_passed']),reason=result['reason'],time_s=result['time_s'],constant_action=result['constant_action']))
    output=dict(count=len(rows),successful_and_audited=sum(r['runtime_success'] and r['audit_passed'] for r in rows),audits_passed=sum(r['audit_passed'] for r in rows),results=rows,scope='Finite diagnostic on one known hard scenario. No held-out validation, learned-policy or full-cycle claim; all25 trials retained.')
    (a.suite/'audited_summary.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))


if __name__=='__main__':main()
