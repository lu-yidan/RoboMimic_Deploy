"""Verify continuation provenance and reused evaluations, then standard audits."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    manifest=read(a.run/'manifest.json');summary=read(a.run/'summary.json');parent=Path(manifest['parent_run'])
    old=read(parent/'summary.json');prior_manifest=read(parent/'manifest.json')
    expected=manifest['parent_checkpoint_sha256']
    checks=dict(parent_checkpoint_unchanged=sha(parent/'policy.zip')==expected==old['checkpoint_sha256'],before_checkpoint_is_parent=sha(a.run/'policy_before.zip')==expected,parent_independently_audited=bool(read(parent/'independent_audit.json')['integrity_passed']),steps_accounted=summary['parent_steps']==old['actual_steps'] and summary['actual_steps']==old['actual_steps']+manifest['additional_steps'] and summary['additional_steps']==manifest['additional_steps'],same_training_distribution=manifest['training_distribution']==prior_manifest['training_distribution'],same_evaluation_angles=manifest['evaluation_angles_deg']==prior_manifest['evaluation_angles_deg'],same_prefix_hashes=manifest['prefixes']==prior_manifest['prefixes'])
    failures=[]
    for name,origin in manifest['reused_evaluations'].items():
        if sha(a.run/name)!=origin['sha256'] or sha(Path(origin['source']))!=origin['sha256']:failures.append(name)
    checks['thirty_reused_files_match']=len(manifest['reused_evaluations'])==30 and not failures
    checks['zero_summary_reused_exactly']=summary['zero']==old['zero']
    checks['before_summary_is_parent_after']=summary['before']==old['after']
    process=subprocess.run([sys.executable,str(ROOT/'tools/audit_biased_load_transfer.py'),'--run',str(a.run)],capture_output=True,text=True)
    (a.run/'standard_audit_console.log').write_text(process.stdout+process.stderr)
    checks['standard_45_trajectory_and_policy_audit']=process.returncode==0 and read(a.run/'independent_audit.json')['integrity_passed']
    result=dict(passed=all(checks.values()),checks=checks,reused_file_failures=failures,scope='Policy/optimizer continuation with simulator/RNG reset, not an uninterrupted simulator run. Reused zero/before data explicitly verified; newafter trajectories independently audited.',source_sha256=sha(Path(__file__)))
    (a.run/'continuation_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
