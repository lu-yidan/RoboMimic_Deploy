"""Launch the adopted local v2 simulation package, with explicit evidence binding."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'outputs/portable_v2_live_T12_eight_20261029_v1'
PROOF=ROOT/'outputs/portable_v2_live_T12_eight_reproduction_20261029_v1/reproduction_audit.json'


def read(path):return json.loads(path.read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_binding(root,package,proof):
    baseline=read(root/'research_baseline.json');release=read(package/'release_manifest.json');evidence=read(proof)
    if baseline['status']!='adopted' or baseline['version']!=2:
        raise ValueError('No adopted physical-v2 baseline')
    if not release['simulation_only'] or release['format']!='g1-physical-v2-multicycle':
        raise ValueError('Expected the audited simulation-only package')
    validation=baseline['multicycle_validation'];case=release['case']
    if release['cycles']!=validation['cycles'] or release['original_audit_sha256']!=validation['case_audits'].get(case):
        raise ValueError('Package case/cycle audit does not belong to the adopted baseline')
    if release['composite']['reference_sha256']!=baseline['reference_sha256']:
        raise ValueError('Package first-cycle reference is no longer adopted')
    if sha(package/release['checkpoint'])!=baseline['policy_sha256']:
        raise ValueError('Package checkpoint differs from adopted policy')
    if sha(package/release['reference'])!=validation['reference_sha256']:
        raise ValueError('Package continuous reference differs from adopted reference')
    if not evidence['passed'] or evidence['release_manifest_sha256']!=sha(package/'release_manifest.json'):
        raise ValueError('Package lacks matching completed reproduction evidence')
    if evidence['golden_audit_sha256']!=release['original_audit_sha256']:
        raise ValueError('Reproduction evidence refers to another accepted run')
    for name,digest in release['files'].items():
        path=(package/name).resolve()
        if not path.is_relative_to(package.resolve()) or sha(path)!=digest:
            raise ValueError('Package asset changed: '+name)
    return dict(case=case,cycles=release['cycles'],package_files=len(release['files']),
        adopted_reference_sha256=baseline['reference_sha256'],policy_sha256=baseline['policy_sha256'],
        completed_reproduction_audit_sha256=sha(proof),simulation_only=True,new_training_steps=0)


def main():
    p=argparse.ArgumentParser(description='Current adopted v2: fresh T12 eight-cycle MuJoCo simulation')
    p.add_argument('--verify-only',action='store_true');p.add_argument('--headless',action='store_true')
    p.add_argument('--output',type=Path);a=p.parse_args()
    proof=verify_binding(ROOT,PACKAGE,PROOF);print(json.dumps(proof),flush=True)
    code="import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); runpy.run_module('deploy_mujoco.run_portable_v2_multicycle',run_name='__main__')"
    command=[sys.executable,'-I','-c',code,str(PACKAGE)]
    if a.verify_only:command+=['--verify-only']
    else:
        out=(a.output if a.output else ROOT/'outputs'/('adopted_v2_live_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
        if out.exists():raise FileExistsError('Choose a new output directory: '+str(out))
        command+=['--output',str(out)]
        if not a.headless:command+=['--viewer']
        print('Fresh dynamics and policy inference; closing early is not a successful completed run.',flush=True)
        print('Output: '+str(out),flush=True)
    subprocess.run(command,cwd=PACKAGE,check=True)


if __name__=='__main__':main()
