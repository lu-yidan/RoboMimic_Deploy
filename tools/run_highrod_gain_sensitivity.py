"""Two declared endpoint correction gains; fixed scene and acceptance."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def trial(item):
    from tools.run_early_contact_reset_suite import run_case
    out,gain,plan=item;name=f'gain_{gain:g}'
    result=run_case((name,{'rung_01':[0.,0.,.01]},False,out,plan,False,0.,False,False,False,True,False,True))
    checks={}
    for tool,filename in [('audit_joint_limits.py','joint_limit_audit.json'),('audit_cycle_contact_identity.py','contact_identity_audit.json')]:
        run=subprocess.run([sys.executable,str(ROOT/'tools'/tool),'--run',str(out/name)],capture_output=True,text=True)
        (out/name/(tool+'.log')).write_text(run.stdout+run.stderr)
        checks[filename]=run.returncode==0 and read(out/name/filename)['passed']
    result.update(gain=gain,extra_checks=checks,passed=bool(result['passed'] and all(checks.values())))
    return result

def main():
    from tools.plan_clearance_cycle import audit_plan
    from training.wall_support_env import WallSupportEnv
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    parent=ROOT/'outputs/early_swing14_lift26_plan_v1/plan.npz';adapted=ROOT/'outputs/highrod_endpoint_adapt_plan_v1/plan.npz'
    old=dict(np.load(parent));new=dict(np.load(adapted));jobs=[];e=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2')
    protocol=dict(gains=[.75,1.25],base_plan_sha256=sha(parent),adapted_plan_sha256=sha(adapted),offsets={'rung_01':[0.,0.,.01]},scope='Correction-amplitude sensitivity only, not an equivalent measured-height error. True scene+10mm fixed. Same inherited thresholds and highclearance plan. Gain1.0 already audited separately; no tuning during this run.')
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for gain in protocol['gains']:
        folder=a.out/f'plan_gain_{gain:g}';folder.mkdir();plan=dict(old);plan['qpos']=old['qpos']+gain*(new['qpos']-old['qpos'])
        np.savez_compressed(folder/'plan.npz',**plan);audit=audit_plan(e,plan)
        audit['parent_ik_residual']=audit.pop('ik_residual');audit['scope']='Nominal densecollision and nativepositions only; inheritedIKerror is not freshIKevidence.'
        (folder/'audit.json').write_text(json.dumps(audit,indent=2))
        (folder/'manifest.json').write_text(json.dumps(dict(gain=gain,plan_sha256=sha(folder/'plan.npz'),source_sha256=sha(Path(__file__))),indent=2))
        if not audit['passed']:raise ValueError('Plan audit failed; keep failed plan before dynamics')
        jobs.append((a.out,gain,folder/'plan.npz'))
    e.close()
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(trial,jobs):
            results.append(result);(a.out/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(dict(gain=result['gain'],passed=result['passed'],reason=result['reason'],time_s=result['time_s'])),flush=True)

if __name__=='__main__':main()
