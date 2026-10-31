"""Same-authority single -> two -> eight cycle validation, with resumable results.

Runs only missing stages. Never promotes a model, changes acceptance, or retries
failed physics. The optional wait case is already executing in another process.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
PYTHON=ROOT/'.venv/Scripts/python.exe'
OUT=ROOT/'outputs/authority125_validation_20261031_v1'


def run_stage(args, log):
    env={k:v for k,v in os.environ.items() if not k.startswith('G1_')}
    env['PYTHONPATH']=str(ROOT)
    env['PYTHONIOENCODING']='utf-8'
    with log.open('w',encoding='utf-8') as f:
        subprocess.run([str(PYTHON),'-u','-m',*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)


def one(case, wait_case):
    prior=None
    for count,name in ((2,'two'),(8,'eight')):
        dest=ROOT/f'outputs/authority125_{name}_20261031_v1'/case
        if not (dest/'summary.json').exists():
            if count==2 and case==wait_case and dest.exists():
                deadline=time.monotonic()+1800
                while not (dest/'summary.json').exists():
                    if (dest/'interrupted.json').exists() or time.monotonic()>deadline:
                        raise RuntimeError('Existing case interrupted or did not finish')
                    time.sleep(10)
            else:
                args=['tools.evaluate_authority_multicycle','--candidate',str(ROOT/'outputs/swing_lift_phase_sine_policy_20261029_v2'),
                      '--plan',str(ROOT/f'outputs/v2_swing_{name}_reference_20261029_v1/plan.npz'),
                      '--single-cycle-record',str(PROJECT/'.dsh-watch/authority_all24/eval'/case),
                      '--case',case,'--output',str(dest)]
                if prior: args+=['--two-cycle-gate',str(prior)]
                run_stage(args,OUT/f'{case}_{name}.log')
        if not (dest/'v2_multicycle_audit.json').exists():
            run_stage(['tools.audit_authority_multicycle',str(dest)],OUT/f'{case}_{name}_audit.log')
        audit=json.loads((dest/'v2_multicycle_audit.json').read_text())
        if not audit['passed']: return dict(case=case,passed=False,stage=count,audit=str(dest/'v2_multicycle_audit.json'))
        prior=dest
    return dict(case=case,passed=True,stage=8,audit=str(prior/'v2_multicycle_audit.json'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-case',default='training_13');p.add_argument('--all',action='store_true')
    a=p.parse_args();OUT.mkdir(exist_ok=True)
    cases=['training_13','training_00','interior_00']
    if a.all:
        available=json.loads((ROOT/'outputs/swing_lift_phase_sine_policy_20261029_v2/manifest.json').read_text())['evaluation_cases']
        cases+=sorted(set(available)-set(cases))
    rows=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending={pool.submit(one,c,a.wait_case):c for c in cases}
        for future in as_completed(pending):
            try: row=future.result()
            except Exception as e: row=dict(case=pending[future],passed=False,error=repr(e))
            rows.append(row)
            (OUT/'progress.json').write_text(json.dumps(dict(requested_cases=cases,finished=len(rows),rows=rows),indent=2))
            print(json.dumps(row),flush=True)
    (OUT/'summary.json').write_text(json.dumps(dict(passed=all(r['passed'] for r in rows),cases=len(cases),rows=rows),indent=2))


if __name__=='__main__': main()
