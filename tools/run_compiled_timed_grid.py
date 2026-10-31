"""Run the predeclared third arm without selecting on either second-cycle arm."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import shutil
import subprocess
import sys
from tools.validate_swing_reference import ROOT,read,write,sha
from tools.run_compiled_hook_pairs import eligible,outcome

SOURCES=('tools/run_compiled_timed_grid.py','tools/evaluate_compiled_timed_hook.py',
    'tools/audit_compiled_timed_hook.py','tools/run_compiled_hook_pairs.py',
    'training/compiled_rung_geometry.py','training/timed_hook_loading.py',
    'training/probe_hook_loading.py','training/physics_telemetry.py')


def prepare(declaration,pilot,out):
    declaration=declaration.resolve();pilot=pilot.resolve()
    d=read(declaration);single=Path(d['single_cycle_screen']);paired=Path(d['paired_experiment'])
    pair=read(paired/'protocol.json');p=read(pilot/'timed_loading_audit.json')
    assert p['passed'] and p['nominal_probe_reproduced']
    assert p['manifest_sha256']==sha(pilot/'manifest.json')
    assert p['trajectory_sha256']==sha(pilot/'trajectory.npz')
    assert p['telemetry_sha256']==sha(pilot/'physics_telemetry.npz')
    assert read(pilot/'timed_loading_protocol.json')['delays_seconds']==d['delays_seconds']
    assert d['paired_protocol_sha256']==sha(paired/'protocol.json')
    assert d['cases']==pair['cases'] and len(d['cases'])==16
    assert not (single/'eval').exists(), 'Freeze runner before reading new-grid outcomes'
    for relative,digest in d['sources'].items():assert sha(ROOT/relative)==digest
    out.mkdir(parents=True,exist_ok=False);sources={}
    for relative in SOURCES:
        target=out/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target);sources[relative]=sha(target)
    write(out/'protocol.json',dict(declaration=str(declaration),declaration_sha256=sha(declaration),
        paired_experiment=str(paired),paired_protocol_sha256=sha(paired/'protocol.json'),
        single_cycle_screen=str(single),single_manifest_sha256=sha(single/'manifest.json'),
        pilot=str(pilot),pilot_audit_sha256=sha(pilot/'timed_loading_audit.json'),
        cases=d['cases'],arm='timed_open_loop',plan=pair['plan'],plan_sha256=pair['plan_sha256'],
        sources=sources,denominator=16,new_training_steps=0,automatic_promotion=False,
        scope='Third arm declared before grid outcomes. Delay calibrated on nominal T12. All first-cycle failures count; no second-cycle success conditioning. No sensor-error or unseen-wall claim.'))
    print(json.dumps(dict(prepared=True,cases=16,output=str(out))),flush=True)


def run(out,workers):
    protocol=read(out/'protocol.json');single=Path(protocol['single_cycle_screen'])
    declaration=Path(protocol['declaration'])
    assert sha(declaration)==protocol['declaration_sha256']
    assert sha(single/'manifest.json')==protocol['single_manifest_sha256']
    assert sha(protocol['plan'])==protocol['plan_sha256']
    assert sha(Path(protocol['pilot'])/'timed_loading_audit.json')==protocol['pilot_audit_sha256']
    for relative,digest in protocol['sources'].items():
        assert sha(ROOT/relative)==digest==sha(out/'source_snapshot'/relative)
    assert not (out/'runs').exists() and not (out/'suite_summary.json').exists()
    for name in ('runs','logs','results'):(out/name).mkdir()
    rows=[];jobs=[]
    for case in protocol['cases']:
        if eligible(single,case):jobs.append(case)
        else:
            row=dict(case=case,arm='timed_open_loop',passed=False,complete=False,status='first_cycle_gate_failed')
            rows.append(row);write(out/'results'/f'{case}.json',row)
    def one(case):
        target=out/'runs'/case;log=out/'logs'/f'{case}.log'
        with log.open('w',encoding='utf8') as stream:
            result=subprocess.run([sys.executable,'-m','tools.evaluate_compiled_timed_hook',
                '--candidate',str(single),'--plan',protocol['plan'],'--case',case,
                '--arm','timed_open_loop','--protocol',str(declaration),'--output',str(target)],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        try:row=outcome(target,case,'timed_open_loop',result.returncode,log)
        except Exception as exc:
            row=dict(case=case,arm='timed_open_loop',passed=False,complete=False,status='result_integrity_error',
                error=repr(exc),process_returncode=result.returncode,run=str(target),log=str(log))
        write(out/'results'/f'{case}.json',row)
        print(json.dumps(dict(case=case,status=row['status'],passed=row['passed'])),flush=True)
        return row
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one,case) for case in jobs]):rows.append(future.result())
    assert len(rows)==len({r['case'] for r in rows})==16
    rows.sort(key=lambda r:r['case'])
    report=dict(finished=True,declared_cases=16,declared_runs=16,arm='timed_open_loop',rows=rows,
        passed_count=sum(r['passed'] for r in rows),protocol_sha256=sha(out/'protocol.json'),
        single_chain_sha256=sha(single/'compiled_reference_chain_audit.json'),
        result_sha256={p.name:sha(p) for p in sorted((out/'results').glob('*.json'))},
        sources_unchanged=all(sha(ROOT/k)==v for k,v in protocol['sources'].items()),
        new_training_steps=0,automatic_promotion=False,scope=protocol['scope'])
    write(out/'suite_summary.json',report)
    print(json.dumps(dict(finished=True,passed_count=report['passed_count'],declared_cases=16)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','run'))
    p.add_argument('--declaration',type=Path);p.add_argument('--pilot',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,choices=(1,2,4),default=4)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.declaration,a.pilot,a.output.resolve())
    else:run(a.output.resolve(),a.workers)
