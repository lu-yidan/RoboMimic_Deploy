"""Run declared development cases through two-cycle and eight-cycle audits.

Existing passed records may be reused only with matching parent, plan and hashes.
Each fresh case starts at time zero. This tool never trains or modifies a policy.
Acquire the project's CPU/repository locks before starting a four-worker suite.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf8')


def verified_reuse(directory,case,parent,plan):
    if directory is None:return None
    target=directory/case
    if not (target/'v2_multicycle_audit.json').exists():return None
    audit=read(target/'v2_multicycle_audit.json');m=read(target/'manifest.json')
    if not audit['passed']:raise ValueError(f'Reuse requested for failed record: {target}')
    assert m['case']==case and Path(m['parent_experiment']).resolve()==parent
    assert m['plan_sha256']==sha(plan)==sha(target/'reference.npz')
    assert all(audit[k]==sha(target/name) for k,name in [('manifest_sha256','manifest.json'),
        ('summary_sha256','summary.json'),('trajectory_sha256','trajectory.npz'),
        ('independent_audit_sha256','independent_audit.json'),('policy_audit_sha256','policy_transfer_audit.json')])
    return target.resolve()


def main():
    p=argparse.ArgumentParser()
    for name in ('candidate','two-plan','eight-plan','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cases',nargs='+',help='Default: all development cases in the adopted parent manifest')
    p.add_argument('--reuse-two-suite',type=Path);p.add_argument('--reuse-eight-suite',type=Path)
    p.add_argument('--workers',type=int,choices=range(1,5),default=4);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();parent=a.candidate.resolve();pm=read(parent/'manifest.json')
    cases=a.cases or list(pm['evaluation_cases'])
    assert cases and len(cases)==len(set(cases)) and all(c in pm['evaluation_cases'] for c in cases)
    assert read(parent/'reference_chain_audit.json')['passed']
    assert read(a.two_plan.parent/'manifest.json')['plan_sha256']==sha(a.two_plan)
    assert read(a.eight_plan.parent/'manifest.json')['bridge_sha256']==sha(a.two_plan)
    reused={c:dict(two=verified_reuse(a.reuse_two_suite,c,parent,a.two_plan),eight=verified_reuse(a.reuse_eight_suite,c,parent,a.eight_plan)) for c in cases}
    protocol=dict(cases=cases,case_role='Development/guard set, not held-out validation',candidate=str(parent),
        parent_manifest_sha256=sha(parent/'manifest.json'),checkpoint_sha256=sha(parent/'policy.zip'),
        two_plan=str(a.two_plan.resolve()),eight_plan=str(a.eight_plan.resolve()),
        two_plan_sha256=sha(a.two_plan),eight_plan_sha256=sha(a.eight_plan),workers=a.workers,training=False,
        height_definition='Cumulative rise at cycle i relative to initial run pose, 0.25*i +/-0.011 m; final total +/-0.010 m. Individual increments reported separately.',
        source_sha256=sha(__file__),reuse={c:{k:(str(v) if v else None) for k,v in r.items()} for c,r in reused.items()},
        selection='All declared cases retained, including failures. Two-cycle gate required before each fresh eight-cycle run.')
    if a.dry_run:print(json.dumps(protocol,indent=2));return
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);write(out/'protocol.json',protocol)
    (out/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    def run_command(command,log):
        with log.open('w',encoding='utf8') as stream:
            subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    def stage(case,cycles,gate=None):
        target=out/f'{cycles}_cycles'/case;target.parent.mkdir(exist_ok=True)
        plan=a.two_plan if cycles==2 else a.eight_plan
        command=[sys.executable,'-m','tools.evaluate_v2_multicycle','--candidate',str(parent),
            '--plan',str(plan.resolve()),'--case',case,'--output',str(target)]
        if gate is not None:command+=['--two-cycle-gate',str(gate)]
        run_command(command,out/f'{case}_{cycles}_run.log')
        run_command([sys.executable,'-m','tools.audit_v2_multicycle',str(target)],out/f'{case}_{cycles}_audit.log')
        return target
    def run(case):
        start=time.monotonic();result=dict(case=case,passed=False)
        try:
            two=reused[case]['two'] or stage(case,2)
            result.update(two_run=str(two),two_audit_sha256=sha(two/'v2_multicycle_audit.json'))
            if not read(two/'v2_multicycle_audit.json')['passed']:
                result['reason']='two_cycle_gate_failed'
            else:
                eight=reused[case]['eight'] or stage(case,8,two)
                result.update(eight_run=str(eight),eight_audit_sha256=sha(eight/'v2_multicycle_audit.json'),
                    passed=read(eight/'v2_multicycle_audit.json')['passed'],reason='completed')
        except Exception as exc:
            result.update(reason='process_or_integrity_error',error=f'{type(exc).__name__}: {exc}')
        result['wall_seconds']=time.monotonic()-start
        write(out/f'{case}_result.json',result);print(json.dumps(result),flush=True);return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        results=list(pool.map(run,cases))
    write(out/'suite_summary.json',dict(passed=all(r['passed'] for r in results),passed_cases=sum(r['passed'] for r in results),
        declared_cases=len(cases),results=results,protocol_sha256=sha(out/'protocol.json'),new_training_steps=0))


if __name__=='__main__':main()
