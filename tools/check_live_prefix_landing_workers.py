"""Verify all eight cases across four processes and two resets per process."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time
from tools.compare_audited_closure_candidates import load_run, sha


def worker(payload):
    parent,output,cases=payload
    command=[sys.executable,'-u','-m','tools.check_live_prefix_landing','--parent',str(parent),'--output',str(output),'--cases',*cases]
    started=time.perf_counter()
    with output.with_suffix('.log').open('w',encoding='utf-8') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
    summary=json.loads((output/'summary.json').read_text()) if (output/'summary.json').exists() else None
    return dict(directory=str(output),cases=cases,exit_code=result.returncode,wall_seconds=time.perf_counter()-started,
                passed=result.returncode==0 and bool(summary and summary['passed']),summary=summary,
                summary_sha256=sha(output/'summary.json') if summary else None)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verified=load_run(a.parent)
    pilot=json.loads((a.pilot/'summary.json').read_text())
    assert pilot['passed'] and pilot['parent_audit_sha256']==verified['audit_sha256']
    for source,digest in pilot['sources'].items():assert sha(source)==digest
    names=list(verified['cases']);assert len(names)==8
    a.output.mkdir(exist_ok=False)
    sources={name:sha(name) for name in ('training/live_prefix_landing_env.py','tools/check_live_prefix_landing.py','tools/check_live_prefix_landing_workers.py')}
    protocol=dict(parent=str(a.parent.resolve()),parent_audit_sha256=verified['audit_sha256'],pilot_summary_sha256=sha(a.pilot/'summary.json'),
                  workers=4,partitions=[names[rank::4] for rank in range(4)],sources=sources,
                  scope='Each worker uses one environment object for two sequential case resets. Real prefix simulation on each reset; compare full records bitwise. No PPO updates.')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    started=time.perf_counter();rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(worker,(a.parent,a.output/f'worker_{rank}',cases)) for rank,cases in enumerate(protocol['partitions'])]
        for future in as_completed(jobs):
            row=future.result();rows.append(row);print(json.dumps({k:row[k] for k in ('directory','cases','passed','wall_seconds')}),flush=True)
    accepted=[item for row in rows if row['summary'] for item in row['summary']['rows']]
    result=dict(passed=all(row['passed'] for row in rows) and {row['case'] for row in accepted}==set(names),
                workers=rows,rows=accepted,wall_seconds=time.perf_counter()-started,protocol_sha256=sha(a.output/'protocol.json'))
    (a.output/'summary.json').write_text(json.dumps(result,indent=2))
    for name,digest in sources.items():
        assert sha(name)==digest
        dest=a.output/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(name).read_bytes())
    print(json.dumps(dict(passed=result['passed'],cases=len(accepted),wall_seconds=result['wall_seconds'])))
    assert result['passed']


if __name__=='__main__':main()
