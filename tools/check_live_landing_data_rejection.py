"""Verify the data auditor rejects rehashed action and reward corruption."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from tools.compare_audited_closure_candidates import sha


def main():
    p=argparse.ArgumentParser()
    for key in ('dataset','seed','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    record_path=sorted(a.dataset.glob('episode_*.json'))[0]
    row=json.loads(record_path.read_text());original=a.dataset/row['file'];original_hash=sha(original)
    assert original_hash==row['trajectory_sha256']
    with np.load(original) as z:arrays={k:v.copy() for k,v in z.items()}
    rows=[]
    for label,key in (('changed_action','actions'),('changed_bonus','rewards')):
        directory=a.output/label;directory.mkdir();shutil.copy2(a.dataset/'protocol.json',directory/'protocol.json')
        altered={k:v.copy() for k,v in arrays.items()}
        if key=='actions':altered[key][10,0]+=np.float32(.01)
        else:altered[key][10]+=.001
        path=directory/row['file'];np.savez_compressed(path,**altered)
        metadata=dict(row,trajectory_sha256=sha(path));path.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
        destination=a.output/(label+'_audit')
        command=[sys.executable,'-m','tools.audit_live_landing_data','--dataset',str(directory),'--seed',str(a.seed),'--output',str(destination),'--partial']
        with (a.output/(label+'.log')).open('w') as stream:completed=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=False)
        report=json.loads((destination/'summary.json').read_text());checks=report['rows'][0]['checks']
        rejected=completed.returncode!=0 and not report['passed']
        intended=not checks['stochastic_action_replay_exact'] if key=='actions' else not checks['quality_bonus']
        rows.append(dict(fixture=label,passed=rejected and intended,failed_checks=[k for k,v in checks.items() if not v],exit_code=completed.returncode))
    result=dict(passed=all(r['passed'] for r in rows),rows=rows,original_unchanged=sha(original)==original_hash,
                auditor_sha256=sha('tools/audit_live_landing_data.py'),source_sha256=sha(__file__),scope='Rehashed corruption fixtures only. Original sampled data remains unchanged; these fixtures must not be used for training.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2));(a.output/'source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result));assert result['passed'] and result['original_unchanged']


if __name__=='__main__':main()
