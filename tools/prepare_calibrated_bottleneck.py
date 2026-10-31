"""Prepare a matched bottleneck-reward follow-up to the completed average arm."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from tools.compare_audited_closure_candidates import load_run, sha
from training.boundary_landing_env import boundary_bonus


def check_reward():
    for mode in ('average', 'bottleneck'):
        for ramp in (0., .2, 1.):
            for closure in (0., .032, .039, .04, .05):
                for rise in (.23, .24, .25, .259, .26, .27):
                    b = boundary_bonus(closure,rise,ramp,mode=mode)
                    assert 0. <= b <= .1*ramp + 1e-15
                    worse_joint=boundary_bonus(closure+.001,rise,ramp,mode=mode)
                    worse_height=boundary_bonus(closure,.25+abs(rise-.25)+.001,ramp,mode=mode)
                    assert worse_joint <= b+1e-15 and worse_height <= b+1e-15
                    assert abs(b-boundary_bonus(closure,.5-rise,ramp,mode=mode))<1e-14
                    if mode=='bottleneck':
                        assert b<=boundary_bonus(closure,rise,ramp,mode='average')+1e-15
        for bad in (float('nan'),float('inf'),-.1):
            try: boundary_bonus(bad,.25,1.,mode=mode)
            except ValueError: pass
            else: raise AssertionError('Invalid closure accepted')
    # Improving the non-limiting component cannot mask the worse constraint.
    assert boundary_bonus(.03,.261,1.,mode='bottleneck')==boundary_bonus(.02,.261,1.,mode='bottleneck')


def run_job(parent,output,mode,cases):
    with output.with_suffix('.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-u','-m','tools.check_calibrated_landing',
            '--parent',str(parent),'--output',str(output),'--mode',mode,'--cases',*cases],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f'Gate failed: {output}, exit {result.returncode}')
    report=json.loads((output/'summary.json').read_text())
    assert report['passed'] and report['reward_reconstruction_passed']
    return dict(directory=str(output.resolve()),mode=mode,cases=cases,
                summary_sha256=sha(output/'summary.json'),passed=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validation-report',type=Path,default=Path('outputs/boundary_repair_report_20261027_v1/summary.json'))
    a=p.parse_args();verified=load_run(a.parent);check_reward()
    validation=json.loads(a.validation_report.read_text())
    assert validation['known_suite_complete'] and validation['after_successes']==36
    assert validation['suites']['old16']['directory']==str(a.parent.resolve())
    for f,h in validation['evidence_sha256'].items():assert sha(f)==h
    assert all(r['success'] for r in verified['rows'].values())
    cases=['interior_00','interior_04','interior_07','boundary_00','boundary_04','boundary_05','boundary_06','boundary_07']
    assert set(cases)<=set(verified['cases'])
    a.output.mkdir(exist_ok=False)
    sources={str(f):sha(f) for f in (Path('training/boundary_landing_env.py'),Path('training/live_prefix_landing_env.py'),
        Path('training/closure_quality_reward.py'),Path('tools/check_calibrated_landing.py'),Path('training/calibrated_landing_env.py'),Path(__file__).relative_to(Path.cwd()))}
    for f in ('tools/train_calibrated_bottleneck.py','tools/evaluate_closure_action_bias_v2.py','tools/audit_action_bias_candidate_v2.py','training/closure_action_bias_policy_v2.py'):sources[f]=sha(f)
    protocol=dict(parent=str(a.parent.resolve()),parent_audit_sha256=verified['audit_sha256'],
        parent_checkpoint_sha256=verified['checkpoint_sha256'],training_cases={k:verified['cases'][k] for k in cases},
        evaluation_cases=verified['cases'],partitions=[cases[r::4] for r in range(4)],
        modes=['bottleneck'],additional_steps_per_arm=32768,seed=20261027,environments=4,
        quality_bonus_weight=.1,sources=sources,scalar_reward_checks_passed=True,
        validation_report=str(a.validation_report.resolve()),validation_report_sha256=sha(a.validation_report),
        scope='Finite bottleneck-reward follow-up from the same audited fixed-calibration parent as the completed average arm: same seed, 32768 steps, optimizer, partitions, controller and acceptance. Only reward aggregation changes. Selected after the average-arm result, not prospectively randomized; single-seed development comparison, not a statistical superiority claim. Neighborhood cases remain outside training; no automatic promotion.')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    for mode in protocol['modes']:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(run_job,a.parent,a.output/f'{mode}_worker_{r}',mode,partition) for r,partition in enumerate(protocol['partitions'])]
            for f in as_completed(futures):
                row=f.result();rows.append(row);print(json.dumps(row),flush=True)
    for source,digest in sources.items(): assert sha(source)==digest
    summary=dict(passed=len(rows)==4 and all(r['passed'] for r in rows),rows=rows,protocol_sha256=sha(a.output/'protocol.json'))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(dict(passed=summary['passed'],workers=len(rows))),flush=True)


if __name__=='__main__':main()
