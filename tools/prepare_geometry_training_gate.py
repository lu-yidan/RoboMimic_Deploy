"""Gate a frozen geometry curriculum with four representative physical replays."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from tools.compare_audited_closure_candidates import load_run, sha
from tools.prepare_calibrated_training import run_job, check_reward, check_terminal_margin


def main():
    p=argparse.ArgumentParser()
    for name in ('baseline','geometry-protocol','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--terminal-margin-weight',type=float,default=0.)
    p.add_argument('--terminal-margin-scale-mrad',type=float,default=1.5)
    a=p.parse_args();check_reward();check_terminal_margin(a.terminal_margin_weight,a.terminal_margin_scale_mrad)
    run=load_run(a.baseline)
    geometry=json.loads((a.geometry_protocol/'protocol.json').read_text())
    result=json.loads((a.baseline/'baseline_result.json').read_text())
    assert result['completed'] and not result['validation_evaluated']
    assert result['protocol_sha256']==sha(a.geometry_protocol/'protocol.json')
    assert run['cases']==geometry['training_cases'] and len(run['cases'])==24
    baseline_summary=json.loads((a.baseline/'summary.json').read_text())
    assert all(row['time_s']>70.22 for row in baseline_summary['after']), 'Frozen-prefix failures cannot be repaired by a suffix learner'
    assert all(run['rows'][k]['success'] for k in run['rows'] if not k.startswith('training_'))
    keys=list(run['cases']);partitions=[keys[r::4] for r in range(4)]
    gate_cases=[[partition[2+r]] for r,partition in enumerate(partitions)]
    assert len({x for row in gate_cases for x in row})==4
    sources={name:sha(name) for name in ('tools/train_calibrated_landing.py','tools/check_calibrated_landing.py',
        'training/calibrated_landing_env.py','training/boundary_landing_env.py','training/live_prefix_landing_env.py',
        'training/closure_quality_reward.py','training/terminal_margin_reward.py','tools/evaluate_closure_action_bias_v2.py',
        'tools/audit_action_bias_candidate_v2.py','training/closure_action_bias_policy_v2.py',
        'tools/prepare_geometry_training_gate.py','tools/prepare_calibrated_training.py')}
    protocol=dict(parent=str(a.baseline.resolve()),parent_audit_sha256=run['audit_sha256'],parent_checkpoint_sha256=run['checkpoint_sha256'],
        training_cases=run['cases'],evaluation_cases=run['cases'],partitions=partitions,gate_cases=gate_cases,
        modes=['average'],additional_steps_per_arm=geometry['planned_additional_steps'],seed=geometry['planned_training_seed'],
        environments=4,quality_bonus_weight=.1,sources=sources,scalar_reward_checks_passed=True,
        terminal_margin=dict(weight=a.terminal_margin_weight,scale_mrad=a.terminal_margin_scale_mrad,
            contract='Add weight*clip(min(40-closure_mrad,(10-|rise_mm-250|)*10)/scale,-1,1) once at the terminal step; '
                     'not scored when the cycle did not complete; acceptance unchanged.'),
        validation_report=str((a.baseline/'baseline_result.json').resolve()),validation_report_sha256=sha(a.baseline/'baseline_result.json'),
        geometry_protocol_sha256=sha(a.geometry_protocol/'protocol.json'),
        scope=geometry['scope']+' Training environment is verified on four representative new cases distributed across the four worker partitions, not all 24 cases. Baseline has time-zero evaluation through the original success/failure termination and independent action/state audits on all 24. Failures after learning entry remain eligible training cases and are not filtered out; any failure before entry blocks this gate. Sixteen validation cases remain unused; this trainer evaluates only training/guard cases. Report this distinction explicitly.')
    a.output.mkdir(exist_ok=False);(a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(run_job,a.baseline,a.output/f'average_worker_{r}','average',cases,
                          a.terminal_margin_weight,a.terminal_margin_scale_mrad) for r,cases in enumerate(gate_cases)]
        for future in as_completed(jobs):rows.append(future.result())
    assert all(sha(name)==digest for name,digest in sources.items())
    summary=dict(passed=len(rows)==4 and all(r['passed'] for r in rows),rows=rows,
        protocol_sha256=sha(a.output/'protocol.json'),representative_gate_cases=gate_cases,validation_evaluated=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));(a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
