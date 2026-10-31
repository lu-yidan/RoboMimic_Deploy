"""Finite sequential training and regression pipeline for a passed reward gate."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
from tools.compare_audited_closure_candidates import sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    gate=json.loads((a.protocol/'summary.json').read_text())
    assert gate['passed'] and gate['protocol_sha256']==sha(a.protocol/'protocol.json')
    a.output.mkdir(exist_ok=False)
    state=dict(completed=False,protocol=str(a.protocol.resolve()),steps=[])
    def save():
        state['updated_at']=datetime.datetime.now().isoformat()
        (a.output/'status.json').write_text(json.dumps(state,indent=2))
    def run(name,module,*args):
        command=[sys.executable,'-u','-m',module,*map(str,args)]
        state['active_step']=name;save();print('START '+name,flush=True)
        with (a.output/f'{name}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        state['steps'].append(dict(name=name,command=command,exit_code=result.returncode))
        save()
        if result.returncode:raise RuntimeError(f'{name} failed: {result.returncode}; inspect preserved log')
        print('DONE '+name,flush=True)
    for mode in ('average','bottleneck'):
        run('train_'+mode,'tools.train_boundary_reward_arm','--protocol',a.protocol,'--mode',mode,'--output',a.output/mode)
    run('matched_comparison','tools.summarize_boundary_reward_experiment','--protocol',a.protocol,
        '--average',a.output/'average','--bottleneck',a.output/'bottleneck','--output',a.output/'matched_comparison')
    for mode in ('average','bottleneck'):
        for suite,parent in [('original8','outputs/refined_live_landing_ppo18432_20261026_v1'),
                             ('grid12','outputs/refined_live_landing_grid12_20261026_v1')]:
            run(mode+'_'+suite,'tools.evaluate_boundary_reward_regression','--candidate',a.output/mode,
                '--grid-parent',parent,'--output',a.output/(mode+'_'+suite),'--workers','4')
        run(mode+'_endpoints','tools.diagnose_candidate_endpoint_margins','--experiment',a.output/mode,
            '--output',a.output/(mode+'_endpoints'))
        run(mode+'_telemetry','tools.evaluate_closure_foreaft_telemetry','--experiment',a.output/(mode+'_original8'),
            '--output',a.output/(mode+'_telemetry'),'--cases','foot_z_plus15','--workers','1','--verify-recorded')
        run(mode+'_telemetry_audit','tools.audit_physics_telemetry',a.output/(mode+'_telemetry'))
    run('physics_comparison','tools.compare_boundary_reward_telemetry',
        '--original',a.output/'average_telemetry','--modified',a.output/'bottleneck_telemetry',
        '--output',a.output/'physics_comparison')
    state.update(completed=True,active_step=None);save()
    subprocess.run([sys.executable,'-u','-m','tools.report_boundary_reward_validation',str(a.output)],check=True)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
