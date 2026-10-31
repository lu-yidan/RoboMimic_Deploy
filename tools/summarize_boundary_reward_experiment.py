"""Compare two audited reward arms; never select a model from reward alone."""
import argparse
import csv
import json
from pathlib import Path
import torch
from stable_baselines3 import PPO
from tools.compare_audited_closure_candidates import load_run,sha
from tools.continue_vector_full_cycle_training import state_difference


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser()
    for key in ('protocol','average','bottleneck','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    protocol=read(a.protocol/'protocol.json')
    parent=load_run(Path(protocol['parent']))
    runs={mode:load_run(getattr(a,mode)) for mode in ('average','bottleneck')}
    manifests={mode:read(getattr(a,mode)/'manifest.json') for mode in runs}
    summaries={mode:read(getattr(a,mode)/'summary.json') for mode in runs}
    common=('training_cases','evaluation_cases','config','seed','environments',
        'controller_modifier','adaptive_reference_command','reference_sha256',
        'prefix_checkpoint_sha256','suffix_start_seconds','initial_checkpoint_sha256',
        'quality_bonus_weight','matched_protocol_sha256','boundary_gate_sha256')
    checks={key:manifests['average'][key]==manifests['bottleneck'][key] for key in common}
    for mode in runs:
        m=manifests[mode];s=summaries[mode];root=getattr(a,mode)
        assert m['boundary_reward_mode']==mode
        assert m['matched_protocol_sha256']==sha(a.protocol/'protocol.json')
        assert s['additional_steps']==protocol['additional_steps_per_arm']
        assert runs[mode]['cases']==parent['cases']
        for f in ('controller_modifier_audit_v2.json','adaptive_reference_audit.json','composite_prefix_audit_v2.json'):
            assert read(root/f)['passed']
    models={mode:PPO.load(getattr(a,mode)/'policy_before.zip',device='cpu') for mode in runs}
    parent_model=PPO.load(Path(protocol['parent'])/'policy.zip',device='cpu')
    checks['equal_initial_parameters']=state_difference(models['average'].policy.state_dict(),models['bottleneck'].policy.state_dict())==0
    checks['equal_initial_optimizer']=state_difference(models['average'].policy.optimizer.state_dict(),models['bottleneck'].policy.optimizer.state_dict())==0
    checks['initial_parameters_match_parent']=state_difference(models['average'].policy.state_dict(),parent_model.policy.state_dict())==0
    checks['initial_optimizer_matches_parent']=state_difference(models['average'].policy.optimizer.state_dict(),parent_model.policy.optimizer.state_dict())==0
    assert all(checks.values())
    rows=[]
    for case in parent['cases']:
        row=dict(case=case,used_for_training=case in protocol['training_cases'])
        for label,run in dict(parent=parent,**runs).items():
            for key,value in run['rows'][case].items():row[label+'_'+key]=value
        rows.append(row)
    outcomes={}
    for mode,run in runs.items():
        s=summaries[mode]
        final_model=PPO.load(getattr(a,mode)/'policy.zip',device='cpu')
        initial_state=models[mode].policy.state_dict();final_state=final_model.policy.state_dict()
        parameter_changes={}
        for group,is_critic in [('actor_and_shared',False),('critic',True)]:
            keys=[k for k in initial_state if k.startswith(('mlp_extractor.value_net.','value_net.'))==is_critic]
            parameter_changes[group]=state_difference({k:initial_state[k] for k in keys},{k:final_state[k] for k in keys})
        with (getattr(a,mode)/'progress.csv').open(newline='') as stream:
            progress=list(csv.DictReader(stream))
        outcomes[mode]=dict(successes=sum(r['success'] for r in run['rows'].values()),cases=len(run['rows']),
            gained_vs_parent=[c for c,r in run['rows'].items() if r['success'] and not parent['rows'][c]['success']],
            lost_vs_parent=[c for c,r in run['rows'].items() if not r['success'] and parent['rows'][c]['success']],
            completed_training_episodes=s['completed_training_episodes'],successful_training_episodes=s['successful_training_episodes'],
            additional_steps=s['additional_steps'],wall_seconds=s['training_wall_seconds'],case_samples=s['case_samples'],
            last_training_metrics=progress[-1],checkpoint_sha256=run['checkpoint_sha256'],audit_sha256=run['audit_sha256'])
        outcomes[mode]['maximum_parameter_changes']=parameter_changes
    result=dict(passed=True,matched_checks=checks,outcomes=outcomes,rows=rows,
        bottleneck_gained_vs_average=[c for c in runs['average']['rows'] if runs['bottleneck']['rows'][c]['success'] and not runs['average']['rows'][c]['success']],
        bottleneck_lost_vs_average=[c for c in runs['average']['rows'] if not runs['bottleneck']['rows'][c]['success'] and runs['average']['rows'][c]['success']],
        protocol_sha256=sha(a.protocol/'protocol.json'),source_sha256=sha(__file__),automatic_promotion=False,
        scope='Single-seed matched-budget development comparison, not proof of statistical superiority. Training reward values differ by definition and cannot be ranked as task performance. Other-suite regression and physics telemetry are separate checks.')
    a.output.mkdir(exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2))
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    with (a.output/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(passed=True,outcomes={k:v['successes'] for k,v in outcomes.items()},
        gained=result['bottleneck_gained_vs_average'],lost=result['bottleneck_lost_vs_average'])))


if __name__=='__main__':main()
