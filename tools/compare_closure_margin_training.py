"""Reject incomplete/unmatched runs before comparing strict task outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from stable_baselines3 import PPO
from tools.continue_vector_full_cycle_training import sha,state_difference

MATCHED=('contract','observation_contract','geometry_mode','config','training_cases','evaluation_cases',
    'seed','parent_steps','additional_steps_requested','environments','parent_checkpoint_sha256','protocol_sha256',
    'reference_sha256','bundle_manifest_sha256','normal_bias_y_deg','force_history_memory','sources',
    'optimizer_initialization','closure_cost_parameters','reward_gate_sha256','training_reward_contract')


def validate_pair(manifests,summaries):
    if [m['closure_cost_weight'] for m in manifests]!=[0.,.02]:raise ValueError('Expected weight0 then0.02')
    for key in MATCHED:
        if manifests[0][key]!=manifests[1][key]:raise ValueError('Unmatched '+key)
    for m,s in zip(manifests,summaries):
        if s.get('evaluation_pending') or s.get('stopped_by_request'):raise ValueError('Incomplete experiment')
        if s['parent_steps']!=m['parent_steps'] or s['additional_steps']!=m['additional_steps_requested'] or s['actual_steps']!=s['parent_steps']+s['additional_steps']:raise ValueError('Incomplete budget')
        if s['cost_steps']!=s['additional_steps'] or not 0<=s['maximum_step_closure_cost']<=m['closure_cost_weight']:raise ValueError('Cost accounting differs')
        if not 0<=s['total_training_closure_cost']<=m['closure_cost_weight']*s['additional_steps']+1e-8:raise ValueError('Cost sum out of bounds')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--control',type=Path,required=True);parser.add_argument('--penalty',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();dirs=[args.control,args.penalty]
    manifests=[json.loads((p/'manifest.json').read_text()) for p in dirs];summaries=[json.loads((p/'summary.json').read_text()) for p in dirs]
    validate_pair(manifests,summaries);torch.set_num_threads(1)
    models=[PPO.load(p/'policy_before.zip',device='cpu') for p in dirs]
    if state_difference(models[0].policy.state_dict(),models[1].policy.state_dict()) or state_difference(models[0].policy.optimizer.state_dict(),models[1].policy.optimizer.state_dict()):raise ValueError('Initial actor/critic/optimizer differ')
    audits=[];coverage=[]
    for p,m,s in zip(dirs,manifests,summaries):
        a=json.loads((p/'independent_training_audit.json').read_text())
        if not a['passed']:raise ValueError('Audit failed')
        for filename,key in [('manifest.json','experiment_manifest_sha256'),('summary.json','experiment_summary_sha256'),('policy.zip','checkpoint_sha256')]:
            if sha(p/filename)!=a[key]:raise ValueError('Audited evidence changed')
        if sha(p/'matched_protocol.json')!=m['protocol_sha256']:raise ValueError('Protocol changed')
        for label in ('zero','after'):
            subset=[r for r in a['trajectories'] if r['label']==label]
            if len(subset)!=len(m['evaluation_cases']) or {r['case'] for r in subset}!=set(m['evaluation_cases']):raise ValueError('Incomplete audit')
            for r in subset:
                if sha(p/f"{label}_{r['case']}.npz")!=r['trajectory_sha256']:raise ValueError('Trajectory changed')
        episodes=[json.loads(line) for line in (p/'training_episodes.jsonl').read_text().splitlines()]
        if len(episodes)!=s['completed_training_episodes']:raise ValueError('Episode count mismatch')
        for e in episodes:
            if abs(e['episode']['r']-(e['base_episode_return']-e['closure_episode_cost']))>1e-5:raise ValueError('Episode reward accounting mismatch')
        coverage.append(dict(weight=m['closure_cost_weight'],completed=dict(Counter(e['case'] for e in episodes)),successful=dict(Counter(e['case'] for e in episodes if e['success'])),total_margin_cost=s['total_training_closure_cost']))
        audits.append(a)
    rows=[]
    for case in manifests[0]['evaluation_cases']:
        members=[next(r for r in a['trajectories'] if r['label']=='after' and r['case']==case) for a in audits]
        outcomes=[r['independently_accepted'] for r in members]
        rows.append(dict(case=case,control_passed=outcomes[0],penalty_passed=outcomes[1],fixed=not outcomes[0] and outcomes[1],regressed=outcomes[0] and not outcomes[1],
            control_endpoints=members[0]['independent_endpoints'],penalty_endpoints=members[1]['independent_endpoints']))
    result=dict(cases=rows,control_passed=sum(r['control_passed'] for r in rows),penalty_passed=sum(r['penalty_passed'] for r in rows),
        fixed_cases=[r['case'] for r in rows if r['fixed']],regressed_cases=[r['case'] for r in rows if r['regressed']],coverage=coverage,
        audit_hashes=[sha(p/'independent_training_audit.json') for p in dirs],protocol_sha256=manifests[0]['protocol_sha256'],source_sha256=sha(__file__),
        scope='One matched seed and budget, five known training cases. Strict evaluation uses the original reward without the training cost. No generalization or hardware claim.')
    args.output.mkdir(exist_ok=False);(args.output/'comparison.json').write_text(json.dumps(result,indent=2));(args.output/'comparison_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result,indent=2))


if __name__=='__main__':main()
