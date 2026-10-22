"""Verify an explicitly reused matched control against the smooth-cost follow-up."""
import argparse
import json
from pathlib import Path
import torch
from stable_baselines3 import PPO
from tools.continue_vector_full_cycle_training import sha,state_difference

COMMON=('contract','observation_contract','geometry_mode','config','training_cases','evaluation_cases','seed','parent_steps','additional_steps_requested','environments','parent_checkpoint_sha256','reference_sha256','bundle_manifest_sha256','normal_bias_y_deg','force_history_memory','optimizer_initialization')


def validate_pair(manifests,summaries):
    for key in COMMON:
        if manifests[0][key]!=manifests[1][key]:raise ValueError('Unmatched '+key)
    if list(manifests[0]['training_cases'])!=list(manifests[1]['training_cases']):raise ValueError('Case order differs')
    if [m['closure_cost_weight'] for m in manifests]!=[0.,.02]:raise ValueError('Expected zero and smooth-cost arms')
    expected=['strict-v2-minus-terminal-closure-cost-v1','strict-v2-minus-terminal-closure-smooth-cost-v1']
    if [m['training_reward_contract'] for m in manifests]!=expected:raise ValueError('Unexpected reward contracts')
    for m,s in zip(manifests,summaries):
        if s.get('evaluation_pending') or s.get('stopped_by_request'):raise ValueError('Incomplete evaluation')
        if s['additional_steps']!=m['additional_steps_requested'] or s['actual_steps']!=s['parent_steps']+s['additional_steps'] or s['parent_steps']!=m['parent_steps'] or s['cost_steps']!=s['additional_steps']:raise ValueError('Incomplete budget')
        if not 0<=s['maximum_step_closure_cost']<=m['closure_cost_weight'] or not 0<=s['total_training_closure_cost']<=m['closure_cost_weight']*s['additional_steps']+1e-8:raise ValueError('Invalid cost accounting')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    protocol=json.loads((args.candidate/'matched_protocol.json').read_text());control=Path(protocol['control_reuse']);paths=[control,args.candidate]
    for file,digest in protocol['control_reuse_hashes'].items():assert sha(control/file)==digest
    manifests=[json.loads((p/'manifest.json').read_text()) for p in paths];summaries=[json.loads((p/'summary.json').read_text()) for p in paths];validate_pair(manifests,summaries)
    assert manifests[0]['protocol_sha256']==protocol['previous_protocol_sha256'] and manifests[1]['protocol_sha256']==sha(args.candidate/'matched_protocol.json')
    assert protocol['previous_comparison_sha256']==sha('outputs/closure_margin_comparison_v1/comparison.json')
    shared={};source_sets=[{k.replace('\\','/'):v for k,v in m['sources'].items() if k.replace('\\','/').startswith('training/')} for m in manifests]
    allowed={'training/closure_margin_reward.py','training/closure_smooth_reward.py'}
    assert set(source_sets[0])^set(source_sets[1])<=allowed
    for source in set(source_sets[0])&set(source_sets[1]):
        assert source_sets[0][source]==source_sets[1][source];shared[source]=source_sets[0][source]
    assert 'training/full_cycle_residual_env.py' in shared and 'training/strict_full_cycle_residual_env.py' in shared
    torch.set_num_threads(1);initial=[PPO.load(p/'policy_before.zip',device='cpu') for p in paths]
    assert state_difference(initial[0].policy.state_dict(),initial[1].policy.state_dict())==0 and state_difference(initial[0].policy.optimizer.state_dict(),initial[1].policy.optimizer.state_dict())==0
    audits=[]
    for p,m,s in zip(paths,manifests,summaries):
        a=json.loads((p/'independent_training_audit.json').read_text());assert a['passed']
        for file,key in [('manifest.json','experiment_manifest_sha256'),('summary.json','experiment_summary_sha256'),('policy.zip','checkpoint_sha256')]:assert sha(p/file)==a[key]
        for label in ('zero','after'):
            subset=[r for r in a['trajectories'] if r['label']==label];assert len(subset)==len(m['evaluation_cases']) and {r['case'] for r in subset}==set(m['evaluation_cases'])
            for r in subset:assert sha(p/f"{label}_{r['case']}.npz")==r['trajectory_sha256']
        episodes=[json.loads(line) for line in (p/'training_episodes.jsonl').read_text().splitlines()];assert len(episodes)==s['completed_training_episodes']
        for row in episodes:assert abs(row['episode']['r']-(row['base_episode_return']-row['closure_episode_cost']))<1e-5
        audits.append(a)
    rows=[]
    for case in manifests[0]['evaluation_cases']:
        members=[next(r for r in a['trajectories'] if r['label']=='after' and r['case']==case) for a in audits];before,after=[r['independently_accepted'] for r in members]
        rows.append(dict(case=case,control_passed=before,smooth_passed=after,fixed=not before and after,regressed=before and not after,control_endpoints=members[0]['independent_endpoints'],smooth_endpoints=members[1]['independent_endpoints']))
    fixed=[r['case'] for r in rows if r['fixed']];regressed=[r['case'] for r in rows if r['regressed']]
    result=dict(rows=rows,control_passed=sum(r['control_passed'] for r in rows),smooth_passed=sum(r['smooth_passed'] for r in rows),fixed=fixed,regressed=regressed,selected=bool(fixed and not regressed),shared_physical_sources=shared,audit_hashes=[sha(p/'independent_training_audit.json') for p in paths],source_sha256=sha(__file__),scope=protocol['scope'])
    args.output.mkdir(exist_ok=False);(args.output/'comparison.json').write_text(json.dumps(result,indent=2));(args.output/'comparison_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({k:result[k] for k in ('control_passed','smooth_passed','fixed','regressed','selected')},indent=2))


if __name__=='__main__':main()
