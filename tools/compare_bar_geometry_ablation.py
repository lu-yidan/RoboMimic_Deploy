"""Compare completed audited blind/oracle arms at identical training budgets."""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from stable_baselines3 import PPO
from tools.continue_vector_full_cycle_training import state_difference, sha

MATCHED=('contract','observation_contract','config','training_cases','evaluation_cases',
    'seed','requested_steps','environments','initial_checkpoint_sha256',
    'parent_checkpoint_sha256','protocol_sha256','reference_sha256',
    'bundle_manifest_sha256','normal_bias_y_deg','force_history_memory','sources',
    'optimizer_initialization')


def validate_pair(manifests,summaries):
    if [m['geometry_mode'] for m in manifests]!=['blind','oracle']:
        raise ValueError('Expected blind then oracle arms')
    for key in MATCHED:
        if manifests[0][key]!=manifests[1][key]:raise ValueError('Unmatched field: '+key)
    for m,s in zip(manifests,summaries):
        if s.get('evaluation_pending') or s.get('stopped_by_request'):
            raise ValueError('Unfinished evaluation or interrupted training')
        if s['actual_steps']!=m['requested_steps'] or s['additional_steps']!=m['requested_steps'] or s['parent_steps']!=0:
            raise ValueError('Unequal or incomplete training budget')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--blind',type=Path,required=True)
    parser.add_argument('--oracle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();dirs=[args.blind,args.oracle]
    manifests=[json.loads((d/'manifest.json').read_text()) for d in dirs]
    summaries=[json.loads((d/'summary.json').read_text()) for d in dirs]
    validate_pair(manifests,summaries)
    torch.set_num_threads(1)
    initials=[PPO.load(d/'policy_before.zip',device='cpu') for d in dirs]
    if state_difference(initials[0].policy.state_dict(),initials[1].policy.state_dict())!=0:
        raise ValueError('Initial actor/critic differ')
    if any(p.policy.optimizer.state for p in initials):raise ValueError('Expected empty initial optimizers')
    for key,value in initials[0].policy.state_dict().items():
        if key.endswith('geometry_projection.weight') and torch.count_nonzero(value):
            raise ValueError('Expected zero initial geometry weights')
    audits=[];coverage=[]
    for d,m,s in zip(dirs,manifests,summaries):
        audit=json.loads((d/'independent_training_audit.json').read_text())
        if not audit['passed']:raise ValueError('Independent audit failed')
        for filename,key in [('manifest.json','experiment_manifest_sha256'),('summary.json','experiment_summary_sha256'),('policy.zip','checkpoint_sha256')]:
            if sha(d/filename)!=audit[key]:raise ValueError('Evidence changed after audit')
        if sha(d/'matched_protocol.json')!=m['protocol_sha256']:raise ValueError('Protocol changed')
        expected=set(m['evaluation_cases'])
        for label in ('zero','after'):
            rows=[r for r in audit['trajectories'] if r['label']==label]
            if len(rows)!=len(expected) or {r['case'] for r in rows}!=expected:raise ValueError('Incomplete audit case coverage')
            for r in rows:
                if sha(d/f"{label}_{r['case']}.npz")!=r['trajectory_sha256']:raise ValueError('Audited trajectory changed')
        audits.append(audit)
        episodes=[json.loads(line) for line in (d/'training_episodes.jsonl').read_text().splitlines()]
        counts=Counter(e['case'] for e in episodes)
        coverage.append(dict(mode=m['geometry_mode'],completed=dict(counts),
            successful=dict(Counter(e['case'] for e in episodes if e['success'])),
            missing=sorted(set(m['training_cases'])-set(counts)),case_samples=s['case_samples']))
    cases=[]
    for case in manifests[0]['evaluation_cases']:
        row=dict(case=case,in_training_set=case in manifests[0]['training_cases'])
        for mode,audit,s in zip(('blind','oracle'),audits,summaries):
            a=next(r for r in audit['trajectories'] if r['label']=='after' and r['case']==case)
            r=next(r for r in s['after'] if r['case']==case)
            row[mode]=dict(accepted=a['independently_accepted'],reason=r['reason'],time_s=r['time_s'],
                endpoints=a['independent_endpoints'],sampled_native_margin_rad=a['sampled_minimum_margin_rad'])
        cases.append(row)
    result=dict(blind_accepted=sum(r['blind']['accepted'] for r in cases),oracle_accepted=sum(r['oracle']['accepted'] for r in cases),
        fixed_cases=[r['case'] for r in cases if r['oracle']['accepted'] and not r['blind']['accepted']],
        regressed_cases=[r['case'] for r in cases if r['blind']['accepted'] and not r['oracle']['accepted']],
        initial_parameters_exact=True,steps_per_arm=manifests[0]['requested_steps'],cases=cases,coverage=coverage,
        evidence={str(d):{f:sha(d/f) for f in ['manifest.json','summary.json','policy_before.zip','policy.zip','independent_training_audit.json','matched_protocol.json']} for d in dirs},
        source_sha256=sha(__file__),scope='Single-seed known-case ablation, equal architecture and budget. Oracle uses privileged geometry; no vision or blind generalization claim. Shorter failed trajectories are not scored as lower peak force improvements.')
    args.output.mkdir(exist_ok=False)
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2))
    (args.output/'comparison_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:result[k] for k in ['blind_accepted','oracle_accepted','fixed_cases','regressed_cases']}))


if __name__=='__main__':main()
