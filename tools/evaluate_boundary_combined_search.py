"""Validate a finite pair of joint/reference proposals in full time-zero physics."""
import argparse,copy,json
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.closure_action_bias_policy_v2 import ClosureActionBiasV2Policy
from training.full_cycle_residual_env import FullCycleResidualEnv
from tools.compare_audited_closure_candidates import load_run,sha
from tools.evaluate_closure_action_bias_v2 import run_case_in_fresh_process


def main():
    p=argparse.ArgumentParser()
    for n in ('parent','action-search','proposal','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    parent=load_run(a.parent);m=json.loads((a.parent/'manifest.json').read_text())
    old=json.loads((a.action_search/'summary.json').read_text());op=json.loads((a.action_search/'protocol.json').read_text())
    proposal=json.loads((a.proposal/'proposals.json').read_text())
    assert old['completed'] and old['protocol_sha256']==sha(a.action_search/'protocol.json')
    assert proposal['action_summary_sha256']==sha(a.action_search/'summary.json')
    assert parent['audit_sha256']==op['parent_audit_sha256']==proposal['parent_audit_sha256']
    for f,h in op['sources'].items():assert sha(f)==h
    chosen=next(r for r in proposal['models'] if r['name']=='common' and r['action_bias_cap']==.5)
    assert chosen['optimization_success'] and chosen['predicted_minimum_margin']>0
    center=np.asarray(m['adaptive_reference_command']);target=np.asarray(chosen['reference_command'])
    commands=[]
    for fraction in (.5,1.):
        commands.append(dict(label=f'combined_{fraction:g}',reference_command=(center+fraction*(target-center)).tolist(),
            tied_action_bias=float(-.2+fraction*(chosen['tied_action_bias']+.2))))
    sources={f:sha(f) for f in [str(Path(__file__).relative_to(Path.cwd())),
        'tools/evaluate_closure_action_bias_v2.py','training/closure_action_bias_policy_v2.py']}
    sources.update({f:h for f,h in m['sources'].items() if f.replace('\\','/').startswith('training/')})
    for f,h in sources.items():assert sha(f)==h
    protocol=dict(kind='combined_action_reference',parent=str(a.parent.resolve()),parent_audit_sha256=parent['audit_sha256'],
        checkpoint_sha256=parent['checkpoint_sha256'],cases=op['cases'],sources=sources,start_seconds=70.2,
        commands=commands,proposal_sha256=sha(a.proposal/'proposals.json'),action_search=str(a.action_search.resolve()),
        action_summary_sha256=sha(a.action_search/'summary.json'),scope='Two predefined full-cycle candidates. Fixed action calibration and reference commands, not PPO learning. Final normalized actions remain clipped to [-1,1], residual limit remains 0.015 rad, original task acceptance unchanged. Local fit is not a success claim.')
    a.output.mkdir(exist_ok=False);(a.output/'protocol.json').write_text(json.dumps(protocol,indent=2));(a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    def manifest(command):
        result=copy.deepcopy(m);bias=np.zeros(26);bias[3:5]=command['tied_action_bias']
        result.update(policy_adapter='closure-action-bias-v2',adaptive_reference_command=command['reference_command'],
            requested_reference_command=command['reference_command'],closure_action_bias=dict(normalized_bias=bias.tolist(),start_seconds=70.2,ramp_seconds=3.,geometry_gate='existing_high_rod'))
        return result
    # Reuse the audited-in-search anchor only after replaying every saved action.
    anchor=[r for r in old['rows'] if r['label']=='common_1'];assert {r['case'] for r in anchor}==set(protocol['cases'])
    am=manifest(dict(tied_action_bias=-.2,reference_command=center.tolist()))
    actor=ClosureActionBiasV2Policy(PPO.load(a.parent/'policy.zip',device='cpu'),am);gate=[]
    for row in anchor:
        assert sha(row['file'])==row['trajectory_sha256']
        with np.load(row['file']) as z:
            predicted=np.asarray([actor.predict(obs,deterministic=True)[0] for obs in z['observations']])
            assert np.array_equal(predicted,z['actions'])
        gate.append(dict(case=row['case'],passed=True,trajectory_sha256=row['trajectory_sha256']))
    (a.output/'adapter_gate.json').write_text(json.dumps(dict(passed=True,rows=gate,fresh_physics=False,scope='Every recorded anchor action exactly replayed by v2. No claim of a new anchor simulation.'),indent=2))
    del actor
    env=FullCycleResidualEnv(plan=m['reference'],cases=m['evaluation_cases']);qadr=env.env.qadr.copy();initial=env.plan['qpos'][0].copy();env.close()
    rows=[];groups=[]
    for command in commands:
        if (a.output/'STOP_REQUESTED').exists():return
        label=command['label'];directory=a.output/label;directory.mkdir();cm=manifest(command)
        def job(case):return case,run_case_in_fresh_process((cm,str((a.parent/'policy.zip').resolve()),parent['checkpoint_sha256'],case,str((directory/case).resolve())))
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(job,c) for c in protocol['cases']]
            for future in as_completed(futures):
                case,info=future.result();path=directory/case/f'after_{case}.npz'
                with np.load(path) as z,np.load(a.parent/f'after_{case}.npz') as before:
                    for k in ('qpos','qvel','time_s','observations','actions','torques','rewards'):
                        n=3510+(1 if k in ('qpos','qvel','time_s') else 0);assert np.array_equal(z[k][:n],before[k][:n])
                    complete=abs(z['time_s'][-1]-113.66)<1e-6
                    tail=(z['time_s']>111.66)&(z['time_s']<=113.6600001)
                    closure=float(abs(z['qpos'][tail][:,qadr]-initial[qadr]).max()) if complete else None
                    rise=z['qpos'][tail,2]-initial[2]
                    margin=min((.04-closure)/.001,(.01-abs(rise-.25).max())/.0001) if complete else -1e9
                row=dict(label=label,case=case,success=bool(info['success']),complete=bool(complete),minimum_margin=float(margin),
                    closure_mrad=closure*1000 if complete else None,rise_max_mm=float(rise.max()*1000) if complete else None,file=str(path.resolve()),trajectory_sha256=sha(path))
                rows.append(row);(a.output/'progress.json').write_text(json.dumps(dict(completed=False,rows=rows),indent=2));print(json.dumps(row),flush=True)
        group=[r for r in rows if r['label']==label]
        groups.append(dict(**command,successes=sum(r['success'] for r in group),minimum_margin=min(r['minimum_margin'] for r in group),neighbor_preserved=next(r['success'] for r in group if r['case']=='boundary_04')))
        if groups[-1]['successes']==len(protocol['cases']):break
    for f,h in sources.items():assert sha(f)==h
    best=max(groups,key=lambda r:(r['neighbor_preserved'],r['successes'],r['minimum_margin']))
    result=dict(completed=True,rows=rows,groups=groups,best=best,protocol_sha256=sha(a.output/'protocol.json'),
        unrun=[c['label'] for c in commands if c['label'] not in [g['label'] for g in groups]],scope=protocol['scope'])
    (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(completed=True,best=best)),flush=True)


if __name__=='__main__':main()
