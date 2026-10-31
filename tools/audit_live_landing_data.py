"""Check sampled data against the frozen seed, parent entry state and reward code."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from training.closure_quality_reward import quality_bonus
from tools.compare_audited_closure_candidates import load_run,sha


def main():
    p=argparse.ArgumentParser()
    for key in ('dataset','seed','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--partial',action='store_true',help='Audit only completed episode files; cannot authorize training.')
    a=p.parse_args();torch.set_num_threads(1)
    protocol=json.loads((a.dataset/'protocol.json').read_text())
    summary=(dict(passed=True,protocol_sha256=sha(a.dataset/'protocol.json'),rows=[json.loads(f.read_text()) for f in sorted(a.dataset.glob('episode_*.json'))])
             if a.partial else json.loads((a.dataset/'summary.json').read_text()))
    assert summary['passed'] and summary['protocol_sha256']==sha(a.dataset/'protocol.json')
    assert protocol['seed_checkpoint_sha256']==sha(a.seed/'policy_initial.zip')
    for path,value in protocol['sources'].items():assert sha(path)==value
    parent=Path(protocol['parent']);verified=load_run(parent)
    assert verified['audit_sha256']==protocol['parent_audit_sha256']
    m=json.loads((parent/'manifest.json').read_text());model=PPO.load(a.seed/'policy_initial.zip',device='cpu')
    selected={(r['case'],r['split']) for r in summary['rows']};expected={(c,s) for c in m['evaluation_cases'] for s in ('train','validation','test')}
    assert summary['rows'] and len(selected)==len(summary['rows']) and selected<=expected
    if not a.partial:assert len(summary['rows'])==24 and selected==expected
    env=FullCycleResidualEnv(plan=m['reference'],cases=m['evaluation_cases']);rows=[]
    try:
        for row in summary['rows']:
            case=row['case'];path=a.dataset/row['file'];assert sha(path)==row['trajectory_sha256']
            assert row['seed']==20301024+row['index']
            with np.load(path) as z:data={k:v.copy() for k,v in z.items()}
            with np.load(parent/f'after_{case}.npz') as z:
                initial_q=z['qpos'][3510].copy();initial_v=z['qvel'][3510].copy();previous=z['actions'][3509].astype(float)
            n=row['steps'];q=data['qpos'];v=data['qvel'];ts=data['time_s'];actions=data['actions'];obs=data['observations']
            env.reset(options={'case':case});e=env.env
            checks=dict(finite=all(np.isfinite(v).all() for v in data.values()),
                        shapes=q.shape==(n+1,e.model.nq) and v.shape==(n+1,e.model.nv) and obs.shape==(n,148) and actions.shape==(n,26),
                        exact_entry_qpos=np.array_equal(q[0],initial_q),exact_entry_qvel=np.array_equal(v[0],initial_v),
                        clock=abs(ts[0]-70.2)<1e-7 and np.all(abs(np.diff(ts)[:-1]-.02)<1e-8) and 0<ts[-1]-ts[-2]<=.02000001,
                        action_bounds=float(abs(actions).max())<=1.,
                        success_clock=not row['success'] or abs(ts[-1]-113.66)<1e-7)
            torch.manual_seed(row['seed']);np.random.seed(row['seed'])
            replay=np.asarray([model.predict(o,deterministic=False)[0] for o in obs])
            checks['stochastic_action_replay_exact']=np.array_equal(actions,replay)
            base_error=bonus_error=tracking_error=0.
            closure_indices=np.flatnonzero(env.plan['phase']=='posture_closure');start=float(env.plan['time'][closure_indices[0]-1])
            for i in range(n):
                reference=env.reference(float(ts[i]))[0]
                tracking_error=max(tracking_error,float(abs(obs[i,:29]-(q[i,e.qadr]-reference[e.qadr])).max()))
                end_reference=env.reference(float(ts[i+1]))[0]
                reward=.02-2*np.mean((q[i+1,e.qadr]-end_reference[e.qadr])**2)-.001*np.mean(actions[i].astype(float)**2)-.001*np.mean((actions[i].astype(float)-previous)**2)
                if i==n-1:reward+=20. if row['success'] else -20.
                previous=actions[i].astype(float);base_error=max(base_error,abs(float(data['base_rewards'][i])-reward))
                u=float(np.clip((ts[i+1]-start)/3.,0.,1.));ramp=u*u*(3-2*u)
                bonus=quality_bonus(float(abs(q[i+1,e.qadr]-e.initial[e.qadr]).max()),float(q[i+1,2]-e.initial[2]),ramp,weight=.1)
                bonus_error=max(bonus_error,abs(float(data['rewards'][i]-data['base_rewards'][i])-bonus))
            checks.update(base_reward=base_error<1e-7,quality_bonus=bonus_error<1e-10,tracking_observation=tracking_error<1e-7,
                          velocity_observation=float(abs(obs[:,29:58]-v[:-1,e.vadr]).max())<1e-6)
            rows.append(dict(case=case,index=row['index'],split=row['split'],passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},
                             base_reward_max_error=base_error,quality_bonus_max_error=bonus_error,tracking_max_error=tracking_error,trajectory_sha256=sha(path)))
        result=dict(passed=all(r['passed'] for r in rows),complete_dataset=not a.partial,rows=rows,dataset_protocol_sha256=sha(a.dataset/'protocol.json'),dataset_summary_sha256=None if a.partial else sha(a.dataset/'summary.json'),
                    seed_checkpoint_sha256=sha(a.seed/'policy_initial.zip'),source_sha256=sha(__file__),
                    scope='24 known-case stochastic episodes. Exact seed action replay and physical entry-state checks, reconstructed reward and observations. Not independent dynamics replay or unseen-geometry validation.')
        a.output.mkdir(exist_ok=False);(a.output/'summary.json').write_text(json.dumps(result,indent=2));(a.output/'source.py').write_bytes(Path(__file__).read_bytes())
        print(json.dumps(dict(passed=result['passed'],episodes=len(rows))));assert result['passed']
    finally:env.close()


if __name__=='__main__':main()
