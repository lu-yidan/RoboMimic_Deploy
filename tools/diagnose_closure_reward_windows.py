"""Offline reward reconstruction on audited deterministic complete trajectories."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.strict_full_cycle_residual_env import FullCycleResidualEnv
from training.closure_quality_reward import quality_bonus
from tools.compare_audited_closure_candidates import load_run, sha


def main():
    parser=argparse.ArgumentParser()
    for name in ('before','after','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    runs={name:load_run(getattr(args,name)) for name in ('before','after')}
    assert runs['before']['cases']==runs['after']['cases']
    assert runs['before']['reference_sha256']==runs['after']['reference_sha256']
    manifest=json.loads((args.after/'manifest.json').read_text())
    assert manifest['quality_bonus_weight']==.1
    env=FullCycleResidualEnv(plan=manifest['reference'],cases=manifest['evaluation_cases'])
    rows=[]
    try:
        for case in manifest['evaluation_cases']:
            env.reset(options={'case':case})
            row=dict(case=case)
            for name in ('before','after'):
                with np.load(getattr(args,name)/f'after_{case}.npz') as trace:
                    t=trace['time_s'][1:];q=trace['qpos'][1:]
                    closure=abs(q[:,env.env.qadr]-env.env.initial[env.env.qadr]).max(axis=1)
                    rise=q[:,2]-env.env.initial[2]
                    u=np.clip((t-99.66)/3.,0.,1.);ramp=u*u*(3-2*u)
                    bonus=np.array([quality_bonus(c,z,r,weight=.1) for c,z,r in zip(closure,rise,ramp)])
                    suffix=t>99.660001;tail=t>111.660001
                    assert suffix.sum()==700 and tail.sum()==100
                    # Audit includes the stored boundary state (101 samples);
                    # reward windows count only the 100 subsequent transitions.
                    audit_tail=t>111.66
                    assert abs(closure[audit_tail].max()*1000-runs[name]['rows'][case]['closure_mrad'])<1e-8
                    row[name]=dict(suffix_base_return=float(trace['rewards'][suffix].sum()),
                        suffix_bonus=float(bonus[suffix].sum()),tail_bonus=float(bonus[tail].sum()),
                        earlier_bonus=float(bonus[suffix & ~tail].sum()),
                        reconstructed_suffix_return=float((trace['rewards'][suffix]+bonus[suffix]).sum()))
            row['changes']={key:row['after'][key]-row['before'][key] for key in row['before']}
            rows.append(row)
    finally:
        env.close()
    result=dict(rows=rows,before_audit_sha256=runs['before']['audit_sha256'],after_audit_sha256=runs['after']['audit_sha256'],
        source_sha256=sha(__file__),reward_source_sha256=sha('training/closure_quality_reward.py'),
        scope='Offline deterministic reward reconstruction, not an on-policy training-return comparison. Same unchanged positive quality bonus; no inference of causal PPO failure from this diagnostic alone.')
    args.output.mkdir(exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2))
    (args.output/'diagnostic_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
