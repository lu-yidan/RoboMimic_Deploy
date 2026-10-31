"""Bind planner provenance, same-policy intervention, prefix and policy audits."""
import argparse
from pathlib import Path
import numpy as np
from training.wall_support_env import WallSupportEnv
from training.swing_foot_lift import refine_swing_foot_lift
from tools.validate_swing_reference import ROOT,read,write,sha
from tools.compare_audited_closure_candidates import load_run


def audit(out):
    out=Path(out).resolve();m=read(out/'manifest.json');plan=Path(m['reference'])
    pm=read(plan.parent/'manifest.json');pa=read(plan.parent/'audit.json')
    assert sha(plan)==m['reference_sha256']==pm['plan_sha256']
    assert sha(plan.parent/'manifest.json')==m['reference_manifest_sha256']
    assert sha(plan.parent/'audit.json')==m['reference_audit_sha256']==pm['audit_sha256'] and pa['passed']
    assert all(sha(plan.parent/'source_snapshot'/k)==v for k,v in pm['sources'].items())
    assert sha(ROOT/'training/swing_foot_lift.py')==pm['sources']['training/swing_foot_lift.py']
    parent=Path(m['parent_experiment']);old=load_run(parent);new=load_run(out)
    assert old['checkpoint_sha256']==new['checkpoint_sha256']==m['parent_checkpoint_sha256']
    assert old['cases']==new['cases'] and old['controller_modifier']==new['controller_modifier']
    assert old['adaptive_reference_command']==new['adaptive_reference_command']
    om=read(parent/'manifest.json')
    assert m['prefix_checkpoint_sha256']==om['prefix_checkpoint_sha256']==sha(m['prefix_checkpoint'])
    assert m['closure_action_bias']==om['closure_action_bias']
    assert sha(pm['parent_plan'])==pm['parent_sha256']==old['reference_sha256']
    # Verify the inherited planner lineage, including the intervening right-foot
    # support adjustment. We refine the final plan, not an earlier superseded plan.
    lineage=[];cursor=Path(pm['parent_plan'])
    while True:
        cm=read(cursor.parent/'manifest.json');ca=read(cursor.parent/'audit.json')
        assert sha(cursor)==cm['plan_sha256'] and ca['passed']
        lineage.append(dict(plan=str(cursor),sha256=sha(cursor),manifest_sha256=sha(cursor.parent/'manifest.json'),audit_sha256=sha(cursor.parent/'audit.json')))
        p=cm.get('parent_plan',cm.get('parent'))
        if not p:break
        cursor=Path(p);cursor=cursor if cursor.is_absolute() else ROOT/cursor
        assert sha(cursor)==cm['parent_sha256']
    options=pm['planner_options'];env=WallSupportEnv(ROOT/'outputs/wall_support_bundle_pd15_v2');env.reset()
    original=dict(np.load(pm['parent_plan']))
    rebuilt,_=refine_swing_foot_lift(env,original,foot_lift=options['foot_lift_m'],
        swing_start=options['swing_interval_s'][0],swing_end=options['swing_interval_s'][1],
        envelope=options['envelope'],taper_start=options['taper_start_s'] or 77.56,
        taper_end=options['taper_end_s'] or 80.66,joint_margin=options['joint_margin_rad'])
    saved=dict(np.load(plan));exact={k:bool(np.array_equal(saved[k],v)) for k,v in rebuilt.items()}
    assert all(exact.values());env.close()
    prefix_reference_unchanged=bool(np.array_equal(saved['qpos'][saved['time']<=70.2],original['qpos'][original['time']<=70.2]))
    prefix=read(out/'composite_prefix_audit_v2.json');assert prefix['passed']
    assert len(prefix['rows'])==len(m['evaluation_cases'])
    assert {r['case'] for r in prefix['rows']}==set(m['evaluation_cases'])
    prefix_boundary_reward_difference=0.
    if prefix_reference_unchanged:
        # Independently bind the older prefix auditor's output to both audited
        # trajectory sets, since its historical report does not contain hashes.
        n=3510
        for case in m['evaluation_cases']:
            with np.load(parent/f'after_{case}.npz') as a,np.load(out/f'after_{case}.npz') as b:
                for key in ('qpos','qvel','time_s','observations','actions','torques'):
                    count=n+1 if key in ('qpos','qvel','time_s') else n
                    assert np.array_equal(a[key][:count],b[key][:count]),(case,key)
                # The boundary reward uses accumulated simulator time slightly
                # beyond 70.2 s, hence the new reference. Record the difference;
                # require exact reward replay independently on the new reference.
                prefix_boundary_reward_difference=max(prefix_boundary_reward_difference,float(abs(a['rewards'][:n]-b['rewards'][:n]).max()))
    if 'reference_sha256' in prefix:
        assert prefix['source_sha256']==m['sources']['tools/validate_swing_reference.py']
        assert prefix['reference_sha256']==m['reference_sha256']
        assert prefix['manifest_sha256']==sha(out/'manifest.json') and prefix['summary_sha256']==sha(out/'summary.json')
        for row in prefix['rows']:
            assert row['passed'] and row['prefix_steps']==3510 and all(row['checks'].values())
            assert row['full_trajectory_sha256']==sha(out/f"after_{row['case']}.npz")
            assert row['trajectory_sha256']==sha(out/'prefix'/f"{row['case']}.npz")
    else:
        assert prefix_reference_unchanged and prefix_boundary_reward_difference==0., 'Fresh new-reference prefix replay required'
    reports={}
    for name in ('independent_candidate_audit.json','controller_modifier_audit_v2.json','adaptive_reference_audit.json','physical_v2_audit.json','composite_prefix_audit_v2.json'):
        r=read(out/name)
        if 'passed' in r:assert r['passed'],name
        if 'manifest_sha256' in r:assert r['manifest_sha256']==sha(out/'manifest.json')
        if 'summary_sha256' in r:assert r['summary_sha256']==sha(out/'summary.json')
        reports[name]=sha(out/name)
    rows=[]
    for case in old['cases']:
        a,b=old['rows'][case],new['rows'][case]
        rows.append(dict(case=case,old_complete=a['complete'],new_complete=b['complete'],
            old_closure_mrad=a['closure_mrad'],new_closure_mrad=b['closure_mrad'],
            delta_mrad=b['closure_mrad']-a['closure_mrad'] if a['complete'] and b['complete'] else None,
            old_strict_success=a['success'],new_strict_success=b['success']))
    def distribution(values):
        return dict(n=len(values),mean=float(np.mean(values)),median=float(np.median(values)),p95=float(np.percentile(values,95)),maximum=float(max(values)))
    physical=read(out/'physical_v2_audit.json')
    assert physical['source_sha256']==m['sources']['tools/validate_swing_reference.py']
    result=dict(passed=True,eligible_single_cycle_reference=physical['physical_v2_count']==len(rows),
        no_training=True,policy_sha256=new['checkpoint_sha256'],reference_sha256=sha(plan),reference_reconstruction_exact=exact,
        prefix_reference_unchanged_through_70p2=prefix_reference_unchanged,lineage=lineage,audits=reports,
        old_reference_prefix_reward_max_difference=prefix_boundary_reward_difference,
        physical_v2_count=physical['physical_v2_count'],original_strict_count=physical['original_strict_count'],rows=rows,
        old_completed_distribution_mrad=distribution([r['old_closure_mrad'] for r in rows if r['old_complete']]),
        new_completed_distribution_mrad=distribution([r['new_closure_mrad'] for r in rows if r['new_complete']]),
        new_paired_distribution_mrad=distribution([r['new_closure_mrad'] for r in rows if r['old_complete'] and r['new_complete']]),
        manifest_sha256=sha(out/'manifest.json'),summary_sha256=sha(out/'summary.json'),source_sha256=sha(__file__),
        scope='Formal generated reference and complete single-cycle plan/prefix/policy evidence on 24 development cases. Different references, identical policies/controllers. Screening used development outcomes; no held-out validation, no new training, no eight-cycle/two-metre certification.')
    write(out/'reference_chain_audit.json',result)
    (out/'reference_chain_audit_source.py').write_bytes(Path(__file__).read_bytes())
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    result=audit(a.output)
    print({k:v for k,v in result.items() if k not in ('rows','lineage','audits')})
