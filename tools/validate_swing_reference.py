"""Formal reference intervention: frozen policies, fresh physics and prefix audit.

No training. Original strict success is retained. Separate physical-v2 results
use the declared 1 mm rise deadband, not an inferred numerical-noise estimate.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import mujoco
import torch
from tools.evaluate_closure_action_bias_v2 import ROOT, sha
from training.strict_full_cycle_residual_env import FullCycleResidualEnv as StrictEnv
from training.full_cycle_residual_env import GROUPS


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, data):
    Path(p).write_text(json.dumps(data,indent=2),encoding='utf-8')


class ObservedStrictEnv(StrictEnv):
    """Only observe contact metrics; leave stepping and original gates unchanged."""
    def reset(self, **kwargs):
        self.physical_tail = []
        return super().reset(**kwargs)

    def check_endpoint(self):
        e, d = self.env, self.env.data
        index = len(self.cycle_results)
        if self.ends[index]-2 < d.time <= self.ends[index]+1e-7:
            metrics = e.contact_metrics()
            expected = {g:f'rung_{(4 if "hook" in g else 0)+index+1:02d}' for g in GROUPS}
            loaded = [c for c in metrics['contacts'] if np.linalg.norm(c['force_on_robot_world_n'])>1.]
            normals={g:0. for g in GROUPS}
            for ci,contact in enumerate(d.contact):
                ba,bb=[int(e.model.geom_bodyid[g]) for g in (contact.geom1,contact.geom2)]
                if (ba in e.robot_bodies)==(bb in e.robot_bodies):continue
                group=e.contact_groups.get(ba if ba in e.robot_bodies else bb)
                if group in normals:
                    wrench=np.zeros(6);mujoco.mj_contactForce(e.model,d,ci,wrench)
                    normals[group]+=float(max(0.,wrench[0]))
            self.physical_tail.append(dict(time_s=float(d.time),rise_m=float(d.qpos[2]-e.initial[2]),
                support_normal_n=normals,
                support_vertical_n={g:float(metrics[g+'_vertical_n']) for g in GROUPS},
                intended_contacts=bool(loaded) and all(c['group'] in expected and c['external_geom']==expected[c['group']] for c in loaded)))
        return super().check_endpoint()

    def info(self, reason, success):
        result = super().info(reason,success)
        if self.finished:
            result['physical_tail_observations'] = self.physical_tail
        return result


def prepare(parent, plan, out):
    parent,plan,out = Path(parent).resolve(),Path(plan).resolve(),Path(out).resolve()
    pm,pa = read(plan.parent/'manifest.json'),read(plan.parent/'audit.json')
    assert pa['passed'] and sha(plan)==pm['plan_sha256']
    assert sha(plan.parent/'audit.json')==pm['audit_sha256']
    assert all(sha(plan.parent/'source_snapshot'/k)==v for k,v in pm['sources'].items())
    old = read(parent/'manifest.json')
    assert pm['parent_sha256']==old['reference_sha256']
    keys = ('contract','observation_contract','geometry_mode','evaluation_cases',
        'normal_bias_y_deg','force_history_memory','bundle_manifest_sha256',
        'suffix_start_seconds','horizon_seconds','prefix_checkpoint_sha256',
        'controller_modifier','adaptive_reference_command','adaptive_reference_scales',
        'policy_adapter','closure_action_bias')
    m = {k:old[k] for k in keys}
    out.mkdir(parents=True,exist_ok=False)
    for name,source in [('policy.zip',parent/'policy.zip'),('prefix_policy.zip',Path(old['prefix_checkpoint']))]:
        shutil.copy2(source,out/name)
    assert sha(out/'policy.zip')==read(parent/'summary.json')['checkpoint_sha256']
    assert sha(out/'prefix_policy.zip')==old['prefix_checkpoint_sha256']
    sources = {}
    for rel,expected in old['sources'].items():
        src = parent/'source_snapshot'/rel
        assert sha(src)==expected
        if rel.replace('\\','/').startswith('training/'):
            assert sha(ROOT/rel)==expected, rel
        dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dest);sources[rel]=expected
    # These are the tools actually executing this intervention, distinct from
    # the parent's historical training tool snapshots.
    for rel in ('tools/validate_swing_reference.py','tools/evaluate_closure_action_bias_v2.py'):
        dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/rel,dest);sources[rel]=sha(dest)
    m.update(reference=str(plan),reference_sha256=sha(plan),sources=sources,
        prefix_checkpoint=str(out/'prefix_policy.zip'),parent_experiment=str(parent),
        parent_manifest_sha256=sha(parent/'manifest.json'),parent_checkpoint_sha256=sha(out/'policy.zip'),
        reference_manifest_sha256=sha(plan.parent/'manifest.json'),reference_audit_sha256=sha(plan.parent/'audit.json'),
        evaluation_only=True,new_training_steps=0,require_inactive_case_exact_parent=False,
        inactive_case_comparison_scope='Reference intervention changes all geometry cases; old-reference bitwise equality is not expected after lift start.',
        scope='Frozen prefix/suffix policies and fixed controllers; only the nominal swing reference changes. 24 development/guard cases, no held-out validation, no new training, one cycle only.')
    write(out/'manifest.json',m)
    write(out/'summary.json',dict(evaluation_pending=True,checkpoint_sha256=sha(out/'policy.zip'),additional_steps=0))
    return out


def worker(out,case,kind):
    out=Path(out).resolve();m=read(out/'manifest.json')
    for rel,digest in m['sources'].items():
        assert sha(out/'source_snapshot'/rel)==digest
        if rel.startswith('training/') or rel=='tools/validate_swing_reference.py':
            assert sha(ROOT/rel)==digest, rel
    assert sha(m['reference'])==m['reference_sha256']
    torch.set_num_threads(1)
    if kind=='eval':
        import tools.evaluate_closure_action_bias_v2 as evaluator
        evaluator.StrictEnv=ObservedStrictEnv
        evaluator.evaluate_case((m,str(out/'policy.zip'),sha(out/'policy.zip'),case,str(out/'eval'/case)))
    else:
        from training.live_prefix_landing_env import LivePrefixLandingEnv
        env=LivePrefixLandingEnv(out,cases={case:m['evaluation_cases'][case]},record=True)
        try:
            env.reset(options={'case':case})
            arrays={k:np.asarray(v) for k,v in env.prefix_trace.items()}
            with np.load(out/f'after_{case}.npz') as z:
                exact={k:bool(np.array_equal(v,z['rewards' if k=='base_rewards' else k][:len(v)])) for k,v in arrays.items()}
                differences={k:float(abs(v-z['rewards' if k=='base_rewards' else k][:len(v)]).max(initial=0)) for k,v in arrays.items()}
            dest=out/'prefix'/f'{case}.npz';np.savez_compressed(dest,**arrays)
            write(out/'prefix'/f'{case}.json',dict(case=case,passed=all(exact.values()),checks=exact,
                maximum_differences=differences,prefix_steps=env.prefix_steps,prefix_seconds=70.2,
                trajectory_sha256=sha(dest),full_trajectory_sha256=sha(out/f'after_{case}.npz')))
        finally:env.close()


def run(out,cases=None,kind='eval',workers=4):
    out=Path(out).resolve();m=read(out/'manifest.json');cases=cases or list(m['evaluation_cases'])
    assert 1<=workers<=4 and len(set(cases))==len(cases) and set(cases)<=set(m['evaluation_cases'])
    (out/kind).mkdir(exist_ok=True)
    def one(case):
        expected=out/kind/(case+'/result.json' if kind=='eval' else case+'.json')
        if expected.exists():
            return read(expected)
        with (out/kind/f'{case}.log').open('w',encoding='utf-8') as stream:
            subprocess.run([sys.executable,'-m','tools.validate_swing_reference','worker',
                '--out',str(out),'--case',case,'--kind',kind],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        return read(expected)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one,c) for c in cases]):
            row=future.result()
            print(json.dumps(dict(case=row['case'],kind=kind,result=row.get('reason',row.get('passed')))),flush=True)
    if kind=='eval' and all((out/'eval'/c/'result.json').exists() for c in m['evaluation_cases']):
        rows=[]
        for c in m['evaluation_cases']:
            row=read(out/'eval'/c/'result.json');rows.append(row)
            src=out/'eval'/c/f'after_{c}.npz';assert sha(src)==row['trajectory_sha256']
            shutil.copy2(src,out/src.name)
        write(out/'after.json',rows)
        write(out/'summary.json',dict(evaluation_pending=False,after=rows,checkpoint_sha256=sha(out/'policy.zip'),additional_steps=0))
    if kind=='prefix' and all((out/'prefix'/f'{c}.json').exists() for c in m['evaluation_cases']):
        rows=[read(out/'prefix'/f'{c}.json') for c in m['evaluation_cases']]
        result=dict(passed=all(r['passed'] for r in rows),rows=rows,reference_sha256=m['reference_sha256'],
            manifest_sha256=sha(out/'manifest.json'),summary_sha256=sha(out/'summary.json'),source_sha256=sha(__file__),
            scope='Independent time-zero LivePrefixLandingEnv replay on the NEW reference, exactly 3510 steps through 70.2 s; all saved state/action/observation/torque/base-reward arrays must match the full policy run. No old-reference prefix substitution.')
        write(out/'composite_prefix_audit_v2.json',result)
        assert result['passed']


def audit(out):
    from tools.audit_action_bias_candidate_v2 import audit as audit_policy
    from tools.audit_trained_closure_controller_v2 import audit as audit_controller
    from tools.audit_closure_foreaft_reference import audit as audit_reference
    out=Path(out).resolve()
    if not (out/'independent_candidate_audit.json').exists():audit_policy(out)
    if not (out/'controller_modifier_audit_v2.json').exists():audit_controller(out)
    if not (out/'adaptive_reference_audit.json').exists():audit_reference(out)
    m=read(out/'manifest.json');s=read(out/'summary.json');ind=read(out/'independent_candidate_audit.json')
    assert ind['passed'] and ind['experiment_manifest_sha256']==sha(out/'manifest.json') and ind['experiment_summary_sha256']==sha(out/'summary.json')
    reference=np.load(m['reference']);rows=[]
    for r in s['after']:
        endpoint=next(x for x in ind['trajectories'] if x['case']==r['case'])['independent_endpoints'][0]
        tail=r['physical_tail_observations'];peaks=r['peaks']
        with np.load(out/f"after_{r['case']}.npz") as z:
            mask=(z['time_s']>113.66-2)&(z['time_s']<=113.66+1e-7)
            rise=z['qpos'][mask,2]-reference['qpos'][0,2]
            rise_trace=bool(len(rise)==len(tail) and np.max(abs(rise-np.array([x['rise_m'] for x in tail])),initial=0)<1e-12)
        checks=dict(complete=abs(r['time_s']-113.66)<1e-7,
            runtime_clean=r['reason'] in ('success','cycle_endpoint_failure') and max(peaks[k] for k in ('other','self','flight'))<1.,
            native_bounds=r['minimum_native_joint_margin_rad']>=0.,tail_count=len(tail)>=99,
            rise_trace=rise_trace,rise_deadband=bool(len(rise) and np.max(abs(rise-.25))<.011),
            support_normal=bool(tail) and all(min(x['support_normal_n'].values())>10 for x in tail),
            support_vertical=bool(tail) and all(min(x['support_vertical_n'].values())>10 for x in tail),
            intended_contacts=bool(tail) and all(x['intended_contacts'] for x in tail),
            settled=all(endpoint[k] is not None and endpoint[k]<limit for k,limit in
                [('effector_error_m',.01),('tilt_deg',3.),('linear_speed_m_s',.01),('angular_speed_rad_s',.1),('joint_speed_rad_s',.1)]))
        rows.append(dict(case=r['case'],physical_v2_passed=all(checks.values()),checks=checks,
            original_strict_success=r['success'],closure_mrad=endpoint['joint_closure_rad']*1000 if endpoint['joint_closure_rad'] is not None else None,
            rise_max_mm=float(rise.max()*1000) if len(rise) else None,
            minimum_support_normal_n=min((min(x['support_normal_n'].values()) for x in tail),default=None),
            minimum_support_vertical_n=min((min(x['support_vertical_n'].values()) for x in tail),default=None)))
    report=dict(physical_v2_count=sum(r['physical_v2_passed'] for r in rows),original_strict_count=sum(r['original_strict_success'] for r in rows),rows=rows,
        rise_tolerance_m=.01,rise_extra_deadband_m=.001,warning_closure_mrad=35.,
        support_measure='Both summed compressive contact-normal force per group (v2 specification) and world-vertical support per original controller contract must exceed 10 N at every tail sample.',
        scope='Single-cycle physical and settled-posture checks, live 2000 Hz runtime limits plus 50 Hz final two-second contact observations. Joint closure is reported separately. No claim of 8-cycle/two-metre validation.',
        summary_sha256=sha(out/'summary.json'),independent_audit_sha256=sha(out/'independent_candidate_audit.json'),source_sha256=sha(__file__))
    write(out/'physical_v2_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','worker','audit'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--parent',type=Path);p.add_argument('--plan',type=Path)
    p.add_argument('--cases',nargs='+');p.add_argument('--case');p.add_argument('--kind',choices=['eval','prefix'],default='eval');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.parent,a.plan,a.out)
    elif a.command=='run':run(a.out,a.cases,a.kind,a.workers)
    elif a.command=='worker':worker(a.out,a.case,a.kind)
    else:audit(a.out)


if __name__=='__main__':main()
