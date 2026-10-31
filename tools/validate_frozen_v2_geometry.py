"""Freeze a separate geometry-validation set without fitting or retraining.

Reuses the formal single-cycle evaluator, live prefix replay and independent
controller/action audits. The already adopted reference is copied by identity,
not treated as a newly optimized planner intervention. Validation outcomes must
not be used for selecting this frozen candidate or adjusting its parameters.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tools.validate_swing_reference import ROOT,read,write,sha,run as evaluate,audit as audit_single
from tools.compare_audited_closure_candidates import load_run


def prepare(candidate,geometry_protocol,out):
    candidate=candidate.resolve();out=out.resolve();geometry_protocol=geometry_protocol.resolve()
    parent=load_run(candidate);m=read(candidate/'manifest.json');chain=read(candidate/'reference_chain_audit.json')
    geometry=read(geometry_protocol)
    assert chain['passed'] and chain['eligible_single_cycle_reference']
    assert chain['manifest_sha256']==sha(candidate/'manifest.json')
    assert chain['summary_sha256']==sha(candidate/'summary.json')
    assert parent['cases']==geometry['training_cases']
    cases=geometry['validation_cases']
    assert len(cases)==16 and not(set(cases)&set(parent['cases']))
    assert all(value not in parent['cases'].values() for value in cases.values())
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(candidate/'source_snapshot',out/'source_snapshot')
    shutil.copy2(candidate/'policy.zip',out/'policy.zip')
    shutil.copy2(m['prefix_checkpoint'],out/'prefix_policy.zip')
    shutil.copy2(geometry_protocol,out/'geometry_protocol.json')
    protocol=dict(candidate=str(candidate),candidate_chain_audit_sha256=sha(candidate/'reference_chain_audit.json'),
        candidate_manifest_sha256=sha(candidate/'manifest.json'),candidate_checkpoint_sha256=parent['checkpoint_sha256'],
        candidate_reference_sha256=parent['reference_sha256'],geometry_protocol_sha256=sha(geometry_protocol),
        cases=cases,case_role='Previously reserved local validation geometries, separate from the 24 development cases. This set may have historical evaluations; do not claim pristine or broad-scene generalization.',
        selection='One frozen policy/reference/controller, all sixteen cases retained; no tuning, retry-based selection, or automatic baseline promotion.',
        physical_acceptance='Existing physical-v2 plus settled endpoint posture; closure is quality-only.',new_training_steps=0,
        source_sha256=sha(__file__))
    write(out/'validation_protocol.json',protocol)
    sources=dict(m['sources'])
    for rel in ('tools/validate_swing_reference.py','tools/validate_frozen_v2_geometry.py'):
        target=out/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/rel,target);sources[rel]=sha(target)
    for rel,digest in sources.items():
        assert sha(out/'source_snapshot'/rel)==digest
        if rel.replace('\\','/').startswith('training/'):assert sha(ROOT/rel)==digest
    m.update(evaluation_cases=cases,training_cases={},original_development_cases=parent['cases'],sources=sources,
        prefix_checkpoint=str(out/'prefix_policy.zip'),parent_experiment=str(candidate),parent_manifest_sha256=sha(candidate/'manifest.json'),
        parent_checkpoint_sha256=parent['checkpoint_sha256'],candidate_chain_audit_sha256=protocol['candidate_chain_audit_sha256'],
        validation_protocol_sha256=sha(out/'validation_protocol.json'),evaluation_only=True,new_training_steps=0,
        require_inactive_case_exact_parent=False,case_role=protocol['case_role'],
        scope='Frozen adopted v2 reference transferred to all sixteen predeclared local validation geometries. Fresh time-zero physics and independent same-reference prefix replay; no new training, tuning, selection or automatic promotion.')
    write(out/'manifest.json',m)
    write(out/'summary.json',dict(evaluation_pending=True,checkpoint_sha256=sha(out/'policy.zip'),additional_steps=0,new_training_steps=0))
    print(json.dumps(dict(prepared=True,cases=len(cases),output=str(out),new_training_steps=0)),flush=True)


def audit(out):
    out=out.resolve();m=read(out/'manifest.json');protocol=read(out/'validation_protocol.json')
    assert sha(__file__)==m['sources']['tools/validate_frozen_v2_geometry.py']==protocol['source_sha256']
    parent=Path(protocol['candidate']);pm=read(parent/'manifest.json');new=load_run(out);old=load_run(parent)
    assert sha(out/'validation_protocol.json')==m['validation_protocol_sha256']
    assert sha(parent/'reference_chain_audit.json')==protocol['candidate_chain_audit_sha256']==m['candidate_chain_audit_sha256']
    assert sha(parent/'manifest.json')==protocol['candidate_manifest_sha256']==m['parent_manifest_sha256']
    assert sha(out/'geometry_protocol.json')==protocol['geometry_protocol_sha256']
    assert read(out/'geometry_protocol.json')['validation_cases']==new['cases']==protocol['cases']
    assert new['checkpoint_sha256']==old['checkpoint_sha256']==protocol['candidate_checkpoint_sha256']
    assert new['reference_sha256']==old['reference_sha256']==protocol['candidate_reference_sha256']
    for name in ('controller_modifier','adaptive_reference_command','closure_action_bias','normal_bias_y_deg',
                 'force_history_memory','geometry_mode','suffix_start_seconds','policy_adapter'):
        assert m[name]==pm[name],name
    assert sha(out/'prefix_policy.zip')==m['prefix_checkpoint_sha256']==pm['prefix_checkpoint_sha256']
    for rel,digest in m['sources'].items():
        assert sha(out/'source_snapshot'/rel)==digest
        if rel.replace('\\','/').startswith('training/'):assert sha(ROOT/rel)==digest
    reports={}
    for name in ('independent_candidate_audit.json','controller_modifier_audit_v2.json','adaptive_reference_audit.json','physical_v2_audit.json'):
        report=read(out/name)
        if 'passed' in report:assert report['passed'],name
        if 'summary_sha256' in report:assert report['summary_sha256']==sha(out/'summary.json')
        reports[name]=sha(out/name)
    rows=read(out/'summary.json')['after'];prefixes=[]
    for case in m['evaluation_cases']:
        record=next(r for r in rows if r['case']==case)
        prefix=out/'prefix'/f'{case}.json'
        if record['time_s']<70.2-1e-7:
            prefixes.append(dict(case=case,passed=False,reason='physical_run_terminated_before_prefix_boundary'))
        elif prefix.exists():
            row=read(prefix)
            assert row['case']==case and row['prefix_steps']==3510
            assert row['passed']==all(row['checks'].values())
            assert row['trajectory_sha256']==sha(out/'prefix'/f'{case}.npz')
            assert row['full_trajectory_sha256']==sha(out/f'after_{case}.npz')
            prefixes.append(row)
        else:prefixes.append(dict(case=case,passed=False,reason='prefix_record_missing'))
    physical=read(out/'physical_v2_audit.json')
    assert set(r['case'] for r in physical['rows'])==set(m['evaluation_cases'])
    # A faithfully recorded early physical failure is a valid negative result,
    # but does not qualify for complete-prefix/eight-cycle promotion.
    prefix_integrity=all(r['passed'] or r.get('reason')=='physical_run_terminated_before_prefix_boundary' for r in prefixes)
    passed=prefix_integrity
    eligible=passed and all(r['passed'] for r in prefixes) and physical['physical_v2_count']==len(rows)
    result=dict(passed=passed,eligible_single_cycle_reference=eligible,validation_cases=len(rows),
        physical_v2_count=physical['physical_v2_count'],original_strict_count=physical['original_strict_count'],
        prefix_replay_count=sum(r['passed'] for r in prefixes),prefix_rows=prefixes,rows=physical['rows'],
        audits=reports,manifest_sha256=sha(out/'manifest.json'),summary_sha256=sha(out/'summary.json'),
        validation_protocol_sha256=sha(out/'validation_protocol.json'),source_sha256=sha(__file__),
        no_training=True,policy_sha256=new['checkpoint_sha256'],reference_sha256=new['reference_sha256'],automatic_promotion=False,
        scope=protocol['case_role']+' One frozen candidate. Failed geometries remain in the denominator. This report audits geometry transfer, not planner reconstruction or eight-cycle climbing.')
    write(out/'reference_chain_audit.json',result);(out/'validation_audit_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k not in ('prefix_rows','rows','audits')}),flush=True)
    return result


def run(out,workers):
    out=out.resolve()
    assert sha(__file__)==read(out/'manifest.json')['sources']['tools/validate_frozen_v2_geometry.py']
    evaluate(out,workers=workers);audit_single(out)
    records=read(out/'summary.json')['after'];eligible=[r['case'] for r in records if r['time_s']>=70.2-1e-7]
    (out/'prefix').mkdir(exist_ok=True)
    def prefix(case):
        with (out/'prefix'/f'{case}.log').open('w',encoding='utf8') as stream:
            result=subprocess.run([sys.executable,'-m','tools.validate_swing_reference','worker','--out',str(out),
                '--kind','prefix','--case',case],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        print(json.dumps(dict(case=case,prefix_process_returncode=result.returncode)),flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(prefix,eligible))
    audit(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','run','audit'))
    p.add_argument('--candidate',type=Path);p.add_argument('--geometry-protocol',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,choices=(1,2,4),default=4)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.candidate,a.geometry_protocol,a.output)
    elif a.command=='run':run(a.output,a.workers)
    else:audit(a.output)
