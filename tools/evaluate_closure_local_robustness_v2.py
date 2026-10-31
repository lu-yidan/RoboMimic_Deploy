"""Predeclared local robustness comparison, with no adaptation on its results."""
import argparse
import json
from pathlib import Path
import shutil
from tools.continue_vector_full_cycle_training import sha
from tools.evaluate_closure_waist_body import evaluate_parallel
from tools.audit_fullcycle_candidate_only import audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    parent=json.loads((a.parent/'manifest.json').read_text())
    verified=json.loads((a.parent/'independent_training_audit.json').read_text())
    assert verified['passed'] and verified['experiment_manifest_sha256']==sha(a.parent/'manifest.json') and verified['experiment_summary_sha256']==sha(a.parent/'summary.json')
    assert sum(r['independently_accepted'] for r in verified['trajectories'] if r['label']=='after')==5
    modifier=json.loads((a.parent/'controller_modifier_audit.json').read_text());assert modifier['passed'] and modifier['manifest_sha256']==sha(a.parent/'manifest.json')
    old=Path(parent['parent_experiment']);old_audit=json.loads((old/'independent_training_audit.json').read_text())
    assert old_audit['passed'] and old_audit['checkpoint_sha256']==sha(old/'policy.zip')
    jobs=[('z14.9',[0.,0.,.0149]),('z15.1',[0.,0.,.0151]),('x-0.5_z15',[-.0005,0.,.015]),('x+0.5_z15',[.0005,0.,.015]),('x-0.1_z15',[-.0001,0.,.015]),('x+0.1_z15',[.0001,0.,.015])]
    cases={name:{'rung_01':offset} for name,offset in jobs}
    a.output.mkdir(exist_ok=False)
    protocol=dict(cases=cases,parent_manifest_sha256=sha(a.parent/'manifest.json'),parent_policy_sha256=sha(a.parent/'policy.zip'),original_policy_sha256=sha(old/'policy.zip'),
        scope='Predeclared local perturbations of the previously tuned highest case. No retuning in this run; not broad independent generalization evidence. Original task gates.',source_sha256=sha(__file__))
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    results={}
    for variant in ('original','modified'):
        out=a.output/variant;out.mkdir()
        m=dict(parent);m['evaluation_cases']=cases;m['training_cases']={};m['scope']=protocol['scope'];m['robustness_variant']=variant
        m['prefix_checkpoint']=str((out/'prefix_policy.zip').resolve())
        shutil.copy2(old/'policy.zip',out/'prefix_policy.zip')
        policy=old/'policy.zip' if variant=='original' else a.parent/'policy.zip'
        shutil.copy2(policy,out/'policy.zip');shutil.copytree(a.parent/'source_snapshot',out/'source_snapshot')
        m['sources']=dict(m['sources'])
        for name in ('tools/evaluate_closure_waist_body.py','tools/audit_fullcycle_candidate_only.py','tools/evaluate_closure_local_robustness_v2.py'):
            dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(name).read_bytes());m['sources'][name]=sha(name)
        if variant=='original':
            m['controller_modifier']=dict(waist_bias=[0.,0.,0.],body_drop_m=0.,waist_ramp_seconds=1.,geometry_gain='clip((raw_obs[138]-.625)/.125,0,1)')
            for k in ('seed_bias','seed_ramp_seconds','seed_gate_sha256','selection_summary_sha256'):m.pop(k,None)
        (out/'manifest.json').write_text(json.dumps(m,indent=2))
        s=dict(checkpoint_sha256=sha(out/'policy.zip'),evaluation_pending=True,evaluation_only=True,training_additional_steps=0)
        (out/'summary.json').write_text(json.dumps(s,indent=2))
        reused={};provenance=[]
        if variant=='original':
            partial=Path('outputs/closure_waist_body_robustness_20261024_v1/original')
            previous=json.loads((partial/'manifest.json').read_text())
            previous_summary=json.loads((partial/'summary.json').read_text())
            assert previous_summary['checkpoint_sha256']==sha(out/'policy.zip')
            for field in ('reference_sha256','prefix_checkpoint_sha256','controller_modifier','normal_bias_y_deg','force_history_memory','geometry_mode'):
                assert previous[field]==m[field]
            for case in cases:
                rpath=partial/'parallel_after'/case/'result.json'
                if not rpath.exists():continue
                assert previous['evaluation_cases'][case]==cases[case]
                row=json.loads(rpath.read_text());name=f'after_{case}.npz';file=rpath.parent/name
                assert sha(file)==row['trajectory_sha256']
                reused[case]=row;shutil.copy2(file,out/name)
                provenance.append(dict(case=case,source=str(file),trajectory_sha256=sha(file),original_manifest_sha256=sha(partial/'manifest.json')))
        missing=[case for case in cases if case not in reused]
        fresh=evaluate_parallel(out,out/'parallel_after',missing,workers=2)['trajectories'] if missing else []
        for row in fresh:
            name=f"after_{row['case']}.npz";shutil.copy2(out/'parallel_after'/name,out/name);reused[row['case']]=row
        rows=[reused[case] for case in cases]
        (out/'reuse_provenance.json').write_text(json.dumps(provenance,indent=2))
        s.update(after=rows,evaluation_pending=False)
        (out/'after.json').write_text(json.dumps(rows,indent=2));(out/'summary.json').write_text(json.dumps(s,indent=2))
        audit(out)
        independent=json.loads((out/'independent_candidate_audit.json').read_text())
        results[variant]={r['case']:r['independently_accepted'] for r in independent['trajectories']}
    paired=[dict(case=name,original=results['original'][name],modified=results['modified'][name]) for name in cases]
    report=dict(completed=True,rows=paired,success_counts={k:sum(v.values()) for k,v in results.items()},protocol_sha256=sha(a.output/'protocol.json'),scope=protocol['scope'])
    (a.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()

