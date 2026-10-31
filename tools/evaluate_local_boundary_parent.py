"""Predeclare a small, previously untuned neighborhood and evaluate the old actor."""
import argparse,copy,json,shutil,sys
from pathlib import Path
from tools.compare_audited_closure_candidates import load_run,sha
from tools.evaluate_closure_foreaft_reference import evaluate_parallel
from tools.audit_fullcycle_candidate_only import audit
from tools.audit_trained_closure_controller_v2 import audit as audit_fixed
from tools.audit_closure_foreaft_reference import audit as audit_reference


def main():
    p=argparse.ArgumentParser()
    for n in ('parent','search','protocol'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args();parent=load_run(a.parent)
    search=json.loads((a.search/'summary.json').read_text());sp=json.loads((a.search/'protocol.json').read_text())
    assert search['completed'] and search['best']['successes']==4
    assert search['protocol_sha256']==sha(a.search/'protocol.json') and sp['parent_audit_sha256']==parent['audit_sha256']
    for f,h in sp['sources'].items():assert sha(f)==h
    center=parent['cases']['boundary_07']
    offsets=[(-.00025,-.000025),(-.00025,.000025),(.00025,-.000025),(.00025,.000025),
             (0.,-.00005),(0.,.00005),(-.0005,0.),(.0005,0.)]
    cases={}
    for i,(dx,dz) in enumerate(offsets):
        case=copy.deepcopy(center);case['rung_01'][0]+=dx;case['rung_01'][2]+=dz
        assert case not in parent['cases'].values();cases[f'local_{i:02d}']=case
    protocol=dict(parent=str(a.parent.resolve()),parent_audit_sha256=parent['audit_sha256'],
        candidate_search=str(a.search.resolve()),candidate_search_sha256=sha(a.search/'summary.json'),
        candidate_selection=search['best'],cases=cases,offsets_x_z_m=offsets,source_sha256=sha(__file__),
        scope='Eight predefined local geometries around known boundary_07, evaluated after fixing the candidate. No fitting on these outcomes; not broad random-geometry generalization. Both policies retain original acceptance and physical limits.')
    protocol=json.loads(json.dumps(protocol))
    if a.protocol.exists():assert json.loads((a.protocol/'protocol.json').read_text())==protocol
    else:
        a.protocol.mkdir();(a.protocol/'protocol.json').write_text(json.dumps(protocol,indent=2));(a.protocol/'source.py').write_bytes(Path(__file__).read_bytes())
    if a.output is None:
        print(json.dumps(dict(prepared=True,cases=cases,candidate_fixed=True)),flush=True);return
    root=Path(__file__).resolve().parents[1];m=json.loads((a.parent/'manifest.json').read_text())
    assert not m.get('policy_adapter')
    a.output.mkdir(exist_ok=False);shutil.copytree(a.parent/'source_snapshot',a.output/'source_snapshot')
    shutil.copy2(a.parent/'policy.zip',a.output/'policy.zip');shutil.copy2(m['prefix_checkpoint'],a.output/'prefix_policy.zip')
    sources=dict(m['sources'])
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if not f:continue
        f=Path(f).resolve()
        if f.suffix!='.py' or not f.is_relative_to(root):continue
        relative=f.relative_to(root)
        if relative.parts[0] not in ('tools','training'):continue
        destination=a.output/'source_snapshot'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(f.read_bytes());sources[str(relative)]=sha(f)
    m.update(sources=sources,parent_experiment=str(a.parent.resolve()),parent_manifest_sha256=sha(a.parent/'manifest.json'),
        prefix_checkpoint=str((a.output/'prefix_policy.zip').resolve()),evaluation_cases=cases,training_cases={},
        evaluation_only=True,new_training_steps=0,require_inactive_case_exact_parent=False,
        neighborhood_protocol_sha256=sha(a.protocol/'protocol.json'),scope=protocol['scope']+' Old-policy comparison trajectories; the new geometries have no prior saved prefix trajectory. Independent per-step action replay still verifies the composite policy.')
    (a.output/'manifest.json').write_text(json.dumps(m,indent=2))
    s=dict(evaluation_only=True,new_training_steps=0,evaluation_pending=True,checkpoint_sha256=sha(a.output/'policy.zip'))
    (a.output/'summary.json').write_text(json.dumps(s,indent=2))
    rows=evaluate_parallel(a.output,a.output/'parallel_after',None,workers=4)['trajectories']
    for row in rows:shutil.copy2(a.output/'parallel_after'/f"after_{row['case']}.npz",a.output/f"after_{row['case']}.npz")
    s.update(evaluation_pending=False,after=rows);(a.output/'after.json').write_text(json.dumps(rows,indent=2));(a.output/'summary.json').write_text(json.dumps(s,indent=2))
    audit(a.output);audit_fixed(a.output);audit_reference(a.output)
    verified=load_run(a.output)
    print(json.dumps(dict(completed=True,successes=sum(r['success'] for r in verified['rows'].values()),cases=len(rows))),flush=True)


if __name__=='__main__':main()
