"""Regress a frozen landing actor on already-observed twelve- or sixteen-case suites."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from tools.compare_audited_closure_candidates import load_run,sha
from tools.evaluate_closure_foreaft_reference import evaluate_parallel
from tools.audit_fullcycle_candidate_only import audit
from tools.audit_closure_flexible_prefix_v2 import audit as audit_prefix
from tools.audit_trained_closure_controller_v2 import audit as audit_fixed
from tools.audit_closure_foreaft_reference import audit as audit_added


def main():
 p=argparse.ArgumentParser()
 for name in ('candidate','grid-parent','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--workers',type=int,choices=(1,2,4),default=4)
 a=p.parse_args();root=Path(__file__).resolve().parents[1]
 candidate=load_run(a.candidate);grid=load_run(a.grid_parent)
 m=json.loads((a.candidate/'manifest.json').read_text());old=json.loads((a.grid_parent/'summary.json').read_text())
 assert candidate['reference_sha256']==grid['reference_sha256']
 gm=json.loads((a.grid_parent/'manifest.json').read_text())
 assert m['prefix_checkpoint_sha256']==gm['prefix_checkpoint_sha256']
 assert m['suffix_start_seconds']==70.2 and m.get('require_inactive_case_exact_parent') is False
 assert candidate['controller_modifier']==grid['controller_modifier'] and candidate['adaptive_reference_command']==grid['adaptive_reference_command']
 known=['foot_z_plus15','foot_z_plus12.5','foot_z_plus10','foot_z_plus7.5','foot_z_plus5']
 assert all(k in candidate['rows'] for k in known)
 assert len(grid['cases']) in (12,16)
 if len(grid['cases'])==12:assert all(r['success'] for r in grid['rows'].values())
 for name in ('adaptive_reference_audit.json','controller_modifier_audit_v2.json'):
  report=json.loads((a.candidate/name).read_text())
  assert report['passed'] and report['manifest_sha256']==sha(a.candidate/'manifest.json') and report['summary_sha256']==sha(a.candidate/'summary.json')
 a.output.mkdir(exist_ok=False);shutil.copytree(a.candidate/'source_snapshot',a.output/'source_snapshot')
 shutil.copy2(a.candidate/'policy.zip',a.output/'policy.zip');shutil.copy2(m['prefix_checkpoint'],a.output/'prefix_policy.zip')
 sources=dict(m['sources'])
 for module in list(sys.modules.values()):
  file=getattr(module,'__file__',None)
  if not file:continue
  file=Path(file).resolve()
  if file.suffix!='.py' or not file.is_relative_to(root):continue
  relative=file.relative_to(root)
  if relative.parts[0] not in ('training','tools'):continue
  dest=a.output/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(file.read_bytes());sources[str(relative)]=sha(file)
 m.update(sources=sources,evaluation_cases=grid['cases'],parent_experiment=str(a.grid_parent.resolve()),parent_manifest_sha256=sha(a.grid_parent/'manifest.json'),
  candidate_experiment=str(a.candidate.resolve()),candidate_manifest_sha256=sha(a.candidate/'manifest.json'),
  prefix_checkpoint=str((a.output/'prefix_policy.zip').resolve()),training_cases={},evaluation_only=True,new_training_steps=0,
  scope='Trained landing-window actor on an already-observed development geometry suite, with unchanged fixed reference modifiers. Full time-zero regression with exact frozen prefix through 70.2 seconds. Failures are reported without suppressing the evaluation; no automatic promotion or independent generalization claim.')
 (a.output/'manifest.json').write_text(json.dumps(m,indent=2))
 s=dict(checkpoint_sha256=sha(a.output/'policy.zip'),evaluation_only=True,new_training_steps=0,evaluation_pending=True,before=old['after'])
 (a.output/'summary.json').write_text(json.dumps(s,indent=2))
 rows=evaluate_parallel(a.output,a.output/'parallel_after',None,workers=a.workers)['trajectories']
 for row in rows:
  name=f"after_{row['case']}.npz";shutil.copy2(a.output/'parallel_after'/name,a.output/name)
 s.update(after=rows,evaluation_pending=False)
 (a.output/'after.json').write_text(json.dumps(rows,indent=2));(a.output/'summary.json').write_text(json.dumps(s,indent=2))
 audit(a.output);audit_prefix(a.output);audit_fixed(a.output);audit_added(a.output)
 result=dict(successes=sum(r['success'] for r in rows),cases=len(rows),lost=[r['case'] for r in rows if grid['rows'][r['case']]['success'] and not r['success']],gained=[r['case'] for r in rows if not grid['rows'][r['case']]['success'] and r['success']],automatic_promotion=False)
 (a.output/'regression.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
