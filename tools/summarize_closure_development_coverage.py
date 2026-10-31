"""Deduplicate declared development geometry cases across audited suites."""
import argparse
import csv
import json
from pathlib import Path
from tools.compare_audited_closure_candidates import load_run,sha

GROUPS={
 'fixed_search_seed':['closure_eightcase_seed_baseline_20261024_v1','closure_combined_margin_grid_20261024_v1'],
 'fivecase_ppo':['closure_fivecase_trained_eightcase_eval_20261024_v1','closure_combined_curriculum_grid_20261024_v1'],
 'eightcase_ppo':['closure_eightcase_curriculum98304_20261024_v1']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 models={};rows=[]
 for label,names in GROUPS.items():
  by_geometry={};runs=[load_run(Path('outputs')/name) for name in names]
  for run in runs:
   assert run['checkpoint_sha256']==runs[0]['checkpoint_sha256']
   assert run['reference_sha256']==runs[0]['reference_sha256']
   assert run['controller_modifier']==runs[0]['controller_modifier'] and run['adaptive_reference_command']==runs[0]['adaptive_reference_command']
   for case,offsets in run['cases'].items():
    key=json.dumps({k:v for k,v in offsets.items() if any(v)},sort_keys=True)
    row=run['rows'][case]
    if key in by_geometry:
     assert by_geometry[key]['success']==row['success']
     assert abs(by_geometry[key]['closure_mrad']-row['closure_mrad'])<1e-7
    else:
     entry=dict(model=label,case=case,geometry=key,**row,source=run['directory'],audit_sha256=run['audit_sha256'])
     by_geometry[key]=entry;rows.append(entry)
  models[label]=dict(unique_cases=len(by_geometry),successes=sum(r['success'] for r in by_geometry.values()),suites=runs)
 assert models['fixed_search_seed']['unique_cases']==models['fivecase_ppo']['unique_cases']==20
 assert models['eightcase_ppo']['unique_cases']==8
 a.output.mkdir(exist_ok=False)
 report=dict(models=models,source_sha256=sha(__file__),scope='Deduplicated known development evaluations, not independent test samples. No binomial confidence interval or new generalization claim. Eight-case PPO has no claimed twelve-case grid result.')
 (a.output/'summary.json').write_text(json.dumps(report,indent=2))
 with (a.output/'cases.csv').open('w',newline='',encoding='utf-8-sig') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 (a.output/'report_source.py').write_bytes(Path(__file__).read_bytes())
 print(json.dumps({k:{field:v[field] for field in ('unique_cases','successes')} for k,v in models.items()}))

if __name__=='__main__':main()
