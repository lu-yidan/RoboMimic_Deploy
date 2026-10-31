"""Merge only complete, compatible, hash-verified landing-window reset caches."""
import argparse
import json
from pathlib import Path
import shutil
from tools.compare_audited_closure_candidates import load_run,sha


def main():
 p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--parts',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 verified=load_run(a.parent);manifest=json.loads((a.parent/'manifest.json').read_text());cases=manifest['evaluation_cases']
 entries={};training={};provenance=[];common=None
 for source in a.parts:
  cp=json.loads((source/'protocol.json').read_text());cs=json.loads((source/'summary.json').read_text())
  assert cs['passed'] and cs['protocol_sha256']==sha(source/'protocol.json')
  signature=tuple(cp[k] for k in ('parent_checkpoint_sha256','reference_sha256','start_seconds','horizon_seconds','python_version'))
  if common is None:common=signature
  assert signature==common and cp['start_seconds']==70.2 and cp['reference_sha256']==manifest['reference_sha256']
  assert cp['parent_checkpoint_sha256']==manifest['prefix_checkpoint_sha256']
  assert {r['case'] for r in cs['rows']}==set(cp['cases']) and len(cs['rows'])==len(cp['cases'])
  for row in cs['rows']:
   name=row['case'];assert row['passed'] and name not in entries and cp['cases'][name]==cases[name]
   assert sha(source/f'{name}.pkl.gz')==row['cache_sha256'];entries[name]=(source,row)
  for name,digest in cp['sources'].items():
   if name.replace('\\','/').startswith('training/'):
    assert sha(name)==digest
    assert name not in training or training[name]==digest
    training[name]=digest
  provenance.append(dict(directory=str(source.resolve()),protocol_sha256=sha(source/'protocol.json'),summary_sha256=sha(source/'summary.json')))
 assert set(entries)==set(cases)
 a.output.mkdir(exist_ok=False)
 for name in cases:
  source,row=entries[name];shutil.copy2(source/f'{name}.pkl.gz',a.output/f'{name}.pkl.gz');assert sha(a.output/f'{name}.pkl.gz')==row['cache_sha256']
 protocol=dict(parent=str(a.parent.resolve()),parent_audit_sha256=verified['audit_sha256'],parent_checkpoint_sha256=common[0],reference=manifest['reference'],reference_sha256=common[1],
  start_seconds=common[2],horizon_seconds=common[3],python_version=common[4],sources=training,cases=cases,parts=provenance,source_sha256=sha(__file__),solver_cache_restarted=True,
  scope='All eight declared landing-window reset caches at 70.2 seconds. Verified parent prefixes only; no trained policy or complete-cycle evaluation shortcut.')
 (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
 result=dict(passed=True,rows=[entries[name][1] for name in cases],protocol_sha256=sha(a.output/'protocol.json'),source_sha256=sha(__file__))
 (a.output/'summary.json').write_text(json.dumps(result,indent=2));(a.output/'merge_source.py').write_bytes(Path(__file__).read_bytes())
 print(json.dumps(dict(passed=True,cases=len(cases))))

if __name__=='__main__':main()
