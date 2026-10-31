"""Create disjoint worker cache views without duplicating all eight native templates."""
import argparse
import json
import os
from pathlib import Path
import shutil
from tools.compare_audited_closure_candidates import sha


def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 cp=json.loads((a.cache/'protocol.json').read_text());cs=json.loads((a.cache/'summary.json').read_text())
 assert cs['passed'] and cs['protocol_sha256']==sha(a.cache/'protocol.json')
 names=list(cp['cases']);assert len(names)==8
 records={r['case']:r for r in cs['rows']};assert set(records)==set(names)
 for case in names:assert records[case]['passed'] and sha(a.cache/f'{case}.pkl.gz')==records[case]['cache_sha256']
 a.output.mkdir(exist_ok=False);rows=[]
 for rank in range(4):
  cases=names[rank::4];dest=a.output/f'worker_{rank}';dest.mkdir();methods={}
  for case in cases:
   source=a.cache/f'{case}.pkl.gz';target=dest/source.name
   try:os.link(source,target);methods[case]='hardlink_immutable_hash_checked_cache'
   except OSError:shutil.copy2(source,target);methods[case]='byte_copy'
   assert sha(target)==records[case]['cache_sha256']
  protocol=dict(cp,cases={k:cp['cases'][k] for k in cases},source_cache=str(a.cache.resolve()),source_cache_summary_sha256=sha(a.cache/'summary.json'),worker_index=rank,
   sampling_scope='Two disjoint cases per worker; balance within each pair. Four workers jointly cover all eight cases. This changes worker-level case correlations.')
  (dest/'protocol.json').write_text(json.dumps(protocol,indent=2));summary=dict(passed=True,rows=[records[k] for k in cases],protocol_sha256=sha(dest/'protocol.json'),source_sha256=sha(__file__))
  (dest/'summary.json').write_text(json.dumps(summary,indent=2));rows.append(dict(worker=rank,cases=cases,summary_sha256=sha(dest/'summary.json'),storage=methods))
 result=dict(passed=True,workers=rows,parent_summary_sha256=sha(a.cache/'summary.json'),source_sha256=sha(__file__),
  scope='Integrity-verified cache partition, not another dynamics validation. Reduces resident templates from 32 to 8 across four workers.')
 (a.output/'summary.json').write_text(json.dumps(result,indent=2));(a.output/'partition_source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(rows))

if __name__=='__main__':main()
