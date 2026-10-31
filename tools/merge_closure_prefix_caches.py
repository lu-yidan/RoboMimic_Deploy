"""Merge hash-verified reset caches sharing the identical frozen prefix actor."""
import argparse,json,shutil
from pathlib import Path
from tools.continue_vector_full_cycle_training import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--extra',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    protocols=[];summaries=[]
    for root in (a.base,a.extra):
        protocol=json.loads((root/'protocol.json').read_text());summary=json.loads((root/'summary.json').read_text())
        assert summary['passed'] and summary['protocol_sha256']==sha(root/'protocol.json') and all(r['passed'] for r in summary['rows'])
        protocols.append(protocol);summaries.append(summary)
    for key in ('parent_checkpoint_sha256','reference_sha256','start_seconds','horizon_seconds'):assert protocols[0][key]==protocols[1][key]
    assert not set(protocols[0]['cases'])&set(protocols[1]['cases'])
    a.output.mkdir(exist_ok=False);sources={};cases={};rows=[];provenance=[]
    for root,protocol,summary in zip((a.base,a.extra),protocols,summaries):
        for path,value in protocol['sources'].items():
            if path in sources:assert sources[path]==value
            if path.replace('\\','/').startswith('training/'):assert sha(path)==value
            sources[path]=value
        cases.update(protocol['cases'])
        for row in summary['rows']:
            name=f"{row['case']}.pkl.gz";assert sha(root/name)==row['cache_sha256'];shutil.copy2(root/name,a.output/name)
            copied=dict(row,source_cache_directory=str(root.resolve()),source_cache_summary_sha256=sha(root/'summary.json'))
            rows.append(copied);(a.output/f"{row['case']}.json").write_text(json.dumps(copied,indent=2))
        provenance.append(dict(source=str(root.resolve()),protocol_sha256=sha(root/'protocol.json'),summary_sha256=sha(root/'summary.json')))
    sources[str(Path(__file__).resolve())]=sha(__file__)
    protocol=dict(protocols[0],cases=cases,sources=sources,provenance=provenance,
        scope='Verified actual continuous prefixes, merged for training resets only. Same prefix actor; five known cases and three additional X/Z perturbations. No mid-episode state replacement.')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2));result=dict(passed=True,rows=rows,protocol_sha256=sha(a.output/'protocol.json'),source_sha256=sha(__file__))
    (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(passed=True,cases=list(cases))),flush=True)

if __name__=='__main__':main()
