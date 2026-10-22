"""Recover the collector after successful physics writes; never rerun physics."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'outputs/window_memory_fullcycles_v1'
protocol=json.loads((out/'protocol.json').read_text());rows=[]
for memory in protocol['memory_options']:
    for case,_ in protocol['cases']:
        name=('memory__' if memory else 'raw__')+case;run=out/name
        summary=json.loads((run/'summary.json').read_text())
        checks={file:json.loads((run/file).read_text())['passed'] for file in ['independent_audit.json','joint_limit_audit.json']}
        rows.append(dict(case=case,memory_enabled=memory,run=name,passed=bool(summary['passed'] and all(checks.values())),reason=summary['reason'],time_s=summary['simulation_time_s'],checks=checks,minimum_native_joint_margin_rad=summary['minimum_native_joint_margin_rad']))
if (out/'results.json').exists():raise FileExistsError('Do not overwrite an earlier collection')
(out/'results.json').write_text(json.dumps(rows,indent=2))
(out/'collection_recovery.json').write_text(json.dumps(dict(reason='run_climb_cycle writes summary.json and returnsNone. Collector previously indexed the return after physics and independent audits completed.',physics_rerun=False,all_six_saved_summaries_and_audits_loaded=True,original_collector_sha256=hashlib.sha256((out/'runner_before_result_fix.py').read_bytes()).hexdigest(),recovery_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
print(json.dumps(rows,indent=2))
