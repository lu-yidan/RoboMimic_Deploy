"""Enforce frozen known-case eligibility before dispatching a grid evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from tools.continue_vector_full_cycle_training import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['control','penalty'],required=True);args=parser.parse_args()
    root=Path('outputs/closure_policy_validation_protocol_v1');protocol=json.loads((root/'protocol.json').read_text())
    p=Path(f'outputs/closure_margin_{args.arm}98304_v1');assert p in [Path(value) for value in protocol['candidates']]
    m=json.loads((p/'manifest.json').read_text());s=json.loads((p/'summary.json').read_text());a=json.loads((p/'independent_training_audit.json').read_text())
    assert a['passed'] and not s['evaluation_pending'] and s['additional_steps']==98304 and m['parent_checkpoint_sha256']==protocol['parent_checkpoint_sha256'] and m['protocol_sha256']==protocol['training_protocol_sha256']
    for file,key in [('policy.zip','checkpoint_sha256'),('manifest.json','experiment_manifest_sha256'),('summary.json','experiment_summary_sha256')]:assert sha(p/file)==a[key]
    successes={r['case'] for r in a['trajectories'] if r['label']=='after' and r['independently_accepted']};lost=sorted(set(protocol['parent_known_successes'])-successes)
    assert sha('outputs/support_reference_validation_v1/protocol.json')==protocol['grid_protocol_sha256']
    output=Path(f'outputs/closure_{args.arm}_policy_validation_v1');record=dict(arm=args.arm,candidate=str(p),eligible=not lost,lost_known_successes=lost,known_successes=sorted(successes),policy_sha256=sha(p/'policy.zip'),audit_sha256=sha(p/'independent_training_audit.json'),selection_protocol_sha256=sha(root/'protocol.json'),source_sha256=sha(__file__))
    dispatch=root/f'{args.arm}_dispatch.json'
    if dispatch.exists():raise FileExistsError(dispatch)
    dispatch.write_text(json.dumps(record,indent=2));(root/f'{args.arm}_dispatch_source.py').write_bytes(Path(__file__).read_bytes())
    if lost:print(json.dumps(record));return
    subprocess.run([sys.executable,'-m','tools.evaluate_reference_policy_transfer','--parent',str(p),'--zero-suite','outputs/support_reference_validation_v1','--output',str(output)],check=True)
    (output/'validation_dispatch.json').write_bytes(dispatch.read_bytes())


if __name__=='__main__':main()
