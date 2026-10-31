"""Prepare the declared hook grid using the existing single-cycle audit chain.

All metadata edits happen before any rollout. The original protocol and both
preparation sources are retained; the adopted experiment is never modified.
"""
import argparse
import json
from pathlib import Path
import shutil
from tools import validate_frozen_v2_geometry as validation
from tools.validate_swing_reference import ROOT, read, write, sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--geometry-output',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    original=args.protocol.resolve();declared=read(original)
    assert declared['status']=='protocol_only_no_new_grid_rollouts'
    candidate=Path(declared['candidate']);parent=read(candidate/'manifest.json')
    assert sha(candidate/'manifest.json')==declared['candidate_manifest_sha256']
    assert sha(candidate/'policy.zip')==declared['checkpoint_sha256']
    assert parent['reference_sha256']==declared['reference_sha256']
    assert sha(original.parent/'protocol_source.py')==declared['source_sha256']
    cases=declared['cases'];assert len(cases)==16
    for offsets in cases.values():
        assert offsets['rung_01']==parent['evaluation_cases']['training_12']['rung_01']
        assert set(offsets)=={'rung_01','rung_06'}
        assert offsets['rung_06'][1]==0 and max(map(abs,offsets['rung_06']))<=.002
    geometry=args.geometry_output.resolve();out=args.output.resolve()
    assert not geometry.exists() and not out.exists()
    geometry.mkdir(parents=True)
    write(geometry/'protocol.json',dict(training_cases=parent['evaluation_cases'],validation_cases=cases,
        original_protocol=str(original),original_protocol_sha256=sha(original),
        role='Newly declared local hand-target development grid; no pristine test-set claim',
        source_sha256=sha(__file__),new_training_steps=0))
    (geometry/'preparation_source.py').write_bytes(Path(__file__).read_bytes())
    validation.prepare(candidate,geometry/'protocol.json',out)
    # The reusable audit tool initially labels its original local validation
    # population. Replace that label prospectively, before any worker starts.
    protocol=read(out/'validation_protocol.json');manifest=read(out/'manifest.json')
    assert read(out/'summary.json')['evaluation_pending']
    assert not (out/'eval').exists() and not (out/'prefix').exists()
    shutil.copy2(original,out/'hook_uncertainty_protocol.json')
    role='Prospectively declared rung_06 X/Z local development grid around T12. Previous exposure is not established; not an independent random-wall test.'
    protocol.update(case_role=role,hook_uncertainty_protocol_sha256=sha(original),
                    preparation_source_sha256=sha(__file__))
    write(out/'validation_protocol.json',protocol)
    relative='tools/prepare_hook_geometry_validation.py'
    target=out/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(Path(__file__),target);manifest['sources'][relative]=sha(target)
    manifest.update(case_role=role,validation_protocol_sha256=sha(out/'validation_protocol.json'),
        hook_uncertainty_protocol_sha256=sha(original),
        scope=role+' Single-cycle screening and independent same-reference prefix audit only. The second-cycle hand-target intervention has not been executed. No training or automatic baseline promotion.')
    write(out/'manifest.json',manifest)
    write(out/'preparation_audit.json',dict(passed=True,rollouts_started=False,cases=len(cases),
        manifest_sha256=sha(out/'manifest.json'),validation_protocol_sha256=sha(out/'validation_protocol.json'),
        original_hook_protocol_sha256=sha(original),geometry_protocol_sha256=sha(geometry/'protocol.json'),
        source_sha256=sha(__file__),new_training_steps=0,
        scope='Pre-execution protocol conversion and label correction only; no physical result.'))
    print(json.dumps(dict(prepared=True,output=str(out),cases=len(cases),rollouts_started=False)),flush=True)


if __name__=='__main__':main()
