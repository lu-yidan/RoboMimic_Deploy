"""Freeze a later-hook geometry grid before evaluating its outcomes.

This only declares an experiment. It does not certify new geometries, modify
the adopted baseline, start training, or claim an independent unseen test set.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from tools.validate_swing_reference import read, write, sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--pilot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    candidate=args.candidate.resolve();pilot=args.pilot.resolve()
    manifest=read(candidate/'manifest.json');chain=read(candidate/'reference_chain_audit.json')
    trial=read(pilot/'manifest.json');audit=read(pilot/'hook_admission_audit.json')
    assert chain['passed'] and chain['manifest_sha256']==sha(candidate/'manifest.json')
    assert audit['passed'] and audit['manifest_sha256']==sha(pilot/'manifest.json')
    assert audit['trajectory_sha256']==sha(pilot/'trajectory.npz')
    assert Path(trial['parent_experiment']).resolve()==candidate and trial['cycles']==2
    assert trial['case']=='training_12'
    assert trial['checkpoint_sha256']==sha(candidate/'policy.zip')
    center=deepcopy(manifest['evaluation_cases']['training_12'])
    assert set(center)=={'rung_01'}
    # Four levels include both extremes and interiors. The nominal center is
    # the already observed T12 pilot, explicitly kept outside this new grid.
    levels=[-.002,-.002/3,.002/3,.002]
    cases={}
    for i,x in enumerate(levels):
        for j,z in enumerate(levels):
            offsets=deepcopy(center);offsets['rung_06']=[x,0.,z]
            cases[f'hook_grid_x{i}_z{j}']=offsets
    assert len(cases)==len({json.dumps(v,sort_keys=True) for v in cases.values()})==16
    assert all(v not in manifest['evaluation_cases'].values() for v in cases.values())
    result=dict(status='protocol_only_no_new_grid_rollouts',cases=cases,
        candidate=str(candidate),candidate_manifest_sha256=sha(candidate/'manifest.json'),
        candidate_chain_audit_sha256=sha(candidate/'reference_chain_audit.json'),
        reference_sha256=manifest['reference_sha256'],checkpoint_sha256=sha(candidate/'policy.zip'),
        pilot=str(pilot),pilot_audit_sha256=sha(pilot/'hook_admission_audit.json'),
        controller_parameters=trial['contact_admission']['parameters'],
        physical_geometry='Static translations of rung_06 and its mounts, X/Z within +/-2 mm, Y unchanged. The earlier rung_01 remains the T12 pilot geometry.',
        observation='Same oracle geometry role inputs; no visual estimate or sensor-error claim.',
        reference='Unchanged nominal reference. No retargeting or per-case controller tuning.',
        arms=['adopted_v2','contact_confirmed_loading_after_first_cycle'],
        stage_order=['Fresh single-cycle and independent prefix/policy audit for every declared geometry.',
                     'For each successful first cycle, continuous two-cycle baseline and candidate from physical time zero.',
                     'Save 2000 Hz telemetry for both arms; retain failed or timed-out cases in all denominators.',
                     'Do not promote a controller or extend to eight cycles until the full paired grid has been reviewed.'],
        primary_outcomes=['physical-v2 two-cycle success','confirmation timeout or runtime termination reason'],
        secondary_outcomes=['whole-cycle peak root acceleration','whole-cycle maximum joint acceleration',
                            'whole-cycle peak net force for all four support groups','endpoint closure quality',
                            'confirmation delay and minimum endpoint normal/vertical support'],
        interpretation='Prospectively frozen local robustness grid after a successful nominal pilot. Historical exposure of these geometries is not established. This is not a pristine blind test, random short-bar wall, learned perception experiment, hardware result, or proof that feedback is necessary.',
        missing_ablation='A matched time-based loading ramp is required before attributing robustness specifically to contact feedback instead of reduced initial loading.',
        selection_policy='Keep all 16 cases. No retry-based cherry-picking, threshold relaxation or parameter search on this grid. Any later tuning starts a separately named development experiment.',
        new_training_steps=0,source_sha256=sha(__file__))
    args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'protocol.json',result)
    (args.output/'protocol_source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(prepared=True,cases=16,rollouts_started=False,output=str(args.output))),flush=True)


if __name__=='__main__':main()
