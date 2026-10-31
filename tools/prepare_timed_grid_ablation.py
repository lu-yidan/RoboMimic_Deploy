"""Predeclare the third arm without changing the existing paired experiment."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
from training.timed_hook_loading import PILOT_DELAYS
from tools.validate_swing_reference import read,write,sha,ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--paired',type=Path,required=True)
    p.add_argument('--schedule-check',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();paired=a.paired.resolve();pair=read(paired/'protocol.json')
    single=Path(pair['single_cycle_screen']);check=read(a.schedule_check/'schedule_check.json')
    assert check['passed'] and check['delays_seconds']==PILOT_DELAYS
    assert check['controller_source_sha256']==sha(ROOT/'training/timed_hook_loading.py')
    assert not (single/'eval').exists() and not (paired/'runs').exists(), 'Declare this comparator before new-grid outcomes'
    assert len(pair['cases'])==16
    a.output.mkdir(parents=True,exist_ok=False)
    sources={}
    for relative in ('training/timed_hook_loading.py','training/probe_hook_loading.py',
                     'tools/prepare_timed_grid_ablation.py','tools/check_timed_hook_pilot.py'):
        target=a.output/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target);sources[relative]=sha(target)
    write(a.output/'protocol.json',dict(status='declared_only_no_grid_runner_or_results',
        created_at=datetime.now().astimezone().isoformat(),paired_experiment=str(paired),
        paired_protocol_sha256=sha(paired/'protocol.json'),single_cycle_screen=str(single),
        single_manifest_sha256=sha(single/'manifest.json'),cases=pair['cases'],
        first_two_arms=pair['arms'],third_arm='timed_open_loop',delays_seconds=PILOT_DELAYS,
        probe_force_n=15.,ramp_seconds=.5,contact_feedback_used=False,new_training_steps=0,
        schedule_check_sha256=sha(a.schedule_check/'schedule_check.json'),sources=sources,
        development_exposure='The delays are calibrated on the already observed T12 pilot. This protocol was saved before any rollout on the new 16-case rung_06 grid. Not a pristine holdout claim.',
        prerequisite='A fresh T12 timed run must pass its physical/policy/timing audit and reproduce the nominal contact-probe pilot before implementing/running the compiled-scene third arm.',
        denominator='All 16 cases. Single-cycle failures count against all arms; do not require either existing second-cycle arm to succeed before the timed run.',
        frozen_controls='Same complete scene compiler, reference, policy, motor limits, QP, initial force cap and release curve. Only the later-hook release trigger uses fixed delays instead of contact feedback.',
        termination_difference='The feedback arm can stop on a 1.5 s confirmation timeout. The open-loop arm has no confirmation state or equivalent sensor timeout. Both retain identical physical termination and endpoint acceptance. Report timeout separately; do not describe this as only a sensor-noise ablation.',
        primary='Physical-v2 complete two-cycle success on all 16 declared geometries; all errors and termination reasons retained.',
        secondary=['Complete paired-cycle acceleration and contact-force peaks for all four support groups',
                   'Whole-cycle root acceleration RMS and squared-torque control-effort proxy',
                   'Endpoint closure quality and support margins'],
        comparison='Reuse the already frozen two-arm records by hash, without rerunning or selecting better attempts. Report failed/truncated runs separately from complete-cycle peak comparisons.',
        feedback_privilege='The feedback arm receives ideal simulated forces AND the true external contact geometry identity. This is not evidence of a deployable force-only rod-recognition system.',
        automatic_promotion=False))
    print(str(a.output/'protocol.json'),flush=True)


if __name__=='__main__':main()
