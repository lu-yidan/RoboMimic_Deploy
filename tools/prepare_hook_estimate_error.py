"""Declare an observation-error follow-up before inspecting the hand-rod grid."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
from tools.validate_swing_reference import ROOT,read,write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--grid',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    grid=a.grid.resolve();d=read(grid/'protocol.json');single=Path(d['single_cycle_screen'])
    assert len(d['cases'])==16 and not (single/'eval').exists()
    a.output.mkdir(parents=True,exist_ok=False);sources={}
    for relative in ('training/estimated_bar_observation.py','training/cycle_relative_bar_observation.py',
                     'training/compiled_rung_geometry.py','tools/prepare_hook_estimate_error.py'):
        target=a.output/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target);sources[relative]=sha(target)
    write(a.output/'protocol.json',dict(status='declared_only_no_followup_runner_or_results',
        created_at=datetime.now().astimezone().isoformat(),known_geometry_grid=str(grid),
        known_grid_protocol_sha256=sha(grid/'protocol.json'),cases=d['cases'],
        estimated_offsets={'rung_06':[0.,0.,0.]},unlisted_estimates='oracle geometry unchanged',
        arms=['adopted_v2','contact_confirmed','timed_open_loop'],
        frozen_policy=True,new_training_steps=0,automatic_promotion=False,sources=sources,
        prerequisite='Review the complete predeclared known-geometry three-arm grid first. This protocol does not start another batch.',
        first_cycle='Fresh single-cycle, independent prefix and policy audits under the ESTIMATED observation contract. Existing oracle first-cycle gates cannot be reused as proof.',
        denominator='All 16 geometries for each arm. First-cycle failure counts against all three arms; never gate an arm on another arm second-cycle success.',
        mechanism='Actual rung_06 is translated by the unchanged frozen XYZ case. Only its normalized geometry feature is reported as zero displacement, following its cycle-relative role. All other original feedback and simulated state inputs remain unchanged.',
        interpretation='A deterministic, local geometry-estimate bias study. Not camera perception, realistic sensor error, fully nonprivileged control, unseen geometry, or a contact-identity estimator.',
        leakage_boundary='Contact-confirmed arm still reads ideal forces and true contact geometry identity; original 136-dimensional state/reference/contact observations remain. Hiding one offset feature does not remove all privileged information.',
        comparison='Reuse the completed oracle-grid records by hash. New estimated-input records get distinct manifests, independent audits and explicit observation_error labels; never pass through the baseline replay or oracle policy auditor unchanged.',
        primary='Complete physical-v2 two-cycle success over the full 16-case denominator.',
        secondary=['All four support-force peaks over complete recorded cycles','Root acceleration peak and RMS',
                   'Squared-torque effort proxy','Endpoint closure and support margins','Confirmation delay and termination reason'],
        tuning='No threshold, reference, policy, force-cap or timing changes based on this grid. Later tuning requires a separately named experiment.'))
    print(str(a.output/'protocol.json'))


if __name__=='__main__':main()
