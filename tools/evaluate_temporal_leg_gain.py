"""Predeclared two-gain calibration on five known height cases."""
import json
import os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from tools.evaluate_reference_transfer import run_case
from tools.continue_vector_full_cycle_training import sha


def main():
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    output=Path('outputs/temporal_leg_gain_calibration_v1');output.mkdir(exist_ok=False)
    plan='outputs/highrod_endpoint_adapt_plan_v1/plan.npz'
    jobs=[(str(output),f'kp{kp:g}_height{h:g}',plan,-3.,f'foot_z_plus{h:g}',{'rung_01':[0.,0.,h/1000]},kp)
        for kp in (200.,300.) for h in (12.5,15.,10.,5.,7.5)]
    protocol=dict(jobs=jobs,source_sha256=sha(__file__),reference_sha256=sha(plan),
        selection='Accept only if at least one audited failure repaired and no successful height regresses. If both qualify choose lower gain. Keep all outcomes.',
        scope='Known-case controller calibration, not generalization or training. Compare with audited zero-action kp100 baseline. Fixed57..65s envelope and unchanged force/torque/native limits.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2));(output/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    results=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run_case,j) for j in jobs]):
            r=future.result();results.append(r);(output/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(r),flush=True)


if __name__=='__main__':main()
